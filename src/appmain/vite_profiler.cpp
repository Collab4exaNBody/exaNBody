/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/
#include <exanb/core/log.h>
#include <exanb/core/thread.h>

#include <onika/omp/ompt_task_timing.h>

#include <iostream>
#include <numeric>
#include <vector>
#include <list>
#include <iomanip>
#include <unordered_map>
#include <random>

#include <sched.h>

#include "vite_profiler.h"
#define _XSTAMP_VITE_VERBOSE 1

using namespace exanb;

static constexpr size_t g_vite_trace_chunk_size = 1048576;

// output file
static onika::trace::TraceOutputFormat* g_vite_output = nullptr;
static ViteLabelFunction g_vite_label;
static ViteColoringFunction g_vite_color;
static ViteFilterFunction g_vite_filter;

// profiling trace
std::chrono::nanoseconds g_vite_trace_start_time {0};
std::chrono::nanoseconds g_vite_trace_min_duration {0};
std::chrono::nanoseconds g_vite_trace_max_duration {0};

static std::list< std::vector<ViteTraceElement> > g_vite_trace;
static std::mutex g_vite_trace_mutex;

const std::string ViteTraceElement::task_label() const
{
  return g_vite_label( *this );
}

struct ViteTraceThreadContext
{
  std::thread::id m_tid;
  ssize_t m_omp_tid = -1;
  std::vector<ViteTraceElement> * m_vite_trace = nullptr;

  void add_trace_record( const ViteTraceElement& t )
  {
    if( m_vite_trace == nullptr || m_vite_trace->size() >= g_vite_trace_chunk_size )
    {
      {
        std::scoped_lock lock( g_vite_trace_mutex );
        m_vite_trace = & g_vite_trace.emplace_back();
      }
      m_vite_trace->reserve( g_vite_trace_chunk_size );
    }
    m_vite_trace->push_back( t );
  }
};

using ThreadCtxMap = std::unordered_map< std::thread::id , ViteTraceThreadContext* >;
static ThreadCtxMap g_vite_static_thread_ctx; // lock free
static ThreadCtxMap g_vite_dynamic_thread_ctx; // with lock

static inline ViteTraceThreadContext& vite_thread_ctx()
{
  auto tid = std::this_thread::get_id();
  auto it = g_vite_static_thread_ctx.find( tid );
  if( it != g_vite_static_thread_ctx.end() )
  {
    return *(it->second);
  }
  else
  {
    std::scoped_lock lock( g_vite_trace_mutex );
    auto it_dyn = g_vite_dynamic_thread_ctx.find( tid );
    if( it_dyn != g_vite_dynamic_thread_ctx.end() ) return *(it_dyn->second);
    auto [it_ins,inserted] = g_vite_dynamic_thread_ctx.insert( { tid , new ViteTraceThreadContext{tid} } );
    assert( inserted );
    return *(it_ins->second);
  }
}

void vite_process_event(const onika::omp::OpenMPToolTaskTiming& e)
{
  vite_thread_ctx().add_trace_record( ViteTraceElement{ e.ctx, e.tag, e.cpu_id, e.timepoint, e.end } );
}

// register calling thread int the static thread context map
static void add_static_thread_context()
{
  std::scoped_lock lock( g_vite_trace_mutex );
  auto tid = std::this_thread::get_id();
  if( g_vite_static_thread_ctx.find(tid) == g_vite_static_thread_ctx.end() )
  {
    //std::cout<<"vite_trace: register static thread "<<tid<<std::endl;
    g_vite_static_thread_ctx.insert( { tid , new ViteTraceThreadContext{ tid , omp_get_thread_num() } } );
  }
}

void vite_start_trace(
  const xsv2ConfigStruct_trace&,
  onika::trace::TraceOutputFormat* output ,
  const ViteLabelFunction& label,
  const ViteColoringFunction& color ,
  const ViteFilterFunction& filter )
{
# pragma omp parallel
  {
    add_static_thread_context();
    
#   pragma omp task
    { add_static_thread_context(); }
    
#   pragma omp taskwait
  }

  // user defined functions
  g_vite_label = label;
  g_vite_color = color;
  g_vite_filter = filter;

  // open file to write trace to
  g_vite_output = output;

  // set start time reference
  g_vite_trace_start_time = wall_clock_time();
}


void vite_end_trace(const xsv2ConfigStruct_trace& trace)
{
  const bool total_time_output = trace.total;
  const bool idle_plot_output = trace.idle;
  const std::string& trigger_marker = trace.trigger;
  const long trigger_start_count = trace.trigger_interval.size()>=1 ? trace.trigger_interval[0] : 0;
  const long trigger_end_count = trace.trigger_interval.size()>=2 ? trace.trigger_interval[1] : -1;

  // future parameters
  //const bool idle_plot = true;
  const size_t idle_resolution = trace.idle_resolution; //4096;
  const ssize_t idle_smoothing = trace.idle_smoothing; //3;

  //std::ofstream vite_output( g_vite_output_filename );
  g_vite_output->open( trace.file );

  // time renormalization
  std::chrono::nanoseconds trace_min_time {0};
  for(const auto& ev : g_vite_trace ) for(const auto& e : ev) 
  {
    if( trace_min_time.count()==0 ) trace_min_time = e.start;
    else if( e.start < trace_min_time ) trace_min_time = e.start;
  }
  std::chrono::nanoseconds trace_max_time {0};
  int64_t trace_duration_mean = 0;
  int64_t trace_duration_min = std::numeric_limits<int64_t>::max();
  int64_t trace_duration_max = 0;
  size_t n_events = 0;
  for(auto& ev : g_vite_trace ) for(auto& e : ev)
  {
    e.start -= trace_min_time;
    e.end -= trace_min_time;
    if( e.end > trace_max_time ) trace_max_time = e.end;
    auto d = e.end.count() - e.start.count();
    trace_duration_min = std::min( trace_duration_min , d );
    trace_duration_max = std::max( trace_duration_max , d );
    trace_duration_mean += d;
    ++ n_events;
  }
  if( n_events > 0 ) { trace_duration_mean /= n_events; }
  else { trace_duration_mean = 0.0; }
  int64_t trace_duration_variance = 0;
  for(auto& ev : g_vite_trace ) for(auto& e : ev)
  {
    int64_t d = e.end.count() - e.start.count();
    trace_duration_variance += std::abs( d - trace_duration_mean );
  }
  if( n_events > 0 ) { trace_duration_variance /= n_events; }
  else { trace_duration_variance = 0.0; }
  g_vite_trace_max_duration = std::chrono::nanoseconds{ /*trace_duration_min*/ trace_duration_mean + (trace_duration_variance*3)/2 };
  g_vite_trace_min_duration = std::chrono::nanoseconds{ /*trace_duration_max*/ std::max( 0l , trace_duration_mean - (trace_duration_variance*3)/2 ) };
  g_vite_trace_start_time = std::chrono::nanoseconds{0};
  
  // ----------------
  
  // count number of utilized system threads
  std::unordered_map< ssize_t , size_t > thread_num;
  size_t n_threads = 0;
  for( const auto& ev : g_vite_trace ) for( const auto& e : ev ) 
  {
    if( thread_num.find(e.rsc_id) == thread_num.end() )
    {
      thread_num[e.rsc_id] = n_threads++;
    }
  }

  // ******* sanity checks and event sort ***********
  std::vector<double> idle_values;
  std::vector<ViteTraceElement> all_trace_events;
  double start_trigger_time = 0.0;
  double end_trigger_time = 0.0;
  double event_duration_max = 0.0;
  {
    std::unordered_map< ssize_t , std::vector<ViteTraceElement> > per_thread_trace;
    for( const auto& ev : g_vite_trace ) for( const auto& e : ev ) 
    {
      per_thread_trace[e.rsc_id].push_back( e );
    }
    g_vite_trace.clear();
    
    double trace_time_max = 0.0;
    
    for(auto& p:per_thread_trace)
    {
      std::sort( p.second.begin() , p.second.end() , [](const auto &a, const auto &b)->bool { return a.start < b.start; } );
      std::string prev_evt_label;
      std::chrono::nanoseconds t { 0 }; //= g_vite_trace_start_time;
      double prev_evt_start=0.0;
      double prev_evt_end=0.0;
      ViteTraceElement* prev_evt = nullptr;
      for(auto& e : p.second)
      {
        std::chrono::duration<double,std::milli> start = e.start;
        std::chrono::duration<double,std::milli> end = e.end;
        if( e.start < t )
        {
          std::cerr<<"Warning: event '"<<e.task_label()<<"' ("<<start.count()<<"->"<<end.count()<<") overlaps '"<<prev_evt_label<<"' ("<<prev_evt_start<<"->"<<prev_evt_end<<") on thread "<<e.rsc_id<<std::endl;
          if( prev_evt != nullptr )
          {
            std::cerr<<"Shorten event '"<<prev_evt_label<<"'"<<std::endl;
            prev_evt->end = e.start;
          }
        }
        if( e.end < e.start )
        {
          std::cerr<<"Warning: event '"<<e.task_label()<<"' starts ("<<start.count()<<") after its ending ("<<end.count()<<") on thread "<<e.rsc_id<<std::endl;
        }
        t = std::max( std::max( t , e.start ) , e.end );
        prev_evt_label = e.task_label();
        prev_evt_start = start.count();
        prev_evt_end = end.count();
        prev_evt = &e;
        trace_time_max = std::max( trace_time_max , std::max( start.count() , end.count() ) );
        event_duration_max = std::max( event_duration_max , end.count() - start.count() );
      }
    }
#   ifdef _XSTAMP_VITE_VERBOSE
    std::cout << "vite_trace: found "<<n_threads<<" cpu threads"<<std::endl;
    std::cout << "vite_trace: "<<g_vite_static_thread_ctx.size()<<" static thread contexts"<<std::endl;
    std::cout << "vite_trace: "<<g_vite_dynamic_thread_ctx.size()<<" dynamic thread contexts"<<std::endl;
#   endif

    // copy and sort events in a unique container
    for(const auto& ev:per_thread_trace)
    {
      all_trace_events.insert( all_trace_events.end() , ev.second.begin() , ev.second.end() );
    }
    std::sort( all_trace_events.begin() , all_trace_events.end() , [](const auto &a, const auto &b)->bool { return a.start < b.start; } );
    bool start_trigger = trigger_marker.empty();
    bool end_trigger = trigger_marker.empty();
    int trigger_index = 0;
    for( const auto& e : all_trace_events ) 
    {
      if( !start_trigger || !end_trigger )
      {
        bool trigger_detected = ( e.task_label() == trigger_marker );
        if( !start_trigger && trigger_detected && trigger_index==trigger_start_count )
        {
          start_trigger = true;
          std::chrono::duration<double,std::milli> start = e.start;
          start_trigger_time = start.count();
          std::cout<<"start trigger '"<<trigger_marker<<"' #"<<trigger_start_count<<std::endl;
        }
        if( trigger_detected && trigger_index==trigger_end_count )
        {
          end_trigger = true;
          std::chrono::duration<double,std::milli> end = e.end;
          end_trigger_time = end.count();
          std::cout<<"end trigger '"<<trigger_marker<<"' #"<<trigger_end_count<<std::endl;
        }        
        if( trigger_detected ) ++ trigger_index;
      }
    }
    if( end_trigger_time == 0.0 )
    {
      std::chrono::duration<double,std::milli> em = trace_max_time;
      end_trigger_time = em.count();
    }
    std::cout<<"trigger selection filter = [ "<<start_trigger_time<<" ; "<<end_trigger_time<<" ]"<<std::endl;

    if( idle_plot_output )
    {
      std::vector<double> idle_sampling(idle_resolution);
      double scaling = idle_resolution / (end_trigger_time - start_trigger_time);
      for(auto& p:per_thread_trace)
      {
        double thread_last = 0.0;
        for(const auto& e : p.second)
        {
          std::chrono::duration<double,std::milli> start = e.start;
          std::chrono::duration<double,std::milli> end = e.end;
          if( start.count()>=start_trigger_time && end.count()<end_trigger_time)
          {
            double ns = std::clamp( (start.count()-start_trigger_time)*scaling , 0.0 , idle_resolution-1.0 );
            double ne = std::clamp( (end.count()-start_trigger_time)*scaling , 0.0 , idle_resolution-1.0 );
            if( ne>ns && ns>thread_last )
            {
              ssize_t idle_start = std::clamp( ssize_t(thread_last) , ssize_t(0) , ssize_t(idle_resolution-1) );
              ssize_t idle_end = std::clamp( ssize_t(ns) , ssize_t(0) , ssize_t(idle_resolution-1) );
              //double start_frac = thread_last - idle_start;
              //double end_frac = ns - idle_end;
              for(ssize_t i=idle_start; i<=idle_end; i++)
              {
                double q = 0.0;
                if(idle_start==idle_end) q = ns - thread_last;
                else if(i==idle_start)   q = 1.0 - ( thread_last - idle_start );
                else if(i==idle_end)     q = ns - idle_end;
                else                     q = 1.0;
                idle_sampling[i] += q;
              }
              thread_last = ne;
            }          
          }
        }
      }
      idle_values.resize(idle_resolution);
      for(ssize_t i=0;i<ssize_t(idle_resolution);i++)
      {
        double f = 0.0;
        for(ssize_t j=(i-idle_smoothing);j<=(i+idle_smoothing);j++) if(j>=0 && j<ssize_t(idle_resolution)) f+=idle_sampling[j];
        idle_values[i] = f/(n_threads*(2*idle_smoothing+1));
      }
      
    }
    
    per_thread_trace.clear();
  }
  // **************************************************

  g_vite_output->start_trace();
  g_vite_output->declare_threads( n_threads , 0.0 , end_trigger_time - start_trigger_time );
  g_vite_output->declare_state("IDL","Idle",{0.,0.,0.});

  // associate a different thread state for every task
  // and one different color for every operator the task comes from
  std::unordered_map< std::string , std::string > short_name_map;
  std::unordered_map< std::string , double > total_time_map;
  size_t short_name_counter = 1;
  for( const auto& e : all_trace_events ) 
  {

    int64_t t = e.end.count() - e.start.count();
    if( t < 0 ) t = 0;
    const std::string opname = e.task_label();
    std::string short_name;
    if( short_name_map.find(opname) == short_name_map.end() )
    {
      std::ostringstream oss;
      oss << "W" << short_name_counter++ ;
      short_name = oss.str();
      short_name_map[ opname ] = short_name;
      total_time_map[ opname ] = 0.0;
    }
    auto col = g_vite_color( e );
    if( ! short_name.empty() ) // new state found
    {
      g_vite_output->declare_state(short_name,opname,col);
    }
  }

  for(size_t i=0;i<n_threads;i++)
  {
    g_vite_output->set_state(i,"IDL",0.,0.0,event_duration_max);
  }

  double global_end_time = 0.0;
  for( const auto& e : all_trace_events ) 
  {
    std::chrono::duration<double,std::milli> time_start_dur = e.start;
    std::chrono::duration<double,std::milli> end = e.end;
    if(time_start_dur.count()>=start_trigger_time && end.count()<end_trigger_time )
    {
      std::chrono::duration<double,std::milli> time_end_dur = e.end;
      double time_start = time_start_dur.count() - start_trigger_time;
      double time_end = time_end_dur.count() - start_trigger_time;
      if( time_end < time_start )
      {
#       ifndef NDEBUG
        std::cerr << "Warning: incorrect start/end time: "<<time_start<<" / "<<time_end<<std::endl;
#       endif
        time_end = time_start;
      }
      if( time_end > time_start )
      {
        size_t tid = thread_num[ e.rsc_id ];
        const std::string opname = e.task_label();
        assert( short_name_map.find(opname) != short_name_map.end() );
        const std::string& task_name = short_name_map[opname];
        g_vite_output->set_state(tid,task_name,time_start , time_end - time_start , event_duration_max );
        total_time_map[ opname ] += time_end - time_start;
        g_vite_output->set_state(tid,"IDL",time_end , 0.0 , event_duration_max );
        global_end_time = std::max( global_end_time , time_end );
      }
    }
  }

  g_vite_output->finalize_trace(n_threads,global_end_time);

  if( idle_plot_output )
  {
    g_vite_output->add_idle_plot( idle_values , 0.0 , end_trigger_time - start_trigger_time );
  }

  if( total_time_output )
  {
    std::vector< std::pair<double,std::string> > total_times;
    for(const auto& p:total_time_map)
    {
      total_times.push_back( { p.second , p.first } );
    }
    std::sort( total_times.begin() , total_times.end() );
    g_vite_output->add_total_time( total_times );
  }

  g_vite_output->close();
}

// ------------------------ general customization functions ----------------------
ViteFilterFunction g_vite_default_filter = [](const ViteTraceElement&) -> bool { return true; };

struct ViteTagRandomColoring
{
  std::unordered_map<std::string,ViteEventColor> event_color;
  std::mt19937_64 re {0};
  inline ViteEventColor operator () (const ViteTraceElement& e)
  {
    using namespace exanb;
    std::uniform_real_distribution<> rndcol(0.3,1.0);
    std::string tag = ( (e.tag==nullptr) ? "null" : e.tag );
    auto op_it = event_color.find(tag);
    if( op_it == event_color.end() )
    {
      double r = rndcol(re);
      double g = rndcol(re);
      double b = rndcol(re);
      event_color[tag] = {r,g,b};
    }
    return event_color[tag];
  }
};
ViteColoringFunction g_vite_tag_rnd_color = ViteTagRandomColoring{};

