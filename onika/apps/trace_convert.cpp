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
#include <onika/trace/trace_output_format.h>
#include <onika/trace/dot_trace_format.h>
#include <onika/trace/vite_trace_format.h>
#include <onika/trace/yaml_trace_format.h>
#include <onika/trace/tikz_trace_format.h>

#include <yaml-cpp/yaml.h>
#include <iostream>
#include <string>
#include <random>

struct YamlDouble
{
  double m_value;
  inline operator double () const { return m_value; }
};
namespace YAML
{
  template<> struct convert<YamlDouble>
  {
    static inline bool decode(const Node& node, YamlDouble& v)
    {
      if( ! node.IsScalar() ) return false;
      char * send = nullptr;
      v.m_value = std::strtod( node.as<std::string>().c_str() , &send );
      return true;
    }
  };
}


int main(int argc,char*argv[])
{
  if( argc < 4 )
  {
    std::cerr<<"Usage: "<<argv[0]<<" vite|dot|plot [+idle] <yaml-input-file> <output-file>"<<std::endl;
    return 1;
  }

  std::string fmt = "yaml";
  std::string input_file = "";
  std::string output_file = "";
  std::string start_evt = "";
  ssize_t start_evt_idx = 0;
  double filter_start_time = std::numeric_limits<double>::lowest();
  std::string end_evt = "";
  ssize_t end_evt_idx = 0;
  double filter_end_time = std::numeric_limits<double>::max();
  bool output_trace = true;
  bool output_idle = true;
  bool shift_to_zero = false;
  std::string padstate = "";
  int thstart = 0;
  int thend = 65536;

  enum StateColorization
  {
    COLOR_UNCHANGED = 0,
    COLOR_APPCTX = 1,
    COLOR_TAG = 2
  };

  StateColorization coloring = COLOR_UNCHANGED;

  for(int argi = 1;argi<argc;argi++)
  {
    std::string arg = argv[argi];
    if( arg == "-vite" ) fmt="vite";
    else if( arg == "-dot" ) fmt="dot";
    else if( arg == "-plot" ) fmt="plot";
    else if( arg == "-yaml" ) fmt="yaml";
    else if( arg == "-tikz" ) fmt="tikz";
    else if( arg == "-notrace" ) output_trace=false;
    else if( arg == "-noidle" ) output_idle=false;
    else if( arg == "-color" )
    {
      std::string c = argv[++argi];
      if(c=="ctx") coloring = COLOR_APPCTX;
      else if(c=="tag") coloring = COLOR_TAG;
    }
    else if( arg == "-start" )
    {
      start_evt = argv[++argi];
      start_evt_idx = std::atol(argv[++argi]);
    }
    else if( arg == "-end" )
    {
      end_evt = argv[++argi];
      end_evt_idx = std::atol(argv[++argi]);
    }
    else if( arg == "-shift" )
    {
      shift_to_zero = true;
    }
    else if( arg == "-pad" )
    {
      padstate = argv[++argi];
    }
    else if( arg == "-threads" )
    {
      thstart = std::atol(argv[++argi]);
      thend = std::atol(argv[++argi]);
    }
    else if( input_file.empty() ) input_file = std::move(arg);
    else output_file = std::move(arg);
  }

  onika::trace::TraceOutputFormat* otf = nullptr;
  if( fmt== "vite" ) otf = new onika::trace::ViteOutputFormat();
  else if( fmt == "dot" ) otf = new onika::trace::DotTraceFormat();
  else if( fmt == "plot" ) { otf = new onika::trace::TraceOutputFormat(); }
  else if( fmt == "yaml" ) { otf = new onika::trace::YAMLTraceFormat(); }
  else if( fmt == "tikz" ) { otf = new onika::trace::TikzTraceFormat(); }
  else { std::cerr<<"Unknown output format '"<<fmt<<"'"<<std::endl; }

  auto data = YAML::LoadFile( input_file );
  otf->open( output_file );

  if( data["idle"] )
  {
    auto value = data["idle"];
    filter_start_time = value["start"].as<YamlDouble>();
    filter_end_time = value["end"].as<YamlDouble>();
    std::cout<<"idle block found : trace interval = [ "<<filter_start_time<<" ; "<<filter_end_time<<" ]\n";
  }

  ssize_t start_evt_count = 0;
  ssize_t end_evt_count = 0;
  if( data["trace"] )
  {
    size_t ntokens = data["trace"].size();
    std::cout<<"trace block found : "<<ntokens<<" \n";
    size_t progress=0;
    for(auto p:data["trace"])
    {
      if( p.IsMap() )
      {      
        std::string k = p.begin()->first.as<std::string>();
        auto value = p.begin()->second;
        if( k == "declare_state" )
        {
          std::string idname = value["short"].as<std::string>();
          std::string fullname = value["full"].as<std::string>();
          if( fullname == start_evt ) { start_evt = idname; }
          if( fullname == end_evt ) { end_evt = idname; }
        }
        else  if( k == "declare_threads" )
        {
          double start = value["start"].as<YamlDouble>();
          double end = value["end"].as<YamlDouble>();
          int n = value["threads"].as<int>();
          thstart = std::max( 0 , thstart );
          thend = std::min( thend , n );
          filter_start_time = std::max( filter_start_time , start );
          filter_end_time = std::min( filter_end_time , end );
        }
      }
      if( (++progress)%1000 == 0 ) std::cout<<"preprocessing stage 1 : " << (progress*100)/ntokens <<"%    \r"<<std::flush;
    }

    std::cout<<std::endl;
    progress=0;

    for(auto p:data["trace"])
    {
      if( p.IsMap() )
      {      
        std::string k = p.begin()->first.as<std::string>();
        if( k == "state" )
        {
          auto value = p.begin()->second;
          auto s = value["st"].as<std::string>();
          if( s == start_evt ) ++ start_evt_count;
          if( s == end_evt ) ++ end_evt_count;
        }
      }
      if( (++progress)%1000 == 0 ) std::cout<<"preprocessing stage 2 : " << (progress*100)/ntokens <<"%    \r"<<std::flush;
    }
    if( start_evt_idx < 0 ) start_evt_idx += start_evt_count;
    if( end_evt_idx < 0 ) end_evt_idx += end_evt_count;
    // std::cout<<start_evt<<" idx = "<<start_evt_idx<<std::endl;
    // std::cout<<end_evt<<" idx = "<<end_evt_idx<<std::endl;

    std::cout<<std::endl;
    progress=0;

    start_evt_count = 0;
    end_evt_count = 0;
    for(auto p:data["trace"])
    {
      if( p.IsMap() )
      {      
        std::string k = p.begin()->first.as<std::string>();
        if( k == "state" )
        {
          auto value = p.begin()->second;
          auto s = value["st"].as<std::string>();
          double time = value["ti"].as<YamlDouble>();
          if( s == start_evt )
          {
            if( start_evt_count == start_evt_idx ) filter_start_time = time;
            ++ start_evt_count;
          }
          if( s == end_evt )
          {
            if( end_evt_count == end_evt_idx ) filter_end_time = time;
            ++ end_evt_count;
          }
        }
      }

      if( (++progress)%1000 == 0 ) std::cout<<"preprocessing stage 3 : " << (progress*100)/ntokens <<"%    \r"<<std::flush;
    }

  }

  std::cout<<"\nconvert '"<<input_file<<"' to '"<<output_file<<"'\n"
           <<"\tformat='"<<fmt<<"'\n"
           <<"\ttrace="<<std::boolalpha<<output_trace<<"\n"
           <<"\tidle="<<std::boolalpha<<output_idle<<"\n"
           <<"\tstart_evt='"<<start_evt<<"'\n"
           <<"\tend_evt='"<<end_evt<<"'\n"
           <<"\tstart_idx="<<start_evt_idx<<"\n"
           <<"\tend_idx="<<end_evt_idx<<"\n"
           <<"\tfilter_start = "<<filter_start_time<<"\n"
           <<"\tfilter_end = "<<filter_end_time<<"\n"
           <<"\tstart_evt_count = "<<start_evt_count<<"\n"
           <<"\tend_evt_count = "<<end_evt_count<<"\n"
           <<"\tthstart = "<<thstart<<"\n"
           <<"\tthend = "<<thend<<"\n"
           <<"\tpadstate = '"<<padstate<<"'\n";

  double time_shift = 0.0;
  if( shift_to_zero )
  {
    time_shift = -filter_start_time;
  }

  if( data["trace"] && output_trace )
  {
    //std::cout<<"parse trace"<<std::endl<<std::flush;
    std::cout << "parsing trace ..." << std::endl;
#   define LOG_TRACE_PARSER std::cout << "threads: "<<nthreads<<" states: "<<nstates<<" events: "<<nevts<<"                \r"<<std::flush
    size_t nthreads = 0;
    size_t nstates = 0;
    size_t nevts = 0;
    size_t ntokens = 0;
    std::vector<double> thread_cur_time;
    
    std::unordered_map<std::string,onika::RGBColord> colormap;
    std::uniform_real_distribution<> rndcol(0.3,1.0);
    std::mt19937_64 re {0};
    auto gencolor = [&re,&rndcol,&colormap](const std::string& s) -> onika::RGBColord
    {
      auto it = colormap.find(s);
      if(it!=colormap.end()) return it->second;
      return colormap[s] = { rndcol(re) , rndcol(re) , rndcol(re) } ;
    };
            
    otf->start_trace();
    for(auto p:data["trace"])
    {
      if( p.IsMap() )
      {      
        std::string k = p.begin()->first.as<std::string>();
        auto value = p.begin()->second;
        //std::cout<<"parse "<<k<< std::endl<<std::flush;
        if( ! value.IsMap() ) { std::cerr<<k<<" value must be a dictionray"<<std::endl; std::abort(); }
        if( k == "declare_state" )
        {
          std::string idname = value["short"].as<std::string>();
          std::string fullname = value["full"].as<std::string>();
          auto rgb = value["color"].as< std::vector<int> >();
          onika::RGBColord color{rgb[0]/255.0,rgb[1]/255.0,rgb[2]/255.0};
          if( coloring != COLOR_UNCHANGED && idname!= "IDL" )
          {
            auto sep = fullname.find('@');
            if( coloring == COLOR_APPCTX ) color = gencolor( fullname.substr(0,sep) );
            else if( coloring == COLOR_TAG ) color = gencolor( fullname.substr(sep+1) );
          }
          //std::cout<<"declare_state("<<idname<<','<<fullname<<','<<onika::format_color_www(color)<<")\n" << std::flush;
          otf->declare_state( idname , fullname , color );
          ++ nstates;
        }
        else  if( k == "declare_threads" )
        {
          //int n = value["threads"].as<int>();
          nthreads = thend - thstart;
          thread_cur_time.assign( nthreads , filter_start_time+time_shift );
          otf->declare_threads( nthreads , filter_start_time+time_shift , filter_end_time+time_shift );
        }
        else  if( k == "state" )
        {
          int t = value["th"].as<int>();
          if( t >= thstart && t < thend )
          {
            t -= thstart;
            std::string state = value["st"].as<std::string>();
            double time = value["ti"].as<YamlDouble>();
            if( time >= filter_start_time && time <= filter_end_time )
            {
              //std::cout<<"set_state("<<t<<','<<state<<','<<time<<")\n" << std::flush;
              if( (time+time_shift) < thread_cur_time[t] ) { std::cerr<<"Warning, time goes backward on thread "<<t<<", event "<<state<<std::endl; }
              otf->set_state( t , state , time+time_shift );
              thread_cur_time[t] = std::max( thread_cur_time[t] , time+time_shift );
              ++ nevts;
            }
          }
        }
        else
        {
          std::cerr<<"Bad trace entry '"<<k<<"'"<<std::endl;
          std::abort();
        }
        ++ ntokens;
        if( ntokens % 4096 ) { LOG_TRACE_PARSER; }
      }
      else
      {
        std::cerr<<"Bad File format"<<std::endl;
        std::abort();
      }
    }
    if( ! padstate.empty() )
    {
      for(size_t t=0;t<nthreads;t++)
      {
        if( thread_cur_time[t] < (filter_end_time+time_shift) )
        {
          otf->set_state( t , padstate , filter_end_time+time_shift );
        }
      }
    }
    otf->finalize_trace(32,0.0);
    LOG_TRACE_PARSER;
#   undef LOG_TRACE_PARSER
    std::cout<<std::endl<<"done"<<std::endl;
  }
  
  if( data["idle"] && output_idle )
  {
    std::cout<<"parsing idle ..."<<std::flush;
    auto value = data["idle"];
    double start = value["start"].as<YamlDouble>();
    double end = value["end"].as<YamlDouble>();
    
    std::vector<YamlDouble> _values =  value["values"].as< std::vector<YamlDouble> >();
    size_t n = _values.size();
    std::vector<double> values;
    for(size_t i=0;i<n;i++)
    {
      double x = start + i * ( end - start ) / n;
      if( x >= filter_start_time && x < filter_end_time ) values.push_back(_values[i].m_value);
    }
    
    otf->add_idle_plot(values,filter_start_time+time_shift,filter_end_time+time_shift);
    std::cout<<" done"<<std::endl;
  }
  
  otf->close();
}

