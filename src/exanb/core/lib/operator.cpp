#include <exanb/core/operator.h>
#include <exanb/core/type_utils.h>
#include <exanb/core/string_utils.h>
#include <exanb/core/thread.h>
#include <exanb/core/operator_task.h>

#include <onika/omp/ompt_interface.h>
#include <onika/omp/ompt_task_timing.h>
#include <onika/memory/memory_usage.h>
#include <onika/cuda/cuda_context.h>

#include <cassert>
#include <memory>
#include <chrono>
#include <cmath>
#include <functional>

namespace exanb
{

  void default_parallel_stats_func(const std::vector<double>& x, int& np, int& r, std::vector<double>& minval, std::vector<double>& maxval, std::vector<double>& avg)
  {
    np = 1;
    r = 0;
    minval = x;
    maxval = x;
    avg = x;
  }

  // ================================================================
  // ========================== OperatorNode ========================
  // ================================================================

  std::shared_ptr<onika::cuda::CudaContext> OperatorNode::s_global_cuda_ctx = nullptr;

  OperatorNode::TimeStampT OperatorNode::s_profiling_timestamp_ref;
  
  bool OperatorNode::s_global_profiling = false;
  bool OperatorNode::s_global_mem_profiling = false;
  bool OperatorNode::s_debug_execution = false;
  /*std::vector<ProfilingFunctionSet>*/ ProfilingFunctionSet OperatorNode::s_profiling_functions;
  size_t OperatorNode::s_global_instance_index = 0;


  void OperatorNode::finalize()
  {
    if( is_terminal() )
    {
      for( auto s : slots() )
      {
        if( s->is_private() )
        {
          //std::cout<<"free resource for private slot '"<< s->pathname() <<"'\n";
          s->free_resource();
        }
      }
    }
  }

  bool OperatorNode::is_looping() const
  {
    return false;
  }

  void OperatorNode::compile()
  {
    assert( ! compiled() );
#   ifndef NDEBUG
    for( const auto& s : in_slots() )
    {
      assert(s.second->owner()==this);
      assert(s.second->name()==s.first);
    }
    for( const auto& s : out_slots() ) 
    {
      assert(s.second->owner()==this);
      assert(s.second->name()==s.first);
    }
#   endif
    auto pn = pathname();
    m_tag = std::make_unique<char[]>( pn.size() + 1 );
    std::memcpy( m_tag.get() ,  pn.c_str() , pn.size() );
    m_tag[ pn.size() ] = '\0';
    m_compiled = true;
  }

  void OperatorNode::set_global_mem_profiling(bool prof)
  {
    s_global_mem_profiling = prof;
  }

  bool OperatorNode::global_mem_profiling()
  {
    return s_global_mem_profiling;
  }

  ssize_t OperatorNode::resident_memory_inc() const
  {
    return m_resident_mem_inc;
  }


  void OperatorNode::set_global_profiling(bool prof)
  {
    s_global_profiling = prof;
    if( s_global_profiling )
    {
      onika::omp::OpenMPToolInterace::set_task_start_callback( task_start_callback );
      onika::omp::OpenMPToolInterace::set_task_stop_callback( task_stop_callback );
    }
    else
    {
      onika::omp::OpenMPToolInterace::set_task_start_callback( nullptr );
      onika::omp::OpenMPToolInterace::set_task_stop_callback( nullptr );
    }
  }

  void OperatorNode::set_profiling(bool prof)
  {
    m_profiling = prof;
  }

  bool OperatorNode::profiling() const
  {
    return m_profiling;
  }
  
  bool OperatorNode::global_profiling()
  {
    return s_global_profiling;
  }

  void OperatorNode::set_debug_execution(bool yn)
  {
    s_debug_execution = yn;
  }

  bool OperatorNode::debug_execution()
  {
    return s_debug_execution;
  }

  std::set< std::pair<std::string,OperatorSlotBase*> > OperatorNode::named_slots() const
  {
    std::set< OperatorSlotBase* > added_slots;
    std::set< std::pair<std::string,OperatorSlotBase*> > all_slots;
    for( auto p : in_slots() )
    {
      all_slots.insert( { p.first , p.second } );
      added_slots.insert( p.second );
    }
    for( auto p : out_slots() )
    {
      if( added_slots.find(p.second) == added_slots.end() )
      {
        all_slots.insert( { p.first , p.second } );
        added_slots.insert( p.second );
      }
    }
    return all_slots;
  }

  int OperatorNode::in_slot_idx(const std::string& k) const
  {
    int n = in_slot_count();
    for(int i=0;i<n;i++)
    {
      if(m_in_slot_storage[i].first==k) return i;
    }
    return -1;
  }
  int OperatorNode::in_slot_idx(const OperatorSlotBase *s) const
  {
    int n = in_slot_count();
    for(int i=0;i<n;i++)
    {
      if(m_in_slot_storage[i].second==s) return i;
    }
    return -1;
  }
  bool OperatorNode::in_slot_rename(const std::string& before, const std::string& after)
  {
    int i = in_slot_idx(before);
    if(i==-1) { return false; }
    assert( in_slot_idx(after) == -1 );
    m_in_slot_storage[i].first = after;
    m_in_slot_storage[i].second->rename( after );
    return true;
  }

  int OperatorNode::out_slot_idx(const std::string& k) const
  {
    int n = out_slot_count();
    for(int i=0;i<n;i++)
    {
      if(m_out_slot_storage[i].first==k) return i;
    }
    return -1;
  }
  int OperatorNode::out_slot_idx(const OperatorSlotBase *s) const
  {
    int n = out_slot_count();
    for(int i=0;i<n;i++)
    {
      if(m_out_slot_storage[i].second==s) return i;
    }
    return -1;
  }
  bool OperatorNode::out_slot_rename(const std::string& before, const std::string& after)
  {
    int i = out_slot_idx(before);
    if(i==-1) { return false; }
    assert( out_slot_idx(after) == -1 );
    m_out_slot_storage[i].first = after;
    m_out_slot_storage[i].second->rename( after );
    return true;
  }
  
  const std::string& OperatorNode::name() const
  {
    return m_name;
  }
  
  void OperatorNode::set_name(const std::string& n)
  {
    m_name=n;
  }

  std::string OperatorNode::pathname() const
  {
    std::string p;
    if( m_parent != nullptr ) { p = m_parent->pathname() + "." ; }
    return p + name();
  }

  OperatorNodeFlavor OperatorNode::in_flavor() const
  {
    OperatorNodeFlavor f;
    for( const auto& s : in_slots() ) { if( s.second->is_input_connectable() ) { f[ s.first ] = s.second->value_type(); } }
    return f;
  }

  OperatorNodeFlavor OperatorNode::out_flavor() const
  {
    OperatorNodeFlavor f;
    for( const auto& s : out_slots() ) { if( s.second->is_output_connectable() ) { f[ s.first ] = s.second->value_type(); } }
    return f;
  }

  OperatorNodeFlavor OperatorNode::flavor() const
  {
    OperatorNodeFlavor f = in_flavor();
    f.merge( out_flavor() );
    return f;
  }

  // FIXME: is_terminal is clearly an indicator of a bad design :
  // should be useless if virtual methods are specialized to change the behavior of batch_node operators
  bool OperatorNode::is_terminal() const
  {
    return true;
  }

  void OperatorNode::apply_graph( std::function<void(OperatorNode*)> f , bool )
  {
    f( this );
  }

  // a terminal operator that cannot be removed even though it has no connected output
  bool OperatorNode::is_sink() const
  {
    return out_slot_count() == 0;
  }

  void OperatorNode::set_profiler( ProfilingFunctionSet profiler )
  {
    s_profiling_functions = profiler;
  }

  void OperatorNode::profile_task_start( const onika::omp::OpenMPToolTaskTiming & evt_info )
  {
    bool do_profiling = global_profiling() && profiling();
    if( do_profiling )
    {
      assert( evt_info.ctx == this );
      if( s_profiling_functions.task_start != nullptr )
      {
        //std::cout << "profile_task_start\n";
        s_profiling_functions.task_start( evt_info );
      }
    }
  }
  
  void OperatorNode::profile_task_stop( const onika::omp::OpenMPToolTaskTiming & evt_info )
  {
    bool do_profiling = global_profiling() && profiling();
    if( do_profiling )
    {
      assert( evt_info.ctx == this );
      if( s_profiling_functions.task_stop != nullptr )
      {
        //std::cout << "profile_task_stop\n";
        s_profiling_functions.task_stop( evt_info );
      }
      m_task_exec_time_accum.fetch_add( uint64_t( evt_info.elapsed().count() ) , std::memory_order_relaxed );
    }
  }

  void OperatorNode::task_start_callback( const onika::omp::OpenMPToolTaskTiming & evt_info )
  {
    if(evt_info.ctx != nullptr)
    {
      OperatorNode* op = reinterpret_cast<OperatorNode*>( evt_info.ctx );
      op->profile_task_start( evt_info );
    }
  }
  
  void OperatorNode::task_stop_callback( const onika::omp::OpenMPToolTaskTiming & evt_info )
  {
    if(evt_info.ctx != nullptr)
    {
      OperatorNode* op = reinterpret_cast<OperatorNode*>( evt_info.ctx );
      op->profile_task_stop( evt_info );
    }
  }

  double OperatorNode::collect_execution_time()
  {
#   ifdef ONIKA_HAVE_OPENMP_TOOLS
    bool do_profiling = global_profiling() && profiling();
    if( do_profiling )
    {
      auto t = m_task_exec_time_accum.exchange(0) / 1000000.0;
      if( t > 0.0 ) m_exec_times.push_back( t );
      return t;
    }
    else
#   endif
    {
      return 0.0;
    }
  }

  // if one implements generate_tasks, then it is backward compatible with sequential/parallel operator execution
  void OperatorNode::execute()
  {
    m_omp_task_mode = true;
  
#   pragma omp parallel
    {
#     pragma omp master
      {
#       pragma omp taskgroup
        {
          generate_tasks();          
        } // --- end of task group ---
      } // --- end of single ---
    } // --- end of parallel section ---
  }
  
  void OperatorNode::generate_tasks()
  {
    lerr << "Fatal: OperatorNode::generate_tasks not implemented in "<<pathname() << std::endl;
    std::abort();
  }

  ::onika::omp::OpenMPTaskInfo* OperatorNode::profile_begin_section(const char* tag)
  {
#   ifdef ONIKA_HAVE_OPENMP_TOOLS
    bool do_profiling = global_profiling() && profiling();
    if( do_profiling )
    {
      ::onika::omp::OpenMPTaskInfo * tinfo = new ::onika::omp::OpenMPTaskInfo { tag }; \
      ::onika::omp::OpenMPToolInterace::task_begin(tinfo);
      return tinfo;
    }
#   endif
    return nullptr;
  }

  void OperatorNode::profile_end_section(::onika::omp::OpenMPTaskInfo* tinfo)
  {
#   ifdef ONIKA_HAVE_OPENMP_TOOLS
    bool do_profiling = global_profiling() && profiling();
    if( do_profiling )
    {
      assert( tinfo != nullptr );
      ::onika::omp::OpenMPToolInterace::task_end(tinfo);
    }
#   endif
  }

  void OperatorNode::reset_profiling_reference_timestamp()
  {
    s_profiling_timestamp_ref = std::chrono::high_resolution_clock::now();
  }

  void OperatorNode::run()
  {
    //if( is_terminal() ) lout << "--> " << pathname() << "\n";
    onika_ompt_declare_task_context(tsk_ctx);
    
    const bool do_profiling = global_profiling() && profiling();
    const bool mem_prof = global_mem_profiling() && is_terminal();
    
    if( s_debug_execution )
    {
      if( is_terminal() ) lout<<pathname()<<" ..."<<std::flush;
      else lout<<pathname()<<" BEGIN"<<std::endl;
    }

#   ifndef ONIKA_HAVE_OPENMP_TOOLS
    TimeStampT T0;
#   endif
    
    ssize_t res_mem_inc = 0;
    if( mem_prof )
    {
      onika::memory::MemoryResourceCounters memcounters;
      memcounters.read();
      res_mem_inc = memcounters.stats[ onika::memory::MemoryResourceCounters::RESIDENT_SET_MB ];
    }
    
    if( do_profiling )
    {
#     ifdef ONIKA_HAVE_OPENMP_TOOLS
      onika_ompt_begin_task_context2( tsk_ctx, this );
#     else
      T0 = std::chrono::high_resolution_clock::now();
      onika::omp::OpenMPToolTaskTiming evt_info = { this , m_tag.get() , omp_get_thread_num() , T0 - s_profiling_timestamp_ref , T0 - s_profiling_timestamp_ref };
      profile_task_start( evt_info );
#     endif

      ONIKA_CU_PROF_RANGE_PUSH( m_tag.get() );
    }

    this->initialize_slots_resource();

    // sequential execution of fork-join parallel components
    // except if OperatorNode derived class implements generate_tasks instead of execute, in which case, task parallelism mode is used
    this->execute();

    if( do_profiling )
    { 
      ONIKA_CU_PROF_RANGE_POP();

      double total_gpu_time = 0.0;
      double total_async_cpu_time = 0.0;
      for(auto gpuctx:m_parallel_execution_contexts)
      {
        if( gpuctx != nullptr )
        {
          total_gpu_time += gpuctx->collect_gpu_execution_time();
          total_async_cpu_time += gpuctx->collect_async_cpu_execution_time();
        }
      }
      m_gpu_times.push_back( total_gpu_time ); // account for 0 (no GPU or async) execution times, to avoid asymetry between mpi processes
      m_async_cpu_times.push_back( total_async_cpu_time );

#     ifdef ONIKA_HAVE_OPENMP_TOOLS
      onika_ompt_end_task_context2( tsk_ctx, this );
      if( parent()==nullptr )
      {
        this->collect_execution_time();
      }
#     else
      auto T1 = std::chrono::high_resolution_clock::now();
      onika::omp::OpenMPToolTaskTiming evt_info = { this , m_tag.get() , omp_get_thread_num() , T0 - s_profiling_timestamp_ref , T1 - s_profiling_timestamp_ref };
      profile_task_stop( evt_info );
      auto exectime = ( T1 - T0 ).count() / 1000000.0;
      m_exec_times.push_back( exectime );
#     endif
    }

    if( mem_prof )
    {
      onika::memory::MemoryResourceCounters memcounters;
      memcounters.read();
      res_mem_inc = memcounters.stats[ onika::memory::MemoryResourceCounters::RESIDENT_SET_MB ] - res_mem_inc;
      m_resident_mem_inc += res_mem_inc;
    }

    if( s_debug_execution )
    {
      if( is_terminal() ) lout << " done"<<std::endl;
      else lout<<pathname()<<" END"<<std::endl;
    }

    bool may_finalize = !is_looping() ;
    auto * p = parent();
    while( p != nullptr )
    {
      may_finalize = may_finalize && !p->is_looping();
      p = p->parent();
    }
    if(may_finalize)
    {
      finalize();
    }
  }

  void OperatorNode::free_all_resources()
  {
    for(auto& p : m_slots)
    {
      p->free_resource();
    }    
  }

  void OperatorNode::set_gpu_enabled(bool b)
  {
    m_gpu_execution_allowed = b;
  }

  onika::parallel::ParallelExecutionContext* OperatorNode::parallel_execution_context(unsigned int id)
  {
    if( id >= m_parallel_execution_contexts.size() )
    {
      m_parallel_execution_contexts.resize( id+1 );
    }
    if( m_parallel_execution_contexts[id] == nullptr )
    {
      m_parallel_execution_contexts[id] = std::make_shared< onika::parallel::ParallelExecutionContext >();
      m_parallel_execution_contexts[id]->m_streamIndex = id;
      m_parallel_execution_contexts[id]->m_cuda_ctx = m_gpu_execution_allowed ? global_cuda_ctx() : nullptr;
      m_parallel_execution_contexts[id]->m_omp_num_tasks = m_omp_task_mode ? omp_get_max_threads() : 0;
    }
    return m_parallel_execution_contexts[id].get();
  }

  void OperatorNode::set_parent( OperatorNode* parent )
  {
    m_parent = parent;
    if( m_parent != nullptr )
    {
      m_depth = m_parent->depth() + 1;
    }
    else
    {
      m_depth = 0;
    }
  }

  void OperatorNode::register_in_slot( const std::string& name, OperatorSlotBase* s )
  {
    assert( ! compiled() );
    assert( in_slot_idx(name) == -1 );
    assert( in_slot_idx(s) == -1 );
    assert( m_in_slot_count < MAX_SLOT_COUNT );
    m_in_slot_storage[ m_in_slot_count ++ ] = { name , s };
    m_slots.insert(s);
  }

  void OperatorNode::register_out_slot( const std::string& name, OperatorSlotBase* s )
  {
    assert( ! compiled() );
    assert( out_slot_idx(name) == -1 );      
    assert( out_slot_idx(s) == -1 );
    assert( m_out_slot_count < MAX_SLOT_COUNT );
    m_out_slot_storage[ m_out_slot_count ++ ] = { name , s };
    m_slots.insert(s);
  }

  void OperatorNode::register_slot( const std::string& name, OperatorSlotBase* s )
  {
    assert( s != nullptr );
    assert( s->owner() == this );
    if( s->is_input() ) { register_in_slot( name , s ); }
    if( s->is_output() ) { register_out_slot( name , s ); }
  }

  void OperatorNode::register_managed_slot( std::shared_ptr<OperatorSlotBase> s )
  {
    assert( ! compiled() );
    m_managed_slots.insert( s );
  }

  // TODO: add a mechanism to register pretty_print functions for types of slot values
  // level_of_detail: 0 none, 1 slot names only, 2 slots names and values, 3 slot values and connections
  LogStreamWrapper& OperatorNode::pretty_print(LogStreamWrapper& out, int level_of_detail, int indent, ProfilePrintParameters& ppp , ParallelValueStatsFunc pstat )
  {
    static const std::string padding = "................................................................................";
    out<< format_string("%*s",indent,"") << name() ;
    
    if( ppp.m_print_profiling )
    {
      int np = 1, rank = 0;
      std::vector<double> min_time, max_time, avg_time;
      size_t total_exec_count = m_exec_times.size();
      double total_exec_time = 0.0;
      double avg_inbalance = 0.0;
      double max_inbalance = 0.0;

      // check cpu executions counts consistency
      {
        std::vector<double> exec_counts( 1 , total_exec_count );
        std::vector<double> exec_counts_min( 1 , 0.0 );
        std::vector<double> exec_counts_max( 1 , 0.0 );
        std::vector<double> exec_counts_avg( 1 , 0.0 );
        pstat( exec_counts, np, rank, exec_counts_min, exec_counts_max, exec_counts_avg );
        if( exec_counts_max[0] != exec_counts_min[0] )
        {
          fatal_error() << "OperatorNode::pretty_print ("<<pathname()<<") : P "<<rank<<" / "<<np<<" : execs = "<<total_exec_count<<", min="<<exec_counts_min[0]<<", max="<<exec_counts_max[0]<<std::endl;
        }
      }

      pstat( m_exec_times, np, rank, min_time, max_time, avg_time );
      assert( avg_time.size() == total_exec_count );
      assert( min_time.size() == total_exec_count );
      assert( max_time.size() == total_exec_count );

      //lout<<name()<<" m_exec_times ("<<m_exec_times.size()<<") : "; for(auto d:m_exec_times) lout<<d<<" "; lout<<std::endl;
      //lout<<name()<<" avg_time ("<<avg_time.size() <<") : "; for(auto d:avg_time) lout<<d<<" "; lout<<std::endl;
      
      for(size_t i=0;i<total_exec_count;i++)
      {
        assert( max_time[i] >= avg_time[i] );
        double inbalance = 0.0;
        if( avg_time[i] > 0. ) { inbalance = ( max_time[i] - avg_time[i] ) / avg_time[i]; }
        max_inbalance = std::max(max_inbalance,inbalance);
        avg_inbalance += inbalance;
        total_exec_time += avg_time[i];
      }
      if(total_exec_count>0) avg_inbalance /= total_exec_count;
      else avg_inbalance = 0.0;

      // GPU specific stats
      std::vector<double> gpu_min_time, gpu_max_time, gpu_avg_time;
      size_t gpu_total_exec_count = m_gpu_times.size();
      double gpu_total_exec_time = 0.0;
      double gpu_avg_inbalance = 0.0;
      double gpu_max_inbalance = 0.0;
      // check GPU executions counts consistency
      {
        std::vector<double> exec_counts( 1 , gpu_total_exec_count );
        std::vector<double> exec_counts_min( 1 , 0.0 );
        std::vector<double> exec_counts_max( 1 , 0.0 );
        std::vector<double> exec_counts_avg( 1 , 0.0 );
        pstat( exec_counts, np, rank, exec_counts_min, exec_counts_max, exec_counts_avg );
        if( exec_counts_max[0] != exec_counts_min[0] )
        {
          fatal_error() << "OperatorNode::pretty_print ("<<pathname()<<") : P "<<rank<<" / "<<np<<" : GPU execs = "<<gpu_total_exec_count<<", min="<<exec_counts_min[0]<<", max="<<exec_counts_max[0]<<std::endl;
        }
      }
      pstat( m_gpu_times, np, rank, gpu_min_time, gpu_max_time, gpu_avg_time );
      assert( gpu_avg_time.size() == gpu_total_exec_count );
      assert( gpu_min_time.size() == gpu_total_exec_count );
      assert( gpu_max_time.size() == gpu_total_exec_count );
      for(size_t i=0;i<gpu_total_exec_count;i++)
      {
        assert( gpu_max_time[i] >= gpu_avg_time[i] );
        double inbalance = 0.0;
        if( gpu_avg_time[i] > 0. ) { inbalance = ( gpu_max_time[i] - gpu_avg_time[i] ) / gpu_avg_time[i]; }
        gpu_max_inbalance = std::max(gpu_max_inbalance,inbalance);
        gpu_avg_inbalance += inbalance;
        gpu_total_exec_time += gpu_avg_time[i];
      }
      gpu_avg_inbalance /= gpu_total_exec_count;

      if( ppp.m_total_time < total_exec_time ) { ppp.m_total_time = total_exec_time; }
      if( ppp.m_inner_loop_time < total_exec_time && is_looping() ) { ppp.m_inner_loop_time = total_exec_time; }

      assert( padding.length() >= 50 );
      int padlen = ( 50 - (name().length()+indent) ) ;
      if( padlen<0 ) { padlen = 0; }
      double pt = 100.*total_exec_time/ppp.m_total_time;
      if( pt >= 0.01 )
      {
        std::string gpu_total_exec_time_str = "           ";
        if( gpu_total_exec_count > 0 && gpu_total_exec_time > 0.0 )
        {
          gpu_total_exec_time_str = format_string(" (%.2e)",gpu_total_exec_time);
        }
        out << format_string(" %*.*s % .3e%s%6.3f  %6.3f %9ld  %5.2f%%",padlen,padlen,padding,total_exec_time,gpu_total_exec_time_str,avg_inbalance,max_inbalance,total_exec_count,pt);
        if( ppp.m_inner_loop_time > 0.0 )
        {
          out << format_string(" / %5.2f%%", 100.*total_exec_time/ppp.m_inner_loop_time );
        }
      }
      else if( gpu_total_exec_count > 0 && gpu_total_exec_time > 0.0 )
      {
        out << " (G)";
      }
    }
    out << std::endl;
    if( level_of_detail>=1 && is_terminal() )
    {
      indent += 2;
      for(const auto& p : named_slots() )
      {
        out << format_string("%*s %s ",indent+p.first.length(),p.first,slot_dir_str(p.second->direction()));
        if( level_of_detail >= 2 )
        {
          std::string t = pretty_short_type( p.second->value_type() );
          // replace type name with readable value if possible
          out << t << " = " << p.second->value_as_string() ;
        }
        out << std::endl;
        if( level_of_detail >= 3 )
        {
          OperatorSlotBase* input_slot = p.second->input();
          while( input_slot!=nullptr && !input_slot->owner()->is_terminal() && input_slot->input()!=nullptr )
          {
            input_slot = input_slot->input();
          }
          if( input_slot != nullptr )
          {
            out << format_string("%*s",indent+2,"") << "from "<<input_slot->pathname() << std::endl;
          }
          for(auto& op:p.second->outputs())
          {
            out << format_string("%*s",indent+2,"") << "to "<<op->pathname() << std::endl;
          }
        }
      }
    }
    return out;
  }

  void OperatorNode::initialize_slots_resource()
  {
    for(const auto& p : in_slots() )
    {
      assert( p.second->owner() == this );
      // ldbg << "RSC: INIT INPUT "<<p.second->pathname() << std::endl;
      p.second->initialize_resource_pointer();
    }
    for(const auto& p : out_slots() )
    {
      assert( p.second->owner() == this );
      // ldbg << "RSC: INIT OUTPUT "<<p.second->pathname() << std::endl;
      p.second->initialize_resource_pointer();
    }
  }

  // output dependencies
  void OperatorNode::outflow_reachable_slots(std::set<const OperatorSlotBase*>& ors) const
  {
    for(const auto& p : out_slots() )
    {
      //ors.insert( p.second );
      p.second->outflow_reachable_slots( ors );
    }
  }

  void OperatorNode::post_graph_build()
  {
    // calculate operator depth
    if( parent() != nullptr )
    {
      m_depth = parent()->depth() + 1;
    }
    else
    {
      m_depth = 0;
    }

    // pre-stored operator stack to help debugging
    if( parent() != nullptr ) { m_backtrace = parent()->m_backtrace + "." + m_name; }
    else { m_backtrace = m_name; }

    // compute operator stack hash
#   pragma omp critical(OperatorNode_get_next_global_index)
    {
      m_hash = s_global_instance_index++;    
    }

    // reserve space for profiling buffers
    if( global_profiling() || profiling() )
    {
      m_exec_times.reserve(4096);
      m_gpu_times.reserve(4096);
      m_async_cpu_times.reserve(4096);
    }
  }

  void OperatorNode::set_global_cuda_ctx( std::shared_ptr<onika::cuda::CudaContext> ctx )
  {
    s_global_cuda_ctx = ctx;
  }
  
  onika::cuda::CudaContext* OperatorNode::global_cuda_ctx()
  {
    return s_global_cuda_ctx.get();
  }

  void OperatorNode::yaml_initialize(const YAML::Node& node)
  {  
    if( node.IsMap() )
    {
      std::set<OperatorSlotBase*> yaml_initialized_slots;
      for(auto& p : in_slots() )
      {
        if( yaml_initialized_slots.find(p.second)==yaml_initialized_slots.end() )
        {
          if( node[p.first] )
          {
            // ldbg << "initialize in slot '"<<p.first<<"' from YAML" << std::endl;
            p.second->yaml_initialize( node[p.first] );
            yaml_initialized_slots.insert( p.second );
          }
        }
      }

      for(auto& p : out_slots() )
      {
        if( yaml_initialized_slots.find(p.second)==yaml_initialized_slots.end() )
        {
          if( node[p.first] )
          {
            // ldbg << "initialize out slot '"<<p.first<<"' from YAML" << std::endl;
            p.second->yaml_initialize( node[p.first] );
            yaml_initialized_slots.insert( p.second );
          }
        }
      }

    }    
  }

  std::ostream& OperatorNode::print_documentation( std::ostream& out ) const
  {
    out << "Operator    : "<< name() << std::endl << std::endl
        << "Description : "<< documentation() << std::endl
        << "Slots       : " << std::endl;
    for(auto& p : named_slots() )
    {
      out << "  " << p.first << std::endl
          << "    flow " << slot_dir_str(p.second->direction()) << std::endl
          << "    type " << pretty_short_type( p.second->value_type() ) << std::endl
          << "    desc " << p.second->documentation() << std::endl;
    }
    return out;
  }

}

