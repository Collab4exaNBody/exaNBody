#pragma once

#include <onika/dac/stencil.h>
#include <onika/dac/box_span.h>

#include <onika/dag/dag.h>
#include <onika/dag/dag_algorithm.h>
#include <onika/dag/dag_filter.h>
#include <onika/dag/dag_reorder.h>
#include <onika/dag/dag_execution.h>
#include <onika/hash_utils.h>
#include <onika/task/parallel_task_cost.h>
#include <onika/task/parallel_task_config.h>
#include <onika/omp/task_detach.h>
#include <onika/omp/dynamic_depend_dispatch.h>
#include <onika/lambda_tools.h>

#include <onika/task/parallel_task_executor.h>
#include <onika/task/ptask_execute.h>
#include <onika/task/dag_scheduler_fifo.h>
#include <onika/task/dag_scheduler_heap.h>
#include <onika/task/dag_scheduler_ompdep.h>
#include <onika/task/dag_scheduler_gpu.h>
#include <onika/task/span_scheduler_gpu.h>

#include <onika/memory/allocator.h>

#include <cstdlib>

// for DAG diagnostics output
#include <fstream>
#include <sstream>
#include <ios>
#include <iomanip>
#include <algorithm>
#include <onika/dag/dag_stream.h>
#include <onika/task/tag_utils.h>

//#define ONIKA_PTASK_DUMP_DAG_TO_DOT 1
#ifdef ONIKA_PTASK_DUMP_DAG_TO_DOT
#include <onika/dag/dag2_stream.h>
#endif


namespace onika
{

  namespace task
  {

    template<class _SpanT, TaskTypeEnum _TaskType, class KernelFunc , class CostFunc, bool _CudaEnabled , class... AccessorsP>
    struct ParallelTaskExecutorImpl<_SpanT,_TaskType,FlatTuple<AccessorsP...>,KernelFunc,CostFunc,_CudaEnabled> final : public ParallelTaskExecutor
    {
      using SpanT = _SpanT;
      using AccTuple = FlatTuple<AccessorsP...>;
      using PTaskProxyT = PTaskProxy<SpanT,_TaskType,AccTuple,KernelFunc,CostFunc,_CudaEnabled>;
      static inline constexpr bool CudaEnabled = _CudaEnabled;

      static_assert( dac::DataAccessControlerSet<AccessorsP...>::is_valid , "inconsistent accessor pack" );
      static_assert( is_onika_array_v<typename SpanT::coord_t> , "data decomposition space coordinate type must be an oarray_t" );

      static inline constexpr std::integral_constant<size_t,SpanT::ndims> ndims {};
      using first_nd_dac_t = dac::first_nd_dac_t< ndims.value , AccessorsP ... >;
      static inline constexpr auto first_nd_dac_index_v = dac::first_nd_dac_index_v< ndims.value , AccessorsP ... >;
      static inline constexpr bool has_first_nd_dac = first_nd_dac_index_v.value < sizeof...(AccessorsP);
      using dac_set_nd_indices = typename dac::DacSetSplitter<AccTuple>::dac_set_nd_indices;
      using dac_set_0d_indices = typename dac::DacSetSplitter<AccTuple>::dac_set_0d_indices;
      
      inline ParallelTaskExecutorImpl(size_t czh, size_t dh, size_t sh, PTaskProxyT && proxy )
        : ParallelTaskExecutor(czh,dh,sh,proxy.m_span, proxy.m_accs, dac_set_nd_indices{} , dac_set_0d_indices{} )
        , m_proxy(std::move(proxy))
      {
        m_proxy.m_queue = nullptr;
      }

      /*inline ~ParallelTaskExecutorImpl()
      {
        std::cout<<"~ParallelTaskExecutorImpl @"<<(void*)this<<"\n";
      }*/

      // ------------ public interface implementation -----------------------      
      inline void update_span_int() override final
      {
        m_proxy.m_span.copy_from( span() );
      }

      inline void update_ptask( ParallelTask* pt ) override final
      {
        m_proxy.m_ptask = pt;
      }

      inline const dag::AbstractWorkShareDAG& dep_graph() const override final
      {
        if(m_reduced_dag) return m_filtered_dag;
        else return m_dag;
      }

      inline const char* tag() const override final
      {
        return m_proxy.m_tag;
      }


      ONIKA_HOST_DEVICE_FUNC inline const dag::WorkShareDAG2<ndims.value> & dep_graph_impl() const
      {
        if(m_reduced_dag) return m_filtered_dag.m_dag;
        else return m_dag.m_dag;
      }
      
      inline void invalidate_graph() override final
      {
        m_dag.clear();
      }

      inline size_t in_dependences( size_t i , const void* indeps[] ) const 
      {
        if constexpr ( has_first_nd_dac && ndims.value > 0 )
        {
          size_t n = dep_graph_impl().item_dep_count( i );
          for(size_t j=0;j<n;j++)
          {
            auto p = m_proxy.m_accs.get(tuple_index<first_nd_dac_index_v.value>).pointer_at( dep_graph_impl().item_dep(i,j) );
            assert( p != nullptr );
            indeps[j] = p;
          }
          return n;
        }
        return 0;
      }

      inline void* out_dependence( size_t i ) const 
      {
        if constexpr ( has_first_nd_dac )
        {
          if( ! dep_graph_impl().empty() )
          {
            return m_proxy.m_accs.get(tuple_index<first_nd_dac_index_v.value>).pointer_at( dep_graph_impl().item_coord(i) );
          }
          return m_proxy.m_accs.get(tuple_index<first_nd_dac_index_v.value>).pointer_at( span(). template index_to_coord<ndims.value> (i) );
        }
        return nullptr;
      }
      
      // return false if graph happens to be trivial
      inline bool build_graph(ParallelTask* pt) override final
      {
        using TimePoint = decltype( std::chrono::high_resolution_clock::now() );

        int team_threads = ParallelTaskConfig::dag_graph_mt();
	      if( team_threads < 1 ) team_threads = 1;
	      //std::cout<<"nt = "<<team_threads<<std::endl;

        assert( span().ndims == ndims.value );
        if constexpr ( ndims.value > 0 )
        {
          dag::stencil_dag_sats_t stats;          
          bool dag_rebuilt = false;

          if( ParallelTaskConfig::dag_diagnostics() )
          {
            stats[dag::DagBuildStats::SCRATCH_PEAK_MEMORY] = 0;
            stats[dag::DagBuildStats::DEP_CONSTRUCTION_TIME] = 0;
            stats[dag::DagBuildStats::NODEP_TASK_COUNT] = 0;
            stats[dag::DagBuildStats::MAX_DEP_DEPTH] = 0;
          }

          if( ! m_dag.m_dag.empty() )
          {
            //std::cout << "reuse existing graph" << std::endl;
            assert_equal_dag( m_dag.m_dag , dag::make_stencil_dag<ndims.value>( span() , stencil() ) );
          }
          else
          {
            //std::cout << "(re)build graph" << std::endl;
            m_dag.m_dag = dag::make_stencil_dag2<ndims.value>( span() , stencil() , stats );
            m_max_dep_depth = stats[dag::DagBuildStats::MAX_DEP_DEPTH];
            // std::cout<<"max depth = "<<m_max_dep_depth<<std::endl;
            assert( m_dag.m_dag.number_of_items() == span().nb_coarse_cells() || m_dag.m_dag.empty() );

            dag_rebuilt = true;
            m_filtered_dag.clear();
            m_costs.clear();
            m_reduce_map.clear();
            m_task_reorder.clear();

            if( m_dag.m_dag.empty() )
            {
              //std::cout<<"stencil generates empty graph\n";
              return false;
            }
          }
          
          size_t n_dag_items = m_dag.m_dag.number_of_items();
          m_costs_changed = false;
          m_reduced_dag = false;
          m_total_skipped = 0;
          m_total_count = 0;
          m_total_cost = 0;

          // ----------- profiling -------------
          TimePoint cost_time_start;
          if( ParallelTaskConfig::dag_diagnostics() )
          {
            cost_time_start = std::chrono::high_resolution_clock::now();
          }
          // ------------------------------------
          
          if( dag_rebuilt || m_reduce_map.empty() )
          {
            //std::cout << "reconstruct/refill costs buffer entirely : n_dag_items="<<n_dag_items << std::endl;
            m_costs.resize( n_dag_items , std::numeric_limits<size_t>::max() );            
            auto* self = this;
            auto reduced_dag = m_reduced_dag;
            auto costs_changed = m_costs_changed;
            auto total_count = m_total_count;
            auto total_cost = m_total_cost;
            auto total_skipped = m_total_skipped;
            auto * costs = m_costs.data();
#           pragma omp taskloop default(none) firstprivate(n_dag_items,self,costs) reduction(or:reduced_dag,costs_changed) reduction(+:total_count,total_cost,total_skipped) num_tasks(team_threads)
            for(size_t i=0;i<n_dag_items;i++)
            {
              auto tci = ptask_execute_costfunc( self->m_proxy , self->task_coord_raw(i) ); //task_cost(pt,i);
              total_count += tci.count;
              total_cost += tci.cost;
              if( tci.cost == 0 )
              {
                assert( tci.count == tci.skipped );
                total_skipped += tci.skipped;
                reduced_dag = true;
              }
              if( tci.cost != costs[i] )
              {
                costs_changed = true;
                costs[i] = tci.cost;
              }
            }
            m_reduced_dag = reduced_dag;
            m_costs_changed = costs_changed;
            m_total_count = total_count;
            m_total_cost = total_cost;
            m_total_skipped = total_skipped; 

          }
          else 
          {
            //std::cout << "compare to reduced costs : n_dag_items="<<n_dag_items << std::endl;
            assert( m_reduce_map.size() == n_dag_items );
            auto* self = this;
            auto reduced_dag = m_reduced_dag;
            auto costs_changed = m_costs_changed;
            auto total_count = m_total_count;
            auto total_cost = m_total_cost;
            auto total_skipped = m_total_skipped;
            auto * reduce_map = m_reduce_map.data();
            auto * costs = m_costs.data();
#           pragma omp taskloop default(none) firstprivate(n_dag_items,self,reduce_map,costs) reduction(or:reduced_dag,costs_changed) reduction(+:total_count,total_cost,total_skipped) num_tasks(team_threads)
            for(size_t i=0;i<n_dag_items /* && !m_costs_changed */ ;i++)
            {
              auto tci = ptask_execute_costfunc( self->m_proxy , self->task_coord_raw(i) ); //task_cost(pt,i);
              total_count += tci.count;
              total_cost += tci.cost;
              if( tci.cost == 0 )
              {
                assert( tci.count == tci.skipped );
                total_skipped += tci.skipped;
                reduced_dag = true;
              }
              if( reduce_map[i] == -1 )
              {
                if( tci.cost != 0 ) costs_changed = true;
              }
              else
              {
                if( tci.cost != costs[reduce_map[i]] ) costs_changed = true;
              }
            }
            m_reduced_dag = reduced_dag;
            m_costs_changed = costs_changed;
            m_total_count = total_count;
            m_total_cost = total_cost;
            m_total_skipped = total_skipped;
            if( m_costs_changed )
            {
#             pragma omp taskloop default(none) firstprivate(n_dag_items,self) num_tasks(team_threads)
              for(size_t i=0;i<n_dag_items;i++)
              {
                 self->m_costs[i] = ptask_execute_costfunc( self->m_proxy , self->task_coord_raw(i) ).cost;
              }
              m_reduce_map.clear();
            }
          }

          // ----------- profiling -------------
          size_t read_costs_time = 0;
          TimePoint reduce_time_start;
          if( ParallelTaskConfig::dag_diagnostics() )
          {
            reduce_time_start = std::chrono::high_resolution_clock::now();
            auto nanosecs = std::chrono::nanoseconds( reduce_time_start - cost_time_start ).count();
            read_costs_time = static_cast<size_t>( nanosecs / 1000 );
          }
          // ------------------------------------

          // std::cout<<"reducable="<<m_reduced_dag<<", m_costs_changed="<<m_costs_changed<<", reduced="<< (m_reduced_dag && ParallelTaskConfig::dag_reduce()) <<std::endl;
          m_reduced_dag = m_reduced_dag && ParallelTaskConfig::dag_reduce();
          
          if( m_reduced_dag )
          {
            if( m_filtered_dag.m_dag.empty() || m_costs_changed )
            {
              // std::cout << "recompute DAG reduction" << std::endl;
              m_filtered_dag.m_dag = filter_dag( m_dag.m_dag , [c=m_costs.data()](size_t i) ->bool {return c[i]!=0;} , m_reduce_map );
              assert( m_reduce_map.size() == n_dag_items );
              for(size_t i=0;i<n_dag_items;i++)
              {
                if( m_reduce_map[i] != -1 )
                {
                  assert( m_reduce_map[i]>=0 && m_reduce_map[i] <= ssize_t(i) );
                  m_costs[ m_reduce_map[i] ] = m_costs[i];
                }
              }
              dag_rebuilt = true;
            }
          }
          else
          {
            m_reduce_map.clear();
          }

          auto & final_dag = m_reduced_dag ? m_filtered_dag.m_dag : m_dag.m_dag ;

          // ----------- profiling -------------
          size_t reduce_dag_time = 0;
          TimePoint reorder_dag_start_time;
          if( ParallelTaskConfig::dag_diagnostics() )
          {
            reorder_dag_start_time = std::chrono::high_resolution_clock::now();
            auto nanosecs = std::chrono::nanoseconds( reorder_dag_start_time - reduce_time_start ).count();
            reduce_dag_time = static_cast<size_t>( nanosecs / 1000 );
          }
          // ------------------------------------

          if( dag_rebuilt )
          {
            //if( ParallelTaskConfig::dag_scheduler() != ONIKA_DAG_SCHEDULER_OMPDEP )
            //{
              //std::cout << "rebuild DAG exe" << std::endl;
              m_priv_dag_exe.rebuild_from_dag( final_dag );
            //}
          }
          
          if( ( dag_rebuilt || m_costs_changed ) && ParallelTaskConfig::dag_reorder() && ParallelTaskConfig::dag_scheduler() == ONIKA_DAG_SCHEDULER_OMPDEP )
          {
            // std::cout << "reorder tasks" << std::endl;
            reorder_fast_dep_unlock( final_dag , m_priv_dag_exe , m_task_reorder , [c=m_costs.data()](size_t a) -> uint64_t { return c[a]; } );
          }
          else
          {
            m_task_reorder.clear();
          }

          // prepare for access to DAG execution data that never reallocates or move data (Cuda compatibility security)
          m_priv_dag_exe.m_dep_countdown.resize( final_dag.number_of_items() , 0 );
          m_dag_exe = m_priv_dag_exe.shallow_copy();

          if( ParallelTaskConfig::dag_diagnostics() )
          {
            if( ! m_dag.m_dag.empty() )
            {
              auto nanosecs = std::chrono::nanoseconds( std::chrono::high_resolution_clock::now() - reorder_dag_start_time ).count();
              size_t reorder_time = static_cast<size_t>( nanosecs / 1000 );
                            
              size_t reduced_source_tasks = stats[dag::DagBuildStats::NODEP_TASK_COUNT];
              if( m_reduced_dag )
              {
                reduced_source_tasks = 0;
                size_t n = final_dag.number_of_items();
                for(size_t i=0;i<n;i++) if( final_dag.item_dep_count(i)==0 ) ++reduced_source_tasks;
              }
              
              std::ofstream fout;
              if( m_diag_count == 0 )
              {
                std::ostringstream oss;
                oss << "trace." << tag_filter_out_path( get_tag() ) << "_H" << std::hex << codezone_hash() << ".info";
                m_diag_base_name = oss.str();
                std::replace( m_diag_base_name.begin() , m_diag_base_name.end() , ':' , '@' );
                fout.open( m_diag_base_name );
                fout<<"build-time ; build-mem ; source-tasks ; max-dep-depth ; total-tasks ; total-deps ; "
                    <<"cost-time ; reorder-dagexe-time ; reduce-time ; reduced-tasks ; reduced-deps ; reduced-source-tasks"<<std::endl;
              }
              else
              {
                fout.open( m_diag_base_name , std::ios::app );
              }
              ++ m_diag_count;

              fout << stats[dag::DagBuildStats::DEP_CONSTRUCTION_TIME] << " ; "
                   << stats[dag::DagBuildStats::SCRATCH_PEAK_MEMORY] << " ; "
                   << stats[dag::DagBuildStats::NODEP_TASK_COUNT] << " ; "
                   << stats[dag::DagBuildStats::MAX_DEP_DEPTH] << " ; "
                   << m_dag.m_dag.number_of_items() << " ; "
                   << m_dag.m_dag.total_dep_count() << " ; "
                   << read_costs_time << " ; "
                   << reorder_time << " ; "
                   << reduce_dag_time << " ; "
                   << final_dag.number_of_items() << " ; "
                   << final_dag.total_dep_count() << " ; "
                   << reduced_source_tasks << std::endl;              
#             ifdef ONIKA_PTASK_DUMP_DAG_TO_DOT
              {
                std::string dotfname = m_diag_base_name + ".dot";
                std::ofstream dotfout ( dotfname );
                onika::dag::dag_to_dot( m_dag.m_dag , span().template coarse_domain_nd<ndims.value> () , dotfout , onika::dag::Dag2DotConfig<ndims.value>{} );  
              }
#             endif
            }
          }


        }
        
        return true;
      }
      
      ONIKA_HOST_DEVICE_FUNC inline oarray_t<size_t,1> task_coord_1( ParallelTask*, size_t i ) const //override final
      {
        if constexpr ( ndims.value == 1 )
        {
          auto c = dep_graph_impl().item_coord_cu(i);
          c[0] = c[0] * m_proxy.m_span.grainsize + m_proxy.m_span.lower_bound[0];
          return c;
        }
        ONIKA_CU_ABORT(); return {};
      }

      ONIKA_HOST_DEVICE_FUNC inline oarray_t<size_t,2> task_coord_2( ParallelTask*, size_t i ) const //override final
      {
        if constexpr ( ndims.value == 2 )
        {
          auto c = dep_graph_impl().item_coord_cu(i);
          c[0] = c[0] * m_proxy.m_span.grainsize + m_proxy.m_span.lower_bound[0];
          c[1] = c[1] * m_proxy.m_span.grainsize + m_proxy.m_span.lower_bound[1];
          return c;
        }
        ONIKA_CU_ABORT(); return {};
      }

      ONIKA_HOST_DEVICE_FUNC inline oarray_t<size_t,3> task_coord_3( ParallelTask*, size_t i ) const //override final
      {
        if constexpr ( ndims.value == 3 )
        {
          auto c = dep_graph_impl().item_coord_cu(i);
          c[0] = c[0] * m_proxy.m_span.grainsize + m_proxy.m_span.lower_bound[0];
          c[1] = c[1] * m_proxy.m_span.grainsize + m_proxy.m_span.lower_bound[1];
          c[2] = c[2] * m_proxy.m_span.grainsize + m_proxy.m_span.lower_bound[2];
          return c;
        }
        ONIKA_CU_ABORT(); return {};
      }

      inline auto task_coord_raw( size_t i ) const
      {
        auto c = m_dag.m_dag.item_coord(i);
        for(size_t i=0;i<ndims.value;i++) c[i] = c[i] * SpanT::grainsize + span().lower_bound[i];
        return c;
      }

      ONIKA_HOST_DEVICE_FUNC inline oarray_t<size_t,ndims.value> task_coord_int( ParallelTask*, size_t i ) const
      {
        static_assert( ndims.value<=3 , "Only dimensions up to 3 are supported" );
        if constexpr ( ndims.value == 0 ) return {};
        if constexpr ( ndims.value == 1 ) return task_coord_1(nullptr,i);
        if constexpr ( ndims.value == 2 ) return task_coord_2(nullptr,i);
        if constexpr ( ndims.value == 3 ) return task_coord_3(nullptr,i);
        ONIKA_CU_ABORT(); return {};
      }

      ONIKA_HOST_DEVICE_FUNC inline void execute_int( ParallelTask* pt , size_t i )
      {
        auto c = task_coord_int(pt,i);
        return ptask_execute_kernel(m_proxy,c);
      }

      inline void execute( ParallelTask* pt ) const override final
      {
        assert( m_proxy.m_ptask == pt );
        onika_ompt_begin_task(m_proxy.m_tag);
        constexpr bool compat = ( ndims == 0 );
        if constexpr ( compat) { pt->account_completed_task( ptask_execute_kernel(m_proxy) ); } // 0D kernels do not have to return number of completed tasks
        if constexpr (!compat) { std::abort(); }
        onika_ompt_end_task(m_proxy.m_tag);
      }
      inline void execute( ParallelTask* pt , size_t i ) const override final
      {
        assert( m_proxy.m_ptask == pt );
        onika_ompt_begin_task(m_proxy.m_tag);
        auto c = task_coord_int(pt,i);
        pt->account_completed_task( ptask_execute_kernel(m_proxy,c) );
        onika_ompt_end_task(m_proxy.m_tag);
      }
      inline void execute( ParallelTask* pt , oarray_t<size_t,1> c ) const override final
      {
        assert( m_proxy.m_ptask == pt );
        onika_ompt_begin_task(m_proxy.m_tag);
        constexpr bool compat = ( ndims == 1 );
        if constexpr ( compat) { pt->account_completed_task( ptask_execute_kernel(m_proxy,c) ); }
        if constexpr (!compat) { std::abort(); }
        onika_ompt_end_task(m_proxy.m_tag);
      }
      inline void execute( ParallelTask* pt , oarray_t<size_t,2> c ) const override final
      {
        assert( m_proxy.m_ptask == pt );
        onika_ompt_begin_task(m_proxy.m_tag);
        constexpr bool compat = ( ndims == 2 );
        if constexpr ( compat) { pt->account_completed_task( ptask_execute_kernel(m_proxy,c) ); }
        if constexpr (!compat) { std::abort(); }
        onika_ompt_end_task(m_proxy.m_tag);
      }
      inline void execute( ParallelTask* pt , oarray_t<size_t,3> c ) const override final
      {
        assert( m_proxy.m_ptask != nullptr );
        assert( pt != nullptr );
        assert( m_proxy.m_ptask == pt );
        onika_ompt_begin_task(m_proxy.m_tag);
        constexpr bool compat = ( ndims == 3 );
        if constexpr ( compat) { pt->account_completed_task( ptask_execute_kernel(m_proxy,c) ); }
        if constexpr (!compat) { std::abort(); }
        onika_ompt_end_task(m_proxy.m_tag);
      }

      void all_tasks_completed_notify() override final
      {
        if( ParallelTaskConfig::dag_diagnostics() )
        {
          if( ParallelTaskConfig::dag_scheduler() != ONIKA_DAG_SCHEDULER_OMPDEP )
          {
            size_t n = dep_graph_impl().number_of_items();
            for(size_t i=0;i<n;i++)
            {
              assert( m_priv_dag_exe.dep_counter(i) /*m_dep_countdown[i].load()*/ == 0 );
            }
          }
        }
      }

      // helper function, intel-19's icpc crashes if pragma is placed inside the "if constexpr"
      static inline void spawn_0d_with_prev_dep_helper( ParallelTask* _self , ParallelTask* _prev_pt )
      {
        ParallelTask* self = _self;
        ParallelTask* prev_pt = _prev_pt;
        
        if( prev_pt != nullptr ) { ONIKA_DBG_MESG_LOCK { std::cout<<"PTaskExecutorImpl["<<self->get_tag()<<"] : depend(in:"<<(void*)prev_pt<<")"<< std::endl; } }

#       pragma omp task default(none) firstprivate(self,prev_pt) depend(in:prev_pt[0])
        {
          if(prev_pt==nullptr){}
          self->m_ptexecutor->execute( self );
        }
      }

      static inline void spawn_omp_all_tasks_int( ParallelTaskExecutorImpl* self, ParallelTask* pt )
      {
        onika_ompt_begin_task("onika-schedule-spawn");

        static constexpr size_t Nd = SpanT::ndims;
        if constexpr ( Nd == 0)
        {
          assert( pt->m_num_elements == 1 );
          assert( pt->m_num_tasks == 1 );
          spawn_0d_with_prev_dep_helper( pt , pt->m_sequenced_before ); // workaround for intel-19 frontend weakness   
        }
        
        if constexpr ( Nd > 0)
        {
          auto ptq = pt->m_ptq;
          if( ptq == nullptr )
          {
            ptq = & ParallelTaskQueue::global_ptask_queue();
          }        
          assert( ptq != nullptr );
          auto * cuda_ctx = ptq->cuda_ctx();
          bool cuda_available = CudaEnabled;
          if( cuda_ctx == nullptr ) cuda_available = false;
          else if( cuda_ctx->m_devices.empty() ) cuda_available = false;

          if( pt->m_trivial_dag )
          {
            bool coarse_coord = ( pt->m_num_tasks != pt->m_num_elements );
            if( cuda_available && ! pt->m_detached )
            {
              if constexpr( CudaEnabled ) SpanSchedulerGPU::schedule_tasks(self,pt,coarse_coord);
            }
            else
            {
              if( pt->m_detached )
              {
                for(size_t i=0;i<pt->m_num_tasks;i++)
                {
                  omp_event_handle_t tmp_evt{};
                  auto fp_i = i;       // ensures that thoses two variables will be firstprivate,
                  auto fp_pt = pt; // even if data sharing clauses are removed (GCC-11 compatibility)
                  OMP_TASK_DETACH( default(none) firstprivate(fp_i,fp_pt) , /**/ , tmp_evt )
                  {
                    auto c = fp_pt->span().template index_to_coord<Nd>(fp_i);
                    fp_pt->m_ptexecutor->execute( fp_pt , c );
                  }
                  pt->omp_completion_events()[fp_i] = std::move(tmp_evt);
                  pt->notify_completion_event_available( fp_i );
                }
              }
              else
              {
                const size_t ntasks = pt->m_num_tasks;
                const size_t team_threads = omp_get_max_threads() * 16;
#               pragma omp taskloop default(none) firstprivate(pt,ntasks,coarse_coord) nogroup num_tasks(team_threads)
                for(size_t i=0;i<ntasks;i++)
                {
                  if(coarse_coord)
                  {
//#                   pragma omp task default(none) firstprivate(i,pt) /* depend(out:xxx)*/
                    { pt->m_ptexecutor->execute( pt , pt->span().template coarse_index_to_coord_base<Nd>(i) ); }
                  }
                  else
                  {
//#                   pragma omp task default(none) firstprivate(i,pt) /* depend(out:xxx)*/
                    { pt->m_ptexecutor->execute( pt , pt->span().template index_to_coord<Nd>(i) ); }
                  }
                }
              }
              
            }
          }
          else
          {
            int dag_scheduler = ParallelTaskConfig::dag_scheduler();
            if( dag_scheduler == ONIKA_DAG_SCHEDULER_AUTO )
            {
              dag_scheduler = cuda_available ? ONIKA_DAG_SCHEDULER_GPU : ONIKA_DAG_SCHEDULER_NATIVEFIFO;
            }
            //std::cout << "executing DAG with scheduler "<<ParallelTaskConfig::dag_scheduler()<<"\n";
            switch( dag_scheduler )
            {
              case ONIKA_DAG_SCHEDULER_OMPDEP      : DagSchedulerOMPDep::schedule_tasks(self,pt); break;
              case ONIKA_DAG_SCHEDULER_NATIVEORDER : DagSchedulerHeap::schedule_tasks(self,pt); break;
              case ONIKA_DAG_SCHEDULER_NATIVEFIFO  : DagSchedulerFifo::schedule_tasks(self,pt); break;
              case ONIKA_DAG_SCHEDULER_GPU         :
                if constexpr( CudaEnabled ) { DagSchedulerGPU::schedule_tasks(self,pt); }
                if constexpr( ! CudaEnabled ) { std::cerr<<"Cannot apply GPU scheduler with cuda disabled PTaskProxy\n"; std::abort(); }
                break;
              default : std::abort();
            }
          }
        }
        
        onika_ompt_end_current_task();
      }

      inline void spawn_omp_all_tasks( ParallelTask* pt ) override final
      {
        auto * self = this;
        if( ParallelTaskConfig::dag_scheduler_tied() )
        {       
#         pragma omp task default(none) firstprivate(pt,self)
          { spawn_omp_all_tasks_int(self,pt); }
        }
        else
        {
#         pragma omp task default(none) firstprivate(pt,self) untied
          { spawn_omp_all_tasks_int(self,pt); }
        }
      }

      inline size_t dag_skipped_elements() const override final
      {
        return m_reduced_dag ? m_total_skipped : 0;
      }

      inline size_t dag_span_elements() const override final
      {
        //std::cout<<"dag_span_elements: m_reduced_dag="<<m_reduced_dag<<", m_total_count="<<m_total_count<<", span().nb_cells()="<<span().nb_cells()<<"\n";
        return m_reduced_dag ? m_total_count : span().nb_cells();
      }

    protected:
      inline ParallelTaskFunctorMem get_functors_mem() override final
      {
        return { & m_proxy.m_func , sizeof(m_proxy.m_func) , typeid(m_proxy.m_func).hash_code() , & m_proxy.m_cost , sizeof(m_proxy.m_cost) , typeid(m_proxy.m_cost).hash_code() };
      }
      inline const char* get_tag() const override final
      {
        return m_proxy.m_tag;
      }

      // ------------------ members --------------------
    private:
      friend struct DagSchedulerFifo;
      friend struct DagSchedulerHeap;
      friend struct DagSchedulerOMPDep;
      friend struct DagSchedulerGPU;
      friend struct SpanSchedulerGPU;
    
      PTaskProxyT m_proxy;

      //SpanT m_span;
      dag::WorkShareDAGAdapter< dag::WorkShareDAG2<ndims.value> > m_dag;
      dag::WorkShareDAGExecution<ndims.value> m_priv_dag_exe;
      decltype( m_priv_dag_exe.shallow_copy() ) m_dag_exe;
      dag::WorkShareDAGAdapter< dag::WorkShareDAG2<ndims.value> > m_filtered_dag;
      std::vector<uint64_t> m_costs;
      std::vector<ssize_t> m_reduce_map;
      //std::vector<size_t> m_task_reorder;
      memory::CudaMMVector<size_t> m_task_reorder;
      std::atomic<size_t> m_unlocked_task_idx;
#     ifdef ONIKA_DAG_PROFILE_FETCH_RETRY
      std::atomic<size_t> m_task_idx_retry;
#     endif

      std::string m_diag_base_name;      
      int m_diag_count = 0;
      int m_scheduler_thread_num = -1;
      size_t m_scheduler_ctxsw = 0;

      size_t m_total_skipped = 0;
      size_t m_total_count = 0;
      size_t m_total_cost = 0;
      bool m_costs_changed = false;
      bool m_reduced_dag = false;
    };

  }
}


