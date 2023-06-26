#pragma once

#include <onika/task/parallel_task.h>
#include <onika/task/parallel_task_executor.h>
#include <fstream>
// #define ONIKA_DAG_PROFILE_FETCH_RETRY 1

namespace onika
{

  namespace task
  {

    struct DagSchedulerFifo
    {

      struct FifoTaskBatch
      {
        uint64_t cost = 0;
        uint32_t n = 0;
        uint32_t s = 0;
      };

      template<class PTDapImpl>
      static inline void schedule_tasks( PTDapImpl * self , ParallelTask* pt )
      {
        // scheduling parameters
        assert( ! ParallelTaskConfig::dag_reorder() );
        const size_t num_threads = omp_get_max_threads();
        //const size_t target_total_tasks = num_threads * ParallelTaskConfig::dag_tasks_per_thread();
        //const size_t bulk_tasks            = static_cast<size_t>( num_threads * ParallelTaskConfig::dag_bulk_task_factor() );
        constexpr unsigned int starvation_threshold = 2; //ParallelTaskConfig::dag_starvation_rounds();
        
        // profiling counters
        self->m_scheduler_ctxsw = 0;
        self->m_scheduler_thread_num = omp_get_thread_num();

        // depend counters initialization
        //auto & final_dag = self->m_reduced_dag ? self->m_filtered_dag.m_dag : self->m_dag.m_dag ;
        const auto & dag = self->dep_graph_impl();
        self->m_dag_exe.reset_dep_countdown( dag , std::memory_order_relaxed );
        self->m_dag_exe.assert_consistency( dag );

        // task heap
        size_t n = self->m_dag_exe.number_of_items(); //m_dep_countdown.size();
        size_t total_submitted_tasks = 0;
        self->m_task_reorder.assign( n , std::numeric_limits<size_t>::max() );
        size_t * __restrict__ idx_heap = self->m_task_reorder.data();
        std::atomic<size_t>* idx_heap_v = (std::atomic<size_t>*) self->m_task_reorder.data();

        // maximum batch size is a tradeof between task aggregation and load balacing guarantee
        const size_t max_tasks_per_thread = std::max( size_t(n / num_threads) , size_t(1) );
        const unsigned int max_batch_tasks = std::min( static_cast<size_t>(ParallelTaskConfig::dag_max_batch_size()) , max_tasks_per_thread );
        const size_t target_total_tasks = num_threads * ParallelTaskConfig::dag_tasks_per_thread();
        const bool scheduler_yield = ParallelTaskConfig::dag_scheduler_yield();
        //const unsigned int init_batch_tasks = max_batch_tasks;

#       define ONIKA_SPAWN_BATCH() \
          ++ total_spawn_count; /*batch_cost_sum += task_batch.cost;*/ if(task_batch.n==max_batch_tasks) ++batch_full_count; \
          dag_spawn_task(self,pt,task_batch); \
          task_batch.s = heap_start; task_batch.n = 0; task_batch.cost = 0

        // current batch
        FifoTaskBatch task_batch = { 0 , 0 , 0 };

        // inital tasks
        size_t heap_cost_sum = 0;
        size_t heap_start = 0;
        size_t heap_end = 0;
        for(size_t i=0;i<n;i++)
        {
          if( self->m_dag_exe.dep_counter(i,std::memory_order_relaxed) == 0 )
          {
            assert( dag.item_dep_count(i) == 0 );
            idx_heap[heap_end++] = i;
            heap_cost_sum += self->m_costs[i];
          }
        }
        size_t total_task_cost = heap_cost_sum;
        
        //size_t unlocked_task_start = heap_end;
        self->m_unlocked_task_idx.store( heap_end , std::memory_order_release );
#       ifdef ONIKA_DAG_PROFILE_FETCH_RETRY
        self->m_task_idx_retry.store( 0 , std::memory_order_release );
#       endif

        size_t batch_cost_target = 1;
        if( target_total_tasks>0 && heap_end>0 ) batch_cost_target = ( heap_cost_sum * n ) / ( heap_end * target_total_tasks );

        size_t starvation_count = 0;
        unsigned int starvation_rounds = 0;
        unsigned int total_spawn_count = 0;
        unsigned int batch_flush_count = 0;
        unsigned int batch_full_count = 0;
        while( total_submitted_tasks < n )
        {
          auto unlocked_task_end = self->m_unlocked_task_idx.load(std::memory_order_acquire);
          size_t next_task = std::numeric_limits<size_t>::max();
          while( heap_end<unlocked_task_end && ( next_task = idx_heap_v[heap_end] .load(std::memory_order_acquire) ) != std::numeric_limits<size_t>::max() )
          {
            size_t next_task_cost = self->m_costs[next_task];
            heap_cost_sum += next_task_cost;
            total_task_cost += next_task_cost;
            idx_heap[heap_end++] = next_task; 
          }

          if( heap_start == heap_end )
          {
            ++ starvation_rounds;
            if( starvation_rounds >= starvation_threshold )
            {
              if( task_batch.n > 0 )
              {
                ++ batch_flush_count;
                ONIKA_SPAWN_BATCH();
              }
              else
              {
                starvation_rounds = 0;
                ++ starvation_count;
                if( scheduler_yield )
                {
#                 pragma omp taskyield
                }
              }
            }
          }
          else
          {
            //bool stop_spawn = false;
            //size_t nb=0;
            while( heap_start<heap_end /*&& !stop_spawn*/ )
            {
              size_t i = idx_heap[heap_start++];
              size_t tcost = self->m_costs[i];
              heap_cost_sum -= tcost;
              ++ total_submitted_tasks;
              ++ task_batch.n;
              task_batch.cost += tcost;
              if( task_batch.n == max_batch_tasks || task_batch.cost >= batch_cost_target )
              {
                //size_t ctxsw = self->m_scheduler_ctxsw;
                //++nb;
                ONIKA_SPAWN_BATCH();
                //stop_spawn = ( ctxsw != self->m_scheduler_ctxsw );
              }
            }
          }

        }

        // flush remaining batch, if any
        if( task_batch.n > 0 )
        {
          ONIKA_SPAWN_BATCH();
        }

#       undef ONIKA_SPAWN_BATCH

        if( ParallelTaskConfig::dag_diagnostics() )
        {
          std::ofstream fout ( self->m_diag_base_name + ".dag_schedule" );
          fout //<< "init_batch_tasks  = " << init_batch_tasks << std::endl
               << "max_batch_tasks   = " << max_batch_tasks << std::endl
               << "total_spawn_count = " << total_spawn_count << std::endl
               << "target_total_tasks= " << target_total_tasks << std::endl
               << "batch_flush_count = " << batch_flush_count << std::endl
               << "batch_full_count  = " << batch_full_count << std::endl
               << "batch_cost_target = " << batch_cost_target << std::endl
               << "avg batch cost    = " << total_task_cost*1.0/total_spawn_count << std::endl
               << "starvation_count  = " << starvation_count << std::endl
               << "m_scheduler_ctxsw = " << self->m_scheduler_ctxsw << std::endl
#           ifdef ONIKA_DAG_PROFILE_FETCH_RETRY
               << "m_task_idx_retry  = " << self->m_task_idx_retry.load(std::memory_order_acquire) << std::endl
#           endif
               << std::flush;
        }

        self->m_scheduler_thread_num = -1;
      }

      template<class PTDapImpl>
      static inline void dag_spawn_task( PTDapImpl* self, ParallelTask* pt , const FifoTaskBatch& batch )
      { 
#       pragma omp task default(none) firstprivate(batch,pt,self)
        {
          if( omp_get_thread_num() == self->m_scheduler_thread_num ) { ++ self->m_scheduler_ctxsw; }
          std::atomic<size_t>* idx_heap_v = (std::atomic<size_t>*) self->m_task_reorder.data();
          size_t n_retry_events=0;
          for(size_t i=0;i<batch.n;i++)
          {
            size_t task_idx = idx_heap_v[ batch.s+i ].load(std::memory_order_relaxed);
            size_t retry = 0;
            while( task_idx == std::numeric_limits<size_t>::max() )
            {
              ++ retry;
              task_idx = idx_heap_v[ batch.s+i ].load(std::memory_order_acquire);
            }
            if(retry>0) { ++ n_retry_events; }
            dag_execute_task(self,pt, task_idx );
          }
#         ifdef ONIKA_DAG_PROFILE_FETCH_RETRY
          if(n_retry_events>0) { self->m_task_idx_retry.fetch_add(n_retry_events,std::memory_order_relaxed); }
#         endif
        }
      }

      template<class PTDapImpl>
      static inline void dag_execute_task( PTDapImpl* self, ParallelTask* pt , size_t i )
      {        
        if constexpr ( PTDapImpl::SpanT::ndims >= 1 )
        {
          auto c = self->task_coord_int(pt,i); // properly scaled with grainsize and shifted with lower_bound
          self->execute(pt,c);
          for(auto d : self->m_dag_exe.item_out_deps(i)/*m_out_deps[i]*/)
          {
            int dcount = self->m_dag_exe.decrease_dep_counter(d,std::memory_order_relaxed);
            assert( dcount > 0 );
            if(dcount==1)
            {
              std::atomic<size_t>* idx_heap_v = (std::atomic<size_t>*) self->m_task_reorder.data();
              idx_heap_v[ self->m_unlocked_task_idx.fetch_add(1,std::memory_order_relaxed) ].store( d , std::memory_order_release );
            }
          }
        }
        if constexpr ( PTDapImpl::SpanT::ndims == 0 ){ std::abort(); }
      }
      
    };


  }
}

