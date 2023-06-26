#pragma once

#include <onika/task/parallel_task.h>
#include <onika/task/parallel_task_executor.h>

namespace onika
{

  namespace task
  {

    struct DagSchedulerHeap
    {

      struct TaskBatch
      {
        static inline constexpr size_t MAX_TASKS = 13;
        uint64_t cost = 0;
        uint32_t n = 0;
        uint32_t ti[MAX_TASKS];
      };

      template<class PTDapImpl>
      static inline void schedule_tasks( PTDapImpl * self , ParallelTask* pt )
      {
        // scheduling parameters
        const bool reorder_tasks           = ParallelTaskConfig::dag_reorder();
        const size_t num_threads           = omp_get_max_threads();
        const size_t bulk_tasks            = static_cast<size_t>( num_threads * ParallelTaskConfig::dag_bulk_task_factor() );
        const size_t target_total_tasks    = static_cast<size_t>( num_threads * ParallelTaskConfig::dag_tasks_per_thread() );
        const bool scheduler_yield         = ParallelTaskConfig::dag_scheduler_yield();

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
        size_t heap_size = 0;
        self->m_task_reorder.resize( n , std::numeric_limits<size_t>::max() );
        size_t * __restrict__ idx_heap = self->m_task_reorder.data();
        std::atomic<size_t>* idx_heap_v = (std::atomic<size_t>*) self->m_task_reorder.data();

        TaskBatch task_batch = { 0 , 0 };

        // inital tasks
        size_t batch_cost_sum = 0;
        for(size_t i=0;i<n;i++)
        {
          if( self->m_dag_exe.dep_counter(i,std::memory_order_relaxed) == 0 )
          {
            assert( dag.item_dep_count(i) == 0 );
            idx_heap[heap_size++] = i;
            batch_cost_sum += self->m_costs[i];
          }
        }
        size_t unlocked_task_start = heap_size;
        self->m_unlocked_task_idx.store( unlocked_task_start , std::memory_order_release );
        
        size_t batch_cost_target = 1;
        if( target_total_tasks>0 && heap_size>0 ) batch_cost_target = ( batch_cost_sum * n ) / ( heap_size * target_total_tasks );
        // size_t initial_tasks = heap_size;
        
        auto idx_cmp = [self]( size_t a , size_t b ) -> bool
          {
            ssize_t unlock_time_a = self->m_costs[a];
            ssize_t unlock_time_b = self->m_costs[b];
            int oa = self->m_dag_exe.item_out_dep_count(a); // .m_out_deps[a].size();
            if(oa>0) unlock_time_a /= oa;
            int ob = self->m_dag_exe.item_out_dep_count(b); //.m_out_deps[b].size();
            if(ob>0) unlock_time_b /= ob;              
            return unlock_time_a > unlock_time_b;
          };

        if( reorder_tasks ) std::make_heap( idx_heap , idx_heap+heap_size , idx_cmp );

        // stdout_stream() << "initial heap size = "<<heap_size<<std::endl;
        // ssize_t flying_tasks = 0; //total_submitted_tasks - ( n - m_n_running_tasks.load(std::memory_order_acquire) );
        size_t starvation_count = 0;
        unsigned int starvation_rounds = 0;
        unsigned int total_spawn_count = 0;
        unsigned int batch_flush_count = 0;
        constexpr unsigned int starvation_threshold = 2;
        while( total_submitted_tasks < n )
        {
          auto unlocked_task_end = self->m_unlocked_task_idx.load(std::memory_order_acquire);
          size_t next_task = std::numeric_limits<size_t>::max();
          while( unlocked_task_start<unlocked_task_end && ( next_task = idx_heap_v[unlocked_task_start] .load(std::memory_order_acquire) ) != std::numeric_limits<size_t>::max() )
          {
            idx_heap[heap_size++] = next_task; ++unlocked_task_start; // idx_heap_v[unlocked_task_start++];
            // stdout_stream() << "push task "<<next_task<<", heap_size="<<heap_size<<", unlocked_task_start="<<unlocked_task_start <<", unlocked_task_end="<<unlocked_task_end<< std::endl;
            if( reorder_tasks ) std::push_heap( idx_heap , idx_heap+heap_size , idx_cmp );
          }
          
          if( heap_size == 0 )
          {
            ++ starvation_rounds;
            if( starvation_rounds >= starvation_threshold )
            {
              if( task_batch.n > 0 )
              {
                ++ batch_flush_count;
                ++ total_spawn_count;
                dag_spawn_task(self,pt,task_batch);
                task_batch.n = 0;
                task_batch.cost = 0;
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
            size_t spawn_count = 0;
            bool stop_spawn = false;
            while( heap_size>0 && spawn_count<bulk_tasks && !stop_spawn )
            {
              if( reorder_tasks ) std::pop_heap( idx_heap , idx_heap+heap_size , idx_cmp );
              auto i = idx_heap[--heap_size];
              ++total_submitted_tasks;
              
              task_batch.ti[ task_batch.n++ ] = i;
              task_batch.cost += self->m_costs[i];
              if( task_batch.n == TaskBatch::MAX_TASKS || task_batch.cost >= batch_cost_target )
              {
                ++ spawn_count;
                size_t ctxsw = self->m_scheduler_ctxsw;
                dag_spawn_task(self,pt,task_batch);
                stop_spawn = ( ctxsw != self->m_scheduler_ctxsw );
                task_batch.n = 0;
                task_batch.cost = 0;
              }
            }
            total_spawn_count += spawn_count;
          }
          
        }

        if( task_batch.n > 0 )
        {
          ++ total_spawn_count;
          dag_spawn_task(self,pt,task_batch);
          task_batch.n = 0;
          task_batch.cost = 0;
        }

        if( ParallelTaskConfig::dag_diagnostics() )
        {
          std::ofstream fout ( self->m_diag_base_name + ".dag_schedule" );
          fout << "total_spawn_count = " << total_spawn_count << std::endl
               << "target_total_tasks= " << target_total_tasks << std::endl
               << "batch_flush_count = " << batch_flush_count << std::endl
               << "batch_cost_target = " << batch_cost_target << std::endl
               << "starvation_count  = " << starvation_count << std::endl
               << "m_scheduler_ctxsw = " << self->m_scheduler_ctxsw << std::endl;
        }

        self->m_scheduler_thread_num = -1;
      }

      template<class PTDapImpl>
      static inline void dag_spawn_task( PTDapImpl* self, ParallelTask* pt , const TaskBatch& batch )
      {
        /*
        TaskBatch batch = _batch;
        for(size_t i=0;i<batch.n;i++)
        {
          for(auto d:self->m_dag_exe.m_out_deps[batch.ti[i]]) { ONIKA_FORCE_ASSERT( self->m_dag_exe.dep_counter(d) > 0 ); }
        }
        */
#       pragma omp task default(none) firstprivate(batch,pt,self)
        {
          if( omp_get_thread_num() == self->m_scheduler_thread_num ) { ++ self->m_scheduler_ctxsw; }
          for(size_t i=0;i<batch.n;i++)
          {
            dag_execute_task(self,pt, batch.ti[i] );
          }
        }
      }

      template<class PTDapImpl>
      static inline void dag_execute_task( PTDapImpl* self, ParallelTask* pt , size_t i )
      {        
        if constexpr ( PTDapImpl::SpanT::ndims >= 1 )
        {
          auto c = self->task_coord_int(pt,i); // properly scaled with grainsize and shifted with lower_bound
          self->execute(pt,c);
          for(auto d : self->m_dag_exe.item_out_deps(i) /*m_out_deps[i]*/ )
          {
            int dcount = self->m_dag_exe.decrease_dep_counter(d ,std::memory_order_relaxed );
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

