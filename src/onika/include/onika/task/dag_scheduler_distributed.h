#pragma once

#include <onika/task/parallel_task.h>
#include <onika/task/parallel_task_executor.h>

namespace onika
{

  namespace task
  {

    struct DagSchedulerDistributed
    {

      struct TaskBatchBuffer
      {
        static inline constexpr size_t MAX_TASKS = 13;
        uint64_t cost = 0;
        uint32_t n = 0;
        uint32_t ti[MAX_TASKS];
      };

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
        //constexpr unsigned int starvation_threshold = 2; //ParallelTaskConfig::dag_starvation_rounds();
        
        // profiling counters
        //self->m_scheduler_ctxsw = 0;
        //self->m_scheduler_thread_num = omp_get_thread_num();

        // depend counters initialization
        auto & final_dag = self->m_reduced_dag ? self->m_filtered_dag.m_dag : self->m_dag.m_dag ;
        self->m_dag_exe.reset_dep_countdown( final_dag , std::memory_order_relaxed );
        self->m_dag_exe.assert_consistency( final_dag );

        // task heap
        size_t n = self->m_dag_exe.number_of_items(); //m_dep_countdown.size();
        //size_t total_submitted_tasks = 0;
        self->m_task_reorder.assign( n , std::numeric_limits<size_t>::max() );
        size_t * __restrict__ idx_heap = self->m_task_reorder.data();
        //std::atomic<size_t>* idx_heap_v = (std::atomic<size_t>*) self->m_task_reorder.data();

        // maximum batch size is a tradeof between task aggregation and load balacing guarantee
        const size_t max_tasks_per_thread  = std::max( size_t(n / num_threads) , size_t(1) );
        const unsigned int max_batch_tasks = std::min( static_cast<size_t>(ParallelTaskConfig::dag_max_batch_size()) , max_tasks_per_thread );
        const size_t target_total_tasks    = static_cast<size_t>( num_threads * ParallelTaskConfig::dag_tasks_per_thread() );
        const size_t bulk_tasks            = static_cast<size_t>( num_threads * ParallelTaskConfig::dag_bulk_task_factor() );

        // inital tasks
        const auto & dag = self->dep_graph_impl();
        size_t heap_cost_sum = 0;
        size_t heap_start = 0;
        size_t heap_end = 0;
        for(size_t i=0;i<n;i++)
        {
          if( dag.item_dep_count(i)==0 )
          {
            idx_heap[heap_end++] = i;
            heap_cost_sum += /*self->m_cost_func( pt , self->task_coord_raw(i) ) .cost;*/ self->m_costs[i];
          }
        }

        // batch structure
        FifoTaskBatch task_batch = { 0 , 0 , 0 };

        // spawn initial task batchs
        const size_t target_batch_cost = ( heap_cost_sum * n ) / ( heap_end * target_total_tasks );
        const size_t init_target_batch_cost = heap_cost_sum / bulk_tasks;
        size_t init_spawn_count = 0;
        while(heap_start<heap_end)
        {
          size_t i = idx_heap[heap_start++];
          size_t tcost = /*self->m_cost_func( pt , self->task_coord_raw(i) ) .cost;*/ self->m_costs[i];
          heap_cost_sum -= tcost;
          ++ task_batch.n;
          task_batch.cost += tcost;
          if( task_batch.n >= max_batch_tasks || task_batch.cost >= init_target_batch_cost )
          {
            ++ init_spawn_count;
            dag_spawn_task(self,pt,task_batch,target_batch_cost);
            task_batch.n = 0;
            task_batch.s = heap_start;
            task_batch.cost = 0;
          }
        }
        if( task_batch.n > 0 )
        {
          ++ init_spawn_count;
          dag_spawn_task(self,pt,task_batch,target_batch_cost);
        }

        if( ParallelTaskConfig::dag_diagnostics() )
        {
          std::ofstream fout ( self->m_diag_base_name + ".dag_schedule" );
          fout << "target_total_tasks     = " << target_total_tasks << std::endl
               << "init_spawn_count       = " << init_spawn_count << std::endl
               << "avg tasks/batch        = " << heap_end*1.0 / init_spawn_count << std::endl
               << "target_batch_cost      = " << target_batch_cost << std::endl
               << "init_target_batch_cost = " << init_target_batch_cost << std::endl;
        }

      }

      template<class PTDapImpl>
      static inline void dag_spawn_task( PTDapImpl* self, ParallelTask* pt , const FifoTaskBatch& batch , size_t target_batch_cost )
      { 
#       pragma omp task default(none) firstprivate(batch,pt,self,target_batch_cost)
        {
          TaskBatchBuffer buf = { 0 , 0 };
          /*std::atomic<size_t>**/ volatile const size_t * idx_heap_v = /*(std::atomic<size_t>*) */ (volatile const size_t *) self->m_task_reorder.data();
          for(size_t i=0;i<batch.n;i++)
          {
            size_t task_idx = idx_heap_v[ batch.s + i ]; //.load(std::memory_order_relaxed);
            if( task_idx == std::numeric_limits<size_t>::max() )
            {
              stdout_stream() <<"DagSchedulerDistributed: Internal error"<<std::endl;
              std::abort();
            }
            dag_execute_task(self,pt, task_idx, buf , target_batch_cost );
          }
          if(buf.n>0) dag_execute_task_buf(self,pt,buf,target_batch_cost);
        }
      }

      template<class PTDapImpl>
      static inline void dag_execute_task( PTDapImpl* self, ParallelTask* pt , size_t i , TaskBatchBuffer& buf, size_t target_batch_cost )
      {        
        if constexpr ( PTDapImpl::SpanT::ndims >= 1 )
        {
          auto c = self->task_coord_int(pt,i); // properly scaled with grainsize and shifted with lower_bound
          self->execute(pt,c);
          for(auto d:self->m_dag_exe.item_out_deps(i)/*m_out_deps[i]*/)
          {
            int dcount = self->m_dag_exe.decrease_dep_counter(d,std::memory_order_relaxed);
            assert( dcount > 0 );
            if(dcount==1)
            {
              buf.ti[ buf.n ++ ] = d;
              buf.cost += /*self->m_cost_func( pt , self->task_coord_raw(d) ) .cost;*/ self->m_costs[d];
              if( buf.n == TaskBatchBuffer::MAX_TASKS || buf.cost >= target_batch_cost )
              {
                dag_spawn_task_buf(self,pt,buf,target_batch_cost);
                buf.n = 0;
                buf.cost = 0;
              }
            }
          }
        }
        if constexpr ( PTDapImpl::SpanT::ndims == 0 ){ std::abort(); }
      }

      template<class PTDapImpl>
      static inline void dag_spawn_task_buf( PTDapImpl* self, ParallelTask* pt , const TaskBatchBuffer& batch , size_t target_batch_cost )
      { 
#       pragma omp task default(none) firstprivate(batch,pt,self,target_batch_cost)
        {
          dag_execute_task_buf(self,pt,batch,target_batch_cost);
        }
      }

      template<class PTDapImpl>
      static inline void dag_execute_task_buf( PTDapImpl* self, ParallelTask* pt , const TaskBatchBuffer& batch , size_t target_batch_cost )
      { 
        TaskBatchBuffer buf = { 0 , 0 };
        for(size_t i=0;i<batch.n;i++)
        {
          size_t task_idx = batch.ti[i];
          dag_execute_task(self,pt, task_idx, buf , target_batch_cost );
        }
        if(buf.n>0) dag_execute_task_buf(self,pt,buf,target_batch_cost);
      }

    };


  }
}

