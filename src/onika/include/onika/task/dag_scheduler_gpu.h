#pragma once

#include <onika/task/parallel_task.h>
#include <onika/task/parallel_task_executor.h>
#include <onika/task/parallel_task_config.h>
#include <onika/cuda/cuda_context.h>
#include <onika/cuda/stl_adaptors.h>
#include <onika/cuda/cuda_error.h>
#include <type_traits>

namespace onika
{

  namespace task
  {

#   define ONIKA_CU_BLOCK_CALL_SINGLE_BCAST_RET( r , expr ) \
      do { \
        ONIKA_CU_BLOCK_SHARED std::remove_cv_t<std::remove_reference_t<decltype(r)> > s_##r; \
        if( ONIKA_CU_THREAD_IDX == 0 ) s_##r = expr ; \
        ONIKA_CU_BLOCK_SYNC(); \
        r = s_##r;\
      } while(false)

    template<class ProxyT, class DagExeT, class coord_t>
    ONIKA_DEVICE_KERNEL_FUNC
    void cuda_intialize_dag_execution(ProxyT proxy , DagExeT dag_exe , unsigned long long * task_heap_storage, const coord_t * dag_coords)
    {
      int n = dag_exe.number_of_items();    
//      unsigned long long & heap_start = * (task_heap_storage+0);
      unsigned long long & heap_end   = * (task_heap_storage+1);
      unsigned long long * task_heap  =    task_heap_storage+2;
      
      const size_t GlobalThreadIdx = ONIKA_CU_BLOCK_IDX * ONIKA_CU_BLOCK_SIZE + ONIKA_CU_THREAD_IDX;
      const size_t GlobalThreadCount = ONIKA_CU_BLOCK_SIZE * ONIKA_CU_GRID_SIZE;
      
      for(size_t ti=GlobalThreadIdx; ti<n; ti+=GlobalThreadCount )
      {
        if( dag_exe.dep_counter(ti,ONIKA_CU_MEM_ORDER_ACQUIRE) == 0 )
        {
          unsigned long long task_index = ONIKA_CU_ATOMIC_ADD( heap_end , 1 , ONIKA_CU_MEM_ORDER_RELAXED );
          ONIKA_CU_ATOMIC_STORE( task_heap[task_index] , ti+1 , ONIKA_CU_MEM_ORDER_RELEASE );
        }
      }

    }

    template<class ProxyT, class DagExeT, class coord_t>
    ONIKA_DEVICE_KERNEL_FUNC
    void cuda_execute_dag_tasks(ProxyT proxy , DagExeT dag_exe , unsigned long long* task_heap_storage, const coord_t * dag_coords)
    {
      using SpanT = typename ProxyT::Span;
      static constexpr size_t Nd = coord_t::array_size ;
      static constexpr unsigned int grainsize = SpanT::grainsize;
      int n = dag_exe.number_of_items();
      
      unsigned long long & heap_start = * (task_heap_storage+0);
      unsigned long long & heap_end   = * (task_heap_storage+1);
      unsigned long long * task_heap  =    task_heap_storage+2;

      ONIKA_CU_BLOCK_SHARED unsigned int terminate;      
      ONIKA_CU_BLOCK_SHARED unsigned long long next_task;
//      if( ONIKA_CU_THREAD_IDX == 0 ) { terminate = 0; }
//      ONIKA_CU_BLOCK_SYNC();

      do
      {
        ONIKA_CU_BLOCK_SHARED unsigned long long task_index;

        if( ONIKA_CU_THREAD_IDX == 0 )
        {
          next_task = ONIKA_CU_ATOMIC_ADD( heap_start , 1 , ONIKA_CU_MEM_ORDER_RELAXED );
//          size_t hend = ONIKA_CU_ATOMIC_LOAD( heap_end , ONIKA_CU_MEM_ORDER_ACQUIRE );
          if( next_task < n )
          {
            task_index = ONIKA_CU_ATOMIC_LOAD( task_heap[next_task] , ONIKA_CU_MEM_ORDER_ACQUIRE );
            while( task_index == 0 )
            {
              //ONIKA_CU_NANOSLEEP(10);
              task_index = ONIKA_CU_ATOMIC_LOAD( task_heap[next_task] , ONIKA_CU_MEM_ORDER_ACQUIRE );
            }
            -- task_index;
          }
          terminate = ( next_task >= n );
        }
        ONIKA_CU_BLOCK_SYNC();

        if( next_task < n )
        {
          auto c = dag_coords[task_index]; for(size_t k=0;k<Nd;k++) c[k] = c[k] * grainsize + proxy.m_span.lower_bound[k];
          ptask_execute_kernel(proxy,c);
          ONIKA_CU_DEVICE_FENCE(); ONIKA_CU_BLOCK_SYNC(); 
          if( ONIKA_CU_THREAD_IDX == 0 )
          {
            dag_exe.set_dep_counter( task_index , 999 );
            auto [dstart,dend] = dag_exe.item_out_deps(task_index);
            for(auto di=dstart; di!=dend; ++di)
            {
              auto d = *di;
              if( dag_exe.decrease_dep_counter( d , ONIKA_CU_MEM_ORDER_RELAXED ) == 1 )
              {
                unsigned long long task_index = ONIKA_CU_ATOMIC_ADD( heap_end , 1 , ONIKA_CU_MEM_ORDER_RELAXED );
                ONIKA_CU_ATOMIC_STORE( task_heap[task_index] , d+1 , ONIKA_CU_MEM_ORDER_RELEASE );              
              }
            }
          }

        }
      }
      while( ! terminate );

      ONIKA_CU_SYSTEM_FENCE(); ONIKA_CU_BLOCK_SYNC();
/*      
      if( ONIKA_CU_BLOCK_IDX==0 && ONIKA_CU_THREAD_IDX==0 )
      {
        for(size_t i=0;i<n;i++)
        {
          auto dc = dag_exe.dep_counter(i);
          if( dc != 999 )
          {
            printf("task %d not processed, depcount=%d\n",int(i),int(dc));
          }
        }
      }
*/
    }

    struct DagSchedulerGPU
    {

      template<class PTDapImpl>
      static inline void schedule_tasks( PTDapImpl * self , ParallelTask* pt )
      {
        // scheduling parameters
        assert( ! ParallelTaskConfig::dag_reorder() );
        assert( pt != nullptr );
        
        auto ptq = pt->m_ptq;
        if( ptq == nullptr )
        {
          ptq = & ParallelTaskQueue::global_ptask_queue();
        }        
        assert( ptq != nullptr );

        auto * cuda_ctx = ptq->cuda_ctx();

        if( cuda_ctx == nullptr )
        {
          std::cerr<<"No Cuda context available, abort.\n";
          std::abort();
        }

        if( cuda_ctx->m_devices.empty() )
        {
          std::cerr<<"No cuda device available, abort.\n";
          std::abort();
        }

        // profiling counters
        self->m_scheduler_ctxsw = 0;
        self->m_scheduler_thread_num = omp_get_thread_num();

        // depend counters initialization
        //auto & final_dag = self->m_reduced_dag ? self->m_filtered_dag.m_dag : self->m_dag.m_dag ;
        const auto & dag = self->dep_graph_impl();
        self->m_dag_exe.reset_dep_countdown( dag , ONIKA_CU_MEM_ORDER_RELAXED );
        self->m_dag_exe.assert_consistency( dag );

        // task heap
        size_t n = self->m_dag_exe.number_of_items(); //m_dep_countdown.size();
        auto custream = cuda_ctx->m_threadStream[0];
        
        const unsigned int BlockSize = ParallelTaskConfig::gpu_block_size();
        const unsigned int GridSize = cuda_ctx->m_devices[0].m_deviceProp.multiProcessorCount * ParallelTaskConfig::gpu_sm_mult() + ParallelTaskConfig::gpu_sm_add();
        // std::cout << "DAG Executor : Using "<< GridSize << 'x' << BlockSize << " GPU threads" <<std::endl << std::flush;
        
        const size_t heap_storage_size = n + 4;
        self->m_task_reorder.resize( heap_storage_size );

        static_assert( sizeof(unsigned long long) == sizeof(std::remove_reference_t<decltype(self->m_task_reorder[0])>) , "unsigned long long does not match size_t" );
        unsigned long long * heap_storage = reinterpret_cast<unsigned long long *>( self->m_task_reorder.data() );
        
        ONIKA_CU_MEMSET( heap_storage , 0 , sizeof(decltype(self->m_task_reorder[0])) * heap_storage_size , custream );
        ONIKA_CU_LAUNCH_KERNEL(GridSize,BlockSize,0,custream, cuda_intialize_dag_execution , self->m_proxy , self->m_dag_exe , heap_storage , dag.m_coords.data() );
        checkCudaErrors( ONIKA_CU_STREAM_SYNCHRONIZE(custream) );
        ONIKA_CU_LAUNCH_KERNEL(GridSize,BlockSize,0,custream, cuda_execute_dag_tasks , self->m_proxy , self->m_dag_exe , heap_storage , dag.m_coords.data() );
        checkCudaErrors( ONIKA_CU_STREAM_SYNCHRONIZE(custream) );

        // std::cout << "DAG Executor : finished" <<std::endl << std::flush;

        pt->account_completed_task( n );
      }
      
    };


  }
}

