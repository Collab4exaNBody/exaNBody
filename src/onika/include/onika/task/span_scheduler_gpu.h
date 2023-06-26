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

    template<class ProxyT, class AbstractSpanT>
    ONIKA_DEVICE_KERNEL_FUNC
    void cuda_execute_span_tasks(ProxyT proxy , AbstractSpanT span , unsigned long long n, unsigned long long* task_counter, bool coarse_coord)
    {
      static constexpr size_t Nd = ProxyT::Span::ndims;
      
      ONIKA_CU_BLOCK_SHARED unsigned long long task_index;
      do
      {
        if( ONIKA_CU_THREAD_IDX == 0 )
        {
          task_index = ONIKA_CU_ATOMIC_ADD( *task_counter , 1 , ONIKA_CU_MEM_ORDER_RELAXED );
        }
        ONIKA_CU_BLOCK_SYNC();

        if( task_index < n )
        {
          auto c = coarse_coord ? span.template coarse_index_to_coord_base<Nd>(task_index) : span.template index_to_coord<Nd>(task_index);
          ptask_execute_kernel(proxy,c);
        }
      }
      while( task_index < n );

      ONIKA_CU_BLOCK_SYNC();
      ONIKA_CU_SYSTEM_FENCE();
    }

    struct SpanSchedulerGPU
    {

      template<class PTDapImpl>
      static inline void schedule_tasks( PTDapImpl * self , ParallelTask* pt , bool coarse_coord )
      {
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

        // refresh Cuda current devive, just in case
        //static constexpr int gpu_index = 0;
        checkCudaErrors( ONIKA_CU_SET_DEVICE( cuda_ctx->m_devices[ 0 /*gpu_index*/ ].device_id ) );

        // profiling counters
        self->m_scheduler_ctxsw = 0;
        self->m_scheduler_thread_num = omp_get_thread_num();

        // task heap
        auto custream = cuda_ctx->m_threadStream[0];
        
        const unsigned int BlockSize = ParallelTaskConfig::gpu_block_size();
        const unsigned int GridSize = cuda_ctx->m_devices[0].m_deviceProp.multiProcessorCount * ParallelTaskConfig::gpu_sm_mult() + ParallelTaskConfig::gpu_sm_add();
//        std::cout << "Span Executor : Using "<< GridSize << 'x' << BlockSize << " GPU threads\n";
        
        const size_t heap_storage_size = 16;
        self->m_task_reorder.resize( heap_storage_size );

        static_assert( sizeof(unsigned long long) == sizeof(std::remove_reference_t<decltype(self->m_task_reorder[0])>) , "unsigned long long does not match size_t" );
        unsigned long long * heap_storage = reinterpret_cast<unsigned long long *>( self->m_task_reorder.data() );

        ONIKA_CU_MEMSET( heap_storage , 0 , sizeof(decltype(self->m_task_reorder[0])) * heap_storage_size , custream );
        ONIKA_CU_LAUNCH_KERNEL(GridSize,BlockSize,0,custream, cuda_execute_span_tasks , self->m_proxy , pt->span() , pt->m_num_tasks , heap_storage, coarse_coord );
        checkCudaErrors( ONIKA_CU_STREAM_SYNCHRONIZE(custream) );

        pt->account_completed_task( pt->m_num_tasks );
      }
      
    };


  }
}

