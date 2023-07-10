#pragma once

#include <onika/task/parallel_task_config.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_error.h>
#include <onika/cuda/device_storage.h>
#include <onika/soatl/field_id.h>
#include <onika/parallel/parallel_execution_context.h>

#ifdef XSTAMP_OMP_NUM_THREADS_WORKAROUND
#include <omp.h>
#endif

namespace onika
{

  namespace parallel
  {

    // this template is here to know if compute buffer must be built or computation must be ran on the fly
    template<class FuncT> struct BlockParallelForFunctorTraits
    {
      static inline constexpr bool CudaCompatible = false;
    };

    template< class FuncT>
    ONIKA_DEVICE_KERNEL_FUNC
    ONIKA_DEVICE_KERNEL_BOUNDS(ONIKA_CU_MAX_THREADS_PER_BLOCK,ONIKA_CU_MIN_BLOCKS_PER_SM)
    void block_parallel_for_gpu_kernel( uint64_t N, const FuncT func, GPUKernelExecutionScratch* scratch )
    {
      // avoid use of compute buffer when possible
      ONIKA_CU_BLOCK_SHARED unsigned int i;
      do
      {
        if( ONIKA_CU_THREAD_IDX == 0 )
        {
          i = ONIKA_CU_ATOMIC_ADD( scratch->counters[0] , 1u );
          //printf("processing cell #%d\n",int(cell_a_no_gl));
        }
        ONIKA_CU_BLOCK_SYNC();
        if( i < N ) { func( i ); }
      }
      while( i < N );
    }

    template<class FuncT>
    inline void block_parallel_for_omp_kernel(
        uint64_t N
      , const FuncT& func
      , unsigned int num_tasks = 0
      , GPUKernelExecutionContext * async_exec_ctx = nullptr
      )
    {
    
      if( num_tasks == 0 )
      {
#       ifdef XSTAMP_OMP_NUM_THREADS_WORKAROUND
        omp_set_num_threads( omp_get_max_threads() );
#       endif
#       pragma omp parallel
        {
#         pragma omp for schedule(dynamic)
          for(uint64_t i=0;i<N;i++) { func( i ); }
        }
      }
      else if( async_exec_ctx != nullptr )
      {
        // enclose a taskgroup inside a task, so that we can wait for a single task which itself waits for the completion of the whole taskloop
        [[maybe_unused]] auto & depvar = *async_exec_ctx;
#       pragma omp task depend(inout:depvar)
        {
          // implicit taskgroup, ensures taskloop has completed before enclosing task ends
#         pragma omp taskloop num_tasks(num_tasks)
          for(uint64_t i=0;i<N;i++) { func( i ); }
        }
      }
      else
      {
          // implicit taskgroup ensures loop completed at end of structured block.
#         pragma omp taskloop num_tasks(num_tasks)
          for(uint64_t i=0;i<N;i++) { func( i ); }
      }
    }

    template< class FuncT >
    static inline void block_parallel_for(
        uint64_t N
      , const FuncT& func
      , GPUKernelExecutionContext * exec_ctx = nullptr
      , bool enable_gpu = true
      , bool async = false )
    {
      if constexpr ( BlockParallelForFunctorTraits<FuncT>::CudaCompatible )
      {
        bool allow_cuda_exec = enable_gpu && ( exec_ctx != nullptr );
        if( allow_cuda_exec ) allow_cuda_exec = ( exec_ctx->m_cuda_ctx != nullptr );
        if( allow_cuda_exec ) allow_cuda_exec = exec_ctx->m_cuda_ctx->has_devices();
        if( allow_cuda_exec )
        {
          exec_ctx->check_initialize();
          const unsigned int BlockSize = std::min( static_cast<size_t>(ONIKA_CU_MAX_THREADS_PER_BLOCK) , static_cast<size_t>(onika::task::ParallelTaskConfig::gpu_block_size()) );
          const unsigned int GridSize = exec_ctx->m_cuda_ctx->m_devices[0].m_deviceProp.multiProcessorCount * onika::task::ParallelTaskConfig::gpu_sm_mult()
                                      + onika::task::ParallelTaskConfig::gpu_sm_add();

          auto custream = exec_ctx->m_cuda_stream;        
          exec_ctx->record_start_event();
          exec_ctx->reset_counters();
          auto * scratch = exec_ctx->m_cuda_scratch.get();

          ONIKA_CU_LAUNCH_KERNEL(GridSize,BlockSize,0,custream, block_parallel_for_gpu_kernel, N, func, scratch );
          
          if( ! async ) { exec_ctx->wait(); }        

          return;
        }
      }

      int prefered_num_tasks = 0;
      if( exec_ctx != nullptr ) prefered_num_tasks = exec_ctx->m_omp_num_tasks;

      // allow tasking mode, means we're in a parallel/single[/taskgroup] scope
      if( prefered_num_tasks > 0 )
      {
        int num_tasks = prefered_num_tasks * onika::task::ParallelTaskConfig::gpu_sm_mult() + onika::task::ParallelTaskConfig::gpu_sm_add() ;
        block_parallel_for_omp_kernel( N , func , num_tasks , async ? exec_ctx : nullptr );        
      }
      else
      {
        block_parallel_for_omp_kernel( N , func );
      }
    }

  }

}

