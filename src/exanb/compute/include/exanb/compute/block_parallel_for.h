#pragma once

#include <onika/task/parallel_task_config.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_error.h>
#include <onika/cuda/device_storage.h>
#include <onika/soatl/field_id.h>
#include <exanb/field_sets.h>

#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/gpu_execution_context.h>

#ifdef XSTAMP_OMP_NUM_THREADS_WORKAROUND
#include <omp.h>
#endif

namespace exanb
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
    uint64_t N,
    const FuncT& func )
  {
#   ifdef XSTAMP_OMP_NUM_THREADS_WORKAROUND
    omp_set_num_threads( omp_get_max_threads() );
#   endif

#   pragma omp parallel
    {
#     pragma omp for schedule(dynamic)
      for(uint64_t i=0;i<N;i++)
      {
        func( i );
      }
    }
  }

  template< class FuncT >
  static inline void block_parallel_for( uint64_t N, const FuncT& func, GPUKernelExecutionContext * exec_ctx = nullptr, bool async = false )
  {
    if constexpr ( BlockParallelForFunctorTraits<FuncT>::CudaCompatible )
    {
      bool allow_cuda_exec = ( exec_ctx != nullptr );
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

    block_parallel_for_omp_kernel(N, func);
  }

}

