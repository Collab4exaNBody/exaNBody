#include <onika/parallel/block_parallel_for.h>

namespace onika
{
  namespace parallel
  {

    ONIKA_DEVICE_KERNEL_FUNC
    ONIKA_DEVICE_KERNEL_BOUNDS(ONIKA_CU_MAX_THREADS_PER_BLOCK,ONIKA_CU_MIN_BLOCKS_PER_SM)
    void block_parallel_for_gpu_kernel_workstealing( uint64_t N, GPUKernelExecutionScratch* scratch )
    {
      // avoid use of compute buffer when possible
      const auto & func = * reinterpret_cast<BlockParallelForGPUFunctor*>( scratch->functor_data );
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

    ONIKA_DEVICE_KERNEL_FUNC
    ONIKA_DEVICE_KERNEL_BOUNDS(ONIKA_CU_MAX_THREADS_PER_BLOCK,ONIKA_CU_MIN_BLOCKS_PER_SM)
    void block_parallel_for_gpu_kernel_regulargrid( GPUKernelExecutionScratch* scratch )
    {
      const auto & func = * reinterpret_cast<BlockParallelForGPUFunctor*>( scratch->functor_data );
      func( ONIKA_CU_BLOCK_IDX );
    }


  }
}


