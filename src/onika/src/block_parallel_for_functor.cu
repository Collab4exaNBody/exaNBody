#include <onika/parallel/block_parallel_for_functor.h>

namespace onika
{
  namespace parallel
  {

    ONIKA_DEVICE_KERNEL_FUNC
    void finalize_functor_adapter( GPUKernelExecutionScratch* scratch )
    {
      if( ONIKA_CU_THREAD_IDX == 0 && ONIKA_CU_BLOCK_IDX == 0 )
      {
        reinterpret_cast<BlockParallelForGPUFunctor*>( scratch->functor_data ) -> ~BlockParallelForGPUFunctor();
      }
    }

  }
}


