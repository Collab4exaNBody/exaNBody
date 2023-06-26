#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>

#ifdef XSTAMP_CUDA_VERSION
#include <onika/cuda/cuda_context.h>
#include <cuda_runtime.h>
#include <onika/cuda/cuda_error.h>
#endif

namespace exanb
{

  struct FinalizeCuda : public OperatorNode
  {
#   ifdef XSTAMP_CUDA_VERSION
    ADD_SLOT( onika::cuda::CudaContext , cuda_ctx , INPUT_OUTPUT );
#   endif

    inline bool is_sink() const override final { return true; } // not a suppressable operator
 
    inline void execute () override final
    {
#     ifdef XSTAMP_CUDA_VERSION
      if( cuda_ctx.has_value() ) if( cuda_ctx->has_devices() )
      {
        checkCudaErrors( cudaDeviceSynchronize() );
        for(const auto &dev : cuda_ctx->m_devices)
        {
          for(const auto& f : dev.m_finalize_destructors) f();
        }
        for(auto s:cuda_ctx->m_threadStream)
        {
          if( s != 0 )
          {
            checkCudaErrors( cudaStreamDestroy( s ) );
          }
        }
        cuda_ctx->m_threadStream.clear();
        cuda_ctx->m_threadStream.shrink_to_fit();
        cuda_ctx->m_devices.clear(); // calls destructors which frees Cuda resources
        cuda_ctx->m_devices.shrink_to_fit();
      }
      ptask_queue().set_cuda_ctx( nullptr );
#     endif
    }
  };
  
  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "finalize_cuda", make_compatible_operator< FinalizeCuda > );
  }

}


