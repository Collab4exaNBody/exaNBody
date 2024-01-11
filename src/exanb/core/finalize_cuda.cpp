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
    ADD_SLOT( bool , enable_cuda , INPUT , REQUIRED );
 
    inline void execute () override final
    {
#     ifdef XSTAMP_CUDA_VERSION
      auto cuda_ctx = global_cuda_ctx();
      if( cuda_ctx != nullptr && *enable_cuda && cuda_ctx->has_devices() )
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
      set_global_cuda_ctx( nullptr );
#     endif
    }
  };
  
  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "finalize_cuda", make_compatible_operator< FinalizeCuda > );
  }

}


