/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <onika/log.h>

#ifdef ONIKA_CUDA_VERSION
#include <onika/cuda/cuda_context.h>
#include <onika/cuda/cuda_error.h>
#endif

namespace exanb
{

  struct FinalizeCuda : public OperatorNode
  {
    ADD_SLOT( bool , enable_cuda , INPUT , REQUIRED );
 
    inline void execute () override final
    {
#     ifdef ONIKA_CUDA_VERSION
      auto cuda_ctx = global_cuda_ctx();
      if( cuda_ctx != nullptr && *enable_cuda && cuda_ctx->has_devices() )
      {
        if( *enable_cuda && cuda_ctx->has_devices() )
        {
          ONIKA_CU_CHECK_ERRORS( ONIKA_CU_DEVICE_SYNCHRONIZE() );
          for(const auto &dev : cuda_ctx->m_devices)
          {
            for(const auto& f : dev.m_finalize_destructors) f();
          }
          for(auto s:cuda_ctx->m_threadStream)
          {
            if( s != 0 )
            {
              ONIKA_CU_CHECK_ERRORS( ONIKA_CU_DESTROY_STREAM( s ) );
            }
          }
          cuda_ctx->m_threadStream.clear();
          cuda_ctx->m_threadStream.shrink_to_fit();
          cuda_ctx->m_devices.clear(); // calls destructors which frees Cuda resources
          cuda_ctx->m_devices.shrink_to_fit();
        }
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


