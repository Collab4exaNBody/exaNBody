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
// #pragma xstamp_cuda_enable  // DO NOT REMOVE THIS LINE !!

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/block_parallel_for.h>

namespace exanb
{
    struct CheckCudaMemAccessFunctor
    {
      double * m_data = nullptr;
      size_t m_size = 0;
      ONIKA_HOST_DEVICE_FUNC inline void operator () ( uint64_t i ) const
      {
        if( ONIKA_CU_THREAD_IDX == 0 )
        {
          m_data[i] = m_data[i] + 1.0;
        }
      }
    };
}

namespace onika
{
  namespace parallel
  {
    template<>
    struct BlockParallelForFunctorTraits< exanb::CheckCudaMemAccessFunctor >
    {
      static inline constexpr bool CudaCompatible = true;
    };
  }
}

namespace exanb
{
  class CheckCudaMemAccess : public OperatorNode
  {
    ADD_SLOT( long , nsamples , INPUT , 4096 );

  public:

    // -----------------------------------------------
    // -----------------------------------------------
    inline void execute ()  override final
    {
      const size_t n = *nsamples;
      onika::memory::CudaMMVector<double> v( n , 0.0 );
      for(size_t i=0;i<n;i++) v[i] = i;
    
      CheckCudaMemAccessFunctor func = { v.data() , v.size() };
      onika::parallel::block_parallel_for( n , func , parallel_execution_context() );

      for(size_t i=0;i<n;i++)
      {
        if( v[i] != static_cast<double>(i+1) )
        {
          fatal_error() << "bad value at index "<<i<<std::endl;
        }
      }
      
      lout << "Cuda memory access is ok"<< std::endl;
    }

    // -----------------------------------------------
    // -----------------------------------------------
    inline std::string documentation() const override final
    {
      return R"EOF(
Tests unified memory acces accross CPU and GPU.
)EOF";
    }

  };

  // === register factories ===
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "check_cuda_mem_access", make_simple_operator< CheckCudaMemAccess > );
  }

}
