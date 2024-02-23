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

#pragma once

#include <onika/parallel/block_parallel_for.h>

namespace onika
{

  namespace parallel
  {  

    template<class T>
    struct MemSetFunctor
    {
      T * __restrict__ m_data = nullptr;
      size_t m_nb_elements = 0;
      T m_value = {};  
      ONIKA_HOST_DEVICE_FUNC inline void operator () ( size_t block_idx ) const
      {
        size_t i = block_idx * ONIKA_CU_BLOCK_SIZE + ONIKA_CU_THREAD_IDX;
        if( i < m_nb_elements ) m_data[i] = m_value;
      }
    };

    template<class T> struct BlockParallelForFunctorTraits< MemSetFunctor<T> >
    {      
      static inline constexpr bool CudaCompatible = true;
    };

    template<class T>
    static inline ParallelExecutionWrapper block_parallel_memset( T * data , uint64_t N, const T& value, ParallelExecutionContext * pec )
    {    
      using FuncT = MemSetFunctor<T>;
      BlockParallelForOptions opts = {};
      opts.omp_scheduling = OMP_SCHED_STATIC;

      FuncT memset_func = { data , N , value };

      size_t block_size = 1;
      bool allow_cuda_exec = ( pec->m_cuda_ctx != nullptr );
      if( allow_cuda_exec ) allow_cuda_exec = pec->m_cuda_ctx->has_devices();
      if( allow_cuda_exec )
      {
        block_size = std::min( static_cast<size_t>(ONIKA_CU_MAX_THREADS_PER_BLOCK) , static_cast<size_t>(onika::parallel::ParallelExecutionContext::gpu_block_size()) );
      }
      size_t nb_blocks = ( N + block_size - 1 ) / block_size ;
     
      return block_parallel_for( nb_blocks , memset_func , pec , opts );
    }

  }

}

