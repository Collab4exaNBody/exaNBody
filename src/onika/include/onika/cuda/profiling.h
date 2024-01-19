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

#include <cuda_runtime.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_error.h>

namespace onika
{

  namespace cuda
  {

    static inline constexpr unsigned int CUDA_MAX_SM_COUNT = 1024;

    struct SMOverlapCounters
    {
      int sm_overlap_counter[CUDA_MAX_SM_COUNT];
      int sm_max_overlapped_blocks[CUDA_MAX_SM_COUNT];
      volatile int execute_or_profile;
    };

    template<class FuncT, class ... Args >
    ONIKA_DEVICE_KERNEL_FUNC void max_overlapped_blocks_per_sm_kernel( SMOverlapCounters* d_counters, FuncT func , Args ... args )
    {
      const unsigned int smid = onika::cuda::get_smid();

      if( d_counters->execute_or_profile == 0 && ONIKA_CU_THREAD_IDX == 0 )
      {
        const int overlap_count = 1 + ONIKA_CU_ATOMIC_ADD( d_counters->sm_overlap_counter[smid] , 1 );
        ONIKA_CU_ATOMIC_MAX( d_counters->sm_max_overlapped_blocks[smid] , overlap_count );
      }
      
      if( d_counters->execute_or_profile == 1 ) // never executed, but the compiler cannot know that
      {
        func( args ... );
      }
      
      if( d_counters->execute_or_profile == 0 && ONIKA_CU_THREAD_IDX == 0 )
      {
       ONIKA_CU_ATOMIC_ADD( d_counters->sm_overlap_counter[smid] , -1 );
      } 
    }

    template<class FuncT, class ... Args >
    inline int max_overlapped_blocks_per_sm(unsigned int BlockSize, FuncT func , Args ... args)
    {
      SMOverlapCounters * d_counters = nullptr;
      checkCudaErrors( cudaMalloc(&d_counters , sizeof(SMOverlapCounters) ) );
      checkCudaErrors( cudaMemset(d_counters, 0, sizeof(SMOverlapCounters) ) );

      max_overlapped_blocks_per_sm_kernel<<< CUDA_MAX_SM_COUNT*8 , BlockSize >>>( d_counters, func, args... );
      checkCudaErrors( cudaDeviceSynchronize() );
      
      SMOverlapCounters h_counters;
      checkCudaErrors( cudaMemcpy(&h_counters, d_counters, sizeof(SMOverlapCounters), cudaMemcpyDeviceToHost) );
      checkCudaErrors( cudaFree(d_counters) );

      int max_overlap = 0;
      for(unsigned int i=0;i<CUDA_MAX_SM_COUNT;i++)
      {
        max_overlap = std::max( max_overlap , h_counters.sm_max_overlapped_blocks[i] );
      }
      return max_overlap;
    }

  }
}

