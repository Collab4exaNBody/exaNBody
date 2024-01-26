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

#include <vector>
#include <onika/cuda/cuda.h>

namespace onika
{

  namespace cuda
  {

    // block wide cooperative memmove
    ONIKA_HOST_DEVICE_FUNC inline void cu_block_memcpy( void* dst , const void* src , unsigned int n )
    {
      static_assert( sizeof(uint64_t) == 8 && sizeof(uint8_t) == 1 , "Wrong standard type sizes" );
            
      uint8_t* __restrict__  dst1 = (uint8_t*) dst;
      const uint8_t* __restrict__ src1 = (const uint8_t*) src;
      
      // non aliasing check
      assert( (dst1+n) <= src1 || dst1 >= (src1+n) );

      uint64_t* __restrict__ dst8 = (uint64_t*) dst;
      const uint64_t* __restrict__ src8 = (const uint64_t*) src;
      unsigned int n8 = n / 8;
      
      unsigned int i = ONIKA_CU_THREAD_IDX;
      for( ; i<n8 ; i+=ONIKA_CU_BLOCK_SIZE) dst8[i] = src8[i];
      i *= 8;
      for( ; i<n  ; i+=ONIKA_CU_BLOCK_SIZE) dst1[i] = src1[i];
    }
    
  }
}

