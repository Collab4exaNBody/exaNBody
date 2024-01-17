#pragma once

//#include <vector>
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

