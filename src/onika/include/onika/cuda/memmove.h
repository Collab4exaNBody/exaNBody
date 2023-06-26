#pragma once

#include <vector>
#include <onika/cuda/cuda.h>

namespace onika
{

  namespace cuda
  {

    // block wide cooperative memmove
    ONIKA_HOST_DEVICE_FUNC inline void cu_block_memmove( void* dst , const void* src , unsigned int n )
    {
      static_assert( sizeof(uint64_t) == 8 && sizeof(uint8_t) == 1 , "Wrong standard type sizes" );
            
      if( dst < src )
      {
        volatile uint8_t* dst1 = (uint8_t*) dst;
        const volatile uint8_t* src1 = (const uint8_t*) src;

        volatile uint64_t* dst8 = (uint64_t*) dst;
        const volatile uint64_t* src8 = (const uint64_t*) src;
        unsigned int n8 = n / 8;
        
        unsigned int i;
        for(i=0;i<n8;i+=ONIKA_CU_BLOCK_SIZE)
        {
          uint64_t x;
          const unsigned int j = i + ONIKA_CU_THREAD_IDX;
          if(j<n8) x = src8[j];
          ONIKA_CU_BLOCK_SYNC();
          if(j<n8) dst8[j] = x;
          ONIKA_CU_BLOCK_SYNC();
        }
        i *= 8;
        for(;i<n;i+=ONIKA_CU_BLOCK_SIZE)
        {
          uint8_t x;
          const unsigned int j = i + ONIKA_CU_THREAD_IDX;
          if(j<n) x = src1[j];
          ONIKA_CU_BLOCK_SYNC();
          if(j<n) dst1[j] = x;
          ONIKA_CU_BLOCK_SYNC();
        }
      }
      else if( dst > src )
      {
        // not implemented yet
        ONIKA_CU_ABORT();
      }
      // else { /* nothing to do */ }
    }

  }
}

