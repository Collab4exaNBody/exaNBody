#pragma once

#include <onika/cuda/cuda.h>
#include <onika/integral_constant.h>

namespace onika
{

  namespace cuda
  {

    template<class T, int BlockSize>
    ONIKA_HOST_DEVICE_FUNC
    static inline T block_reduce_add(const T& x , onika::IntConst<BlockSize> )
    {
      ONIKA_CU_BLOCK_SHARED UnitializedPlaceHolder<T> _sdata[ BlockSize ];
      T* sdata = (T*) _sdata;
      sdata[ ONIKA_CU_THREAD_IDX ] = x;
      ONIKA_CU_BLOCK_FENCE(); ONIKA_CU_BLOCK_SYNC();
      for (unsigned int s=BlockSize/2; s>0; s>>=1)
      {
        if (ONIKA_CU_THREAD_IDX < s)
        {
          sdata[ONIKA_CU_THREAD_IDX] += sdata[ONIKA_CU_THREAD_IDX + s];
        }
        ONIKA_CU_BLOCK_FENCE(); ONIKA_CU_BLOCK_SYNC();
      }
      return sdata[ 0 ];
    }
    
    template<class T, int BlockSize>
    ONIKA_HOST_DEVICE_FUNC
    static inline T block_reduce_min(const T& x , onika::IntConst<BlockSize> )
    {
      ONIKA_CU_BLOCK_SHARED UnitializedPlaceHolder<T> _sdata[ BlockSize ];
      T* sdata = (T*) _sdata;
      sdata[ ONIKA_CU_THREAD_IDX ] = x;
      ONIKA_CU_BLOCK_FENCE(); ONIKA_CU_BLOCK_SYNC();
      for (unsigned int s=BlockSize/2; s>0; s>>=1)
      {
        if (ONIKA_CU_THREAD_IDX < s)
        {
          if( sdata[ONIKA_CU_THREAD_IDX] > sdata[ONIKA_CU_THREAD_IDX + s] ) sdata[ONIKA_CU_THREAD_IDX] = sdata[ONIKA_CU_THREAD_IDX + s];
        }
        ONIKA_CU_BLOCK_FENCE(); ONIKA_CU_BLOCK_SYNC();
      }
      return sdata[ 0 ];
    }

    template<class T, int BlockSize>
    ONIKA_HOST_DEVICE_FUNC
    static inline T block_reduce_max(const T& x , onika::IntConst<BlockSize> )
    {
      ONIKA_CU_BLOCK_SHARED UnitializedPlaceHolder<T> _sdata[ BlockSize ];
      T* sdata = (T*) _sdata;
      sdata[ ONIKA_CU_THREAD_IDX ] = x;
      ONIKA_CU_BLOCK_FENCE(); ONIKA_CU_BLOCK_SYNC();
      for (unsigned int s=BlockSize/2; s>0; s>>=1)
      {
        if (ONIKA_CU_THREAD_IDX < s)
        {
          if( sdata[ONIKA_CU_THREAD_IDX] < sdata[ONIKA_CU_THREAD_IDX + s] ) sdata[ONIKA_CU_THREAD_IDX] = sdata[ONIKA_CU_THREAD_IDX + s];
        }
        ONIKA_CU_BLOCK_FENCE(); ONIKA_CU_BLOCK_SYNC();
      }
      return sdata[ 0 ];
    }

  }
}

