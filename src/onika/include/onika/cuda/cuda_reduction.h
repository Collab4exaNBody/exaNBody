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

