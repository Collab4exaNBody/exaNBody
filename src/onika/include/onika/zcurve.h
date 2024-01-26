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

#include <onika/oarray.h>
#include <onika/grid_grain.h>
#include <cstdlib>

#include <onika/cuda/cuda.h>

namespace onika
{
    template<class FuncT, size_t Nd, size_t GrainSizeLog2>
    ONIKA_HOST_DEVICE_FUNC static inline void z_order_apply( GridGrainPo2<Nd,GrainSizeLog2> , FuncT && f )
    {
      //static constexpr size_t GrainSize = 1ull << GrainSizeLog2 ;
      static constexpr size_t NCells = 1ull << (GrainSizeLog2*Nd);
      for(size_t c=0;c<NCells;c++)
      {
        onika::oarray_t<size_t,Nd> x = ZeroArray<size_t,Nd>::zero;
        size_t t = c;
        for(unsigned int b=0;b<GrainSizeLog2;b++)
        {
          for(unsigned int d=0;d<Nd;d++)
          {
            x[d] |= ( ( t & size_t(1) ) << b );
            t = t >> 1;
          }
        }
        f ( x );
      }
    }

}
