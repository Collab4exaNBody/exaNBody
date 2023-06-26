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
