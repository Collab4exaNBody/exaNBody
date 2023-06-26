#pragma once

#include <onika/oarray.h>
#include <cstdlib>

namespace onika
{

   template<size_t N> struct IsPo2
    {
      static constexpr bool value = (N%2==0) && IsPo2<N/2>::value;
      static constexpr unsigned int log2 = 1 + IsPo2<N/2>::log2;
    };
    template<> struct IsPo2<0> { static constexpr bool value=false; static constexpr unsigned int log2 = 0; };
    template<> struct IsPo2<1> { static constexpr bool value=true;  static constexpr unsigned int log2 = 0; };
    template<size_t N> static inline constexpr bool is_po2_v = IsPo2<N>::value;
    template<size_t N> static inline constexpr unsigned int po2_log2_v = IsPo2<N>::log2;

    template<size_t _Nd, size_t _GrainSizeLog2>
    struct GridGrainPo2
    {
      static inline constexpr size_t Nd = _Nd;
      static inline constexpr size_t GrainSizeLog2 = _GrainSizeLog2;
      static inline constexpr size_t GrainSize = 1ull << GrainSizeLog2;    
      static inline constexpr size_t NCells = 1ull << ( GrainSizeLog2 * Nd );
    };

    template<size_t _Nd, ssize_t _GrainSize=-1>
    struct GridGrain
    {
      static inline constexpr size_t Nd = _Nd;
      static inline constexpr size_t GrainSize = _GrainSize;
    };

    template<size_t _Nd>
    struct GridGrain<_Nd,-1>
    {
      static inline constexpr size_t Nd = _Nd;
      size_t GrainSize = 0;
    };

    template<size_t Nd, ssize_t GS>
    static inline oarray_t<size_t,Nd> subgridcoord( const GridGrain<Nd,GS> ggrid, const oarray_t<size_t,Nd>& c, const oarray_t<size_t,Nd>& gc)
    {
      oarray_t<size_t,Nd> r;
      for(size_t i=0;i<Nd;i++) r[i] = c[i] * ggrid.GrainSize + gc[i];
      return r;
    }

    template<class GG, class FuncT>
    static inline void grid_grain_apply( GG g , FuncT && f )
    {
      if ( g.Nd >0 && g.GrainSize > 0 )
      {
        oarray_t<size_t,g.Nd> c = ZeroArray<size_t,g.Nd>::zero;
        while( c[g.Nd-1] < g.GrainSize )
        {
          f( c );
          ++ c[0];
          size_t i = 0;
          while( i<(g.Nd-1) && c[i]==g.GrainSize ) { c[i]=0; ++i; ++c[i]; }
        }
      }
    }
    
}
