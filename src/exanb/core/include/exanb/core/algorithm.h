#pragma once

#include <cstdlib>
#include <omp.h>
#include <assert.h>

namespace exanb
{

  // clamp is only available in c++17
  template<typename T>
  static inline const T&
  clamp( const T& x, const T& lower, const T& upper )
  {
    if( x < lower ) return lower;
    else if( upper < x ) return upper;
    else return x;
  }

  // exclusive prefix sum
  template<typename T>
  static inline void exclusive_prefix_sum(T* a, size_t N)
  {
    T s = 0;
    for(size_t i=0;i<N;i++)
    {
      T ns = s+a[i];
      a[i] = s;
      s = ns;
    }
  }
  
}

