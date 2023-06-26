#pragma once

#include <cmath>
#include <tk/spline.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/ro_shallow_copy.h>
#include <onika/cuda/cuda_math.h>
#include <onika/cuda/stl_adaptors.h>

namespace tk
{
  // LennardJones Parameters
  struct ReadOnlyTkSpline
  {
    size_t m_x_size;
    double m_b0 = 0.0;
    double m_c0 = 0.0;
    const double* __restrict__ m_x = nullptr;
    const double* __restrict__ m_y = nullptr;
    const double* __restrict__ m_a = nullptr;
    const double* __restrict__ m_b = nullptr;
    const double* __restrict__ m_c = nullptr;

    ReadOnlyTkSpline() = default;
    ReadOnlyTkSpline(const ReadOnlyTkSpline&) = default;
    ReadOnlyTkSpline(ReadOnlyTkSpline&&) = default;
    ReadOnlyTkSpline& operator = (const ReadOnlyTkSpline&) = default;
    ReadOnlyTkSpline& operator = (ReadOnlyTkSpline&&) = default;

    inline ReadOnlyTkSpline( const spline& s )
      : m_x_size( s.m_x.size() )
      , m_b0( s.m_b0 )
      , m_c0( s.m_c0 )
      , m_x( s.m_x.data() )
      , m_y( s.m_y.data() )
      , m_a( s.m_a.data() )
      , m_b( s.m_b.data() )
      , m_c( s.m_c.data() )
      {}

    ONIKA_HOST_DEVICE_FUNC inline double operator() (double x) const
    {
      using onika::cuda::max;
      using onika::cuda::lower_bound;
      
      size_t n = m_x_size;
      // find the closest point m_x[idx] < x, idx=0 even if x<m_x[0]
      const double* it;
      it = lower_bound( m_x , m_x+n , x );
      int idx = max( int(it-m_x)-1, 0);

      double h=x-m_x[idx];
      double interpol;
      if(x<m_x[0]) {
          // extrapolation to the left
          interpol=(m_b0*h + m_c0)*h + m_y[0];
      } else if(x>m_x[n-1]) {
          // extrapolation to the right
          interpol=(m_b[n-1]*h + m_c[n-1])*h + m_y[n-1];
      } else {
          // interpolation
          interpol=((m_a[idx]*h + m_b[idx])*h + m_c[idx])*h + m_y[idx];
      }
      return interpol;
    }

  };

}

// specialize ReadOnlyShallowCopyType so ReadOnlyEwaldParms is the read only type for EwaldParms
namespace onika { namespace cuda {
  template<> struct ReadOnlyShallowCopyType< tk::spline > { using type = tk::ReadOnlyTkSpline; };
} }

