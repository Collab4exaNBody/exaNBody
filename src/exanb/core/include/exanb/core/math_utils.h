#pragma once

#include <type_traits>
#include <cstdint>
#include <cmath>
#include <exanb/core/basic_types_def.h>

namespace exanb
{

  ONIKA_HOST_DEVICE_FUNC static inline unsigned int icbrt(unsigned int x)
  {
    return static_cast<unsigned int>( round( cbrt( double(x) ) ) );
  }

  // integer cubical root
  // needs testing and perf measurement before it's really used
  ONIKA_HOST_DEVICE_FUNC static inline uint32_t icbrt64(uint64_t x) noexcept
  {
    int s;
    uint32_t y;
    uint64_t b;
    y = 0;
    for (s = 63; s >= 0; s -= 3) {
      y += y;
      b = 3*y*((uint64_t) y + 1) + 1;
      if ((x >> s) >= b) {
        x -= b << s;
        y++;
      }
    }
    return y;
  }

  Vec3d unitvec_uv( double u, double v );

  void matrix_scale_min_max( const Mat3d& A, double& fmin, double& fmax);

  void symmetric_matrix_eigensystem(const Mat3d& A, Vec3d eigenvectors[3], double eigenvalues[3]);
}

