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

