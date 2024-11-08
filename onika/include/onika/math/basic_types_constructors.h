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

#include <onika/math/basic_types_def.h>
#include <onika/cuda/cuda.h>

#include <algorithm>
#include <cmath>
#include <array>

namespace onika { namespace math
{

  // ============================ constructors ===============================
  
  ONIKA_HOST_DEVICE_FUNC inline IJK make_ijk( ssize_t i, ssize_t j, ssize_t k )
  {
    return IJK { i, j, k };
  }

  ONIKA_HOST_DEVICE_FUNC inline IJK make_ijk( Vec3d p )
  {
    return make_ijk( static_cast<ssize_t>( floor(p.x) ), static_cast<ssize_t>( floor(p.y) ), static_cast<ssize_t>( floor(p.z) ) );
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d make_vec3d( const double x[3] )
  {
    return Vec3d{x[0],x[1],x[2]};
  }

  ONIKA_HOST_DEVICE_FUNC inline Mat3d make_mat3d( const Vec3d& a, const Vec3d& b, const Vec3d& c)
  {
    return Mat3d{a.x,b.x,c.x, a.y,b.y,c.y, a.z,b.z,c.z};
  }

  ONIKA_HOST_DEVICE_FUNC inline constexpr Mat3d make_identity_matrix()
  {
    return Mat3d { 1.,0.,0., 0.,1.,0., 0.,0.,1. };
  }

  ONIKA_HOST_DEVICE_FUNC inline constexpr Mat3d make_diagonal_matrix(const Vec3d& v)
  {
    return Mat3d { v.x,0.,0., 0.,v.y,0., 0.,0.,v.z };
  }

  ONIKA_HOST_DEVICE_FUNC inline constexpr Mat3d make_zero_matrix()
  {
    return Mat3d { 0.,0.,0., 0.,0.,0., 0.,0.,0. };
  }

} }

