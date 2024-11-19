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
#include <onika/math/basic_types_operators.h>
#include <onika/math/quaternion.h>
#include <cmath>
#include <onika/cuda/cuda.h>
#include <random>

namespace onika { namespace math
{
  ONIKA_HOST_DEVICE_FUNC inline Quaternion operator - (Quaternion& q)
  {
    return Quaternion{ -q.w, -q.x, -q.y, -q.z };
  }

  ONIKA_HOST_DEVICE_FUNC inline Quaternion vabs(const Quaternion& q)
  {
    return Quaternion{ fabs(q.w), fabs(q.x), fabs(q.y), fabs(q.z) };
  }

  ONIKA_HOST_DEVICE_FUNC inline Quaternion operator - (const Quaternion& q)
  {
    return Quaternion{ -q.w, -q.x, -q.y, -q.z };
  }

  ONIKA_HOST_DEVICE_FUNC inline Quaternion operator - (const Quaternion& q, const Quaternion& r)
  {
    return Quaternion{ q.w-r.w, q.x-r.x, q.y-r.y, q.z-r.z };
  }

  ONIKA_HOST_DEVICE_FUNC inline Quaternion operator + (const Quaternion& q, const Quaternion& r)
  {
    return Quaternion{ q.w+r.w, q.x+r.x, q.y+r.y, q.z+r.z };
  }

  ONIKA_HOST_DEVICE_FUNC inline Quaternion operator * (const Quaternion& q, const Quaternion& r)
  {
    return Quaternion{ q.w*r.w, q.x*r.x, q.y*r.y, q.z*r.z };
  }

  ONIKA_HOST_DEVICE_FUNC inline Quaternion operator / (const Quaternion& q, const Quaternion& r)
  {
    return Quaternion{ q.w/r.w, q.x/r.x, q.y/r.y, q.z/r.z };
  }

  ONIKA_HOST_DEVICE_FUNC inline Quaternion operator * (const Quaternion& q, double scale)
  {
    return Quaternion{ q.w*scale, q.x*scale, q.y*scale, q.z*scale };
  }
  ONIKA_HOST_DEVICE_FUNC inline Quaternion operator / (const Quaternion& q, double scale)
  {
    return Quaternion{ q.w/scale, q.x/scale, q.y/scale, q.z/scale };
  }
  ONIKA_HOST_DEVICE_FUNC inline Quaternion operator * (double scale, const Quaternion& q)
  {
    return Quaternion{ q.w*scale, q.x*scale, q.y*scale, q.z*scale };
  }
  ONIKA_HOST_DEVICE_FUNC inline Quaternion operator / (double scale, const Quaternion& q)
  {
    return Quaternion{ q.w/scale, q.x/scale, q.y/scale, q.z/scale };
  }

  ONIKA_HOST_DEVICE_FUNC inline Quaternion& operator /= (Quaternion& q, double b)
  {
    q.w /= b;
    q.x /= b;
    q.y /= b;
    q.z /= b;
    return q;
  }
  
  ONIKA_HOST_DEVICE_FUNC inline Quaternion& operator += (Quaternion& q, Quaternion& r)
  {
    q.w += r.w;
    q.x += r.x;
    q.y += r.y;
    q.z += r.z;
    return q;
  }

  ONIKA_HOST_DEVICE_FUNC inline bool operator == (Quaternion& q1, Quaternion& q2)
  {
    if( (q1.w == q2.w) && (q1.x == q2.x) && (q1.y == q2.y) && (q1.z == q2.z) ) return true;
    return false; 
  }

  ONIKA_HOST_DEVICE_FUNC inline bool operator != (Quaternion& q1, Quaternion& q2)
  {
    if( (q1.w != q2.w) && (q1.x != q2.x) && (q1.y != q2.y) && (q1.z != q2.z) ) return true;
    return false; 
  }
  
  ONIKA_HOST_DEVICE_FUNC inline double norm(const Quaternion& q)
  {
    return ( sqrt( q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z ) );
  }

  ONIKA_HOST_DEVICE_FUNC inline Quaternion normalize(const Quaternion& q)
  {
    return ( q / norm( q ) );
  }

  ONIKA_HOST_DEVICE_FUNC inline Quaternion get_conjugated(const Quaternion& q)
  {
    Quaternion ret;
    ret.w = q.w;
    ret.x = -q.x;
    ret.y = -q.y;
    ret.z = -q.z;
    return ret;
  }

  /// @brief Return the time derivative of the quaternion \f$ \dot{q} = \frac{1}{2} \omega q\f$
  ONIKA_HOST_DEVICE_FUNC inline Quaternion dot(const Quaternion& q, const Vec3d& omega)
  {
    Quaternion ret;
    ret.x = 0.5 * (omega.y * q.z - omega.z * q.y + omega.x * q.w);
    ret.y = 0.5 * (omega.z * q.x - omega.x * q.z + omega.y * q.w);
    ret.z = 0.5 * (omega.x * q.y - omega.y * q.x + omega.z * q.w);
    ret.w = -0.5 *  dot( omega , Vec3d{q.x,q.y,q.z} ) ; // real dot product
    return ret;
  }

  /// @brief Return d2Q/dt2
  ONIKA_HOST_DEVICE_FUNC inline Quaternion ddot (const Quaternion& q, const Vec3d& vrot, const Vec3d& arot)
  {
    auto xyz = Vec3d{q.x,q.y,q.z};
    auto tmp1 = q.w * arot + cross(arot, xyz);
    auto tmp2 = - dot(arot , xyz);
    //quat q1(s * domega + cross(domega, v), -domega * v); // remark: ctor is quat(v, s)
    Quaternion q1 = {tmp1.x, tmp1.y, tmp1.z, tmp2}; 
    Vec3d c = cross(vrot, xyz);
    //quat q2(-(omega * v) * omega + cross(omega, s * omega + c), omega * (s * omega + c));
    tmp1 = - dot(vrot, xyz) * vrot + cross (vrot, q.w*vrot + c);
    tmp2 = dot(vrot , (q.w * vrot + c)); 
    Quaternion q2 = {tmp1.x, tmp1.y, tmp1.z, tmp2};
    q1 += (q2 = q2 * 0.5);
    q1 = q1 * 0.5;
    return q1;
  }

  // Rotation of a vector by a quat
  ONIKA_HOST_DEVICE_FUNC inline Vec3d operator * (const Quaternion& q, const Vec3d& v)
  {
    const auto xyz = Vec3d{q.x,q.y,q.z};
    auto qv = cross(xyz, v);
    auto qqv = cross(xyz, qv);
    return v + qv*2.0*q.w + qqv*2.0;
  }

  inline void randomize(Quaternion& q, bool seedTime = false) 
  {
    // @see http://hub.jmonkeyengine.org/t/random-quaternions/8431
    static std::default_random_engine engine;
    if (seedTime == true) engine.seed(std::chrono::system_clock::now().time_since_epoch().count());
    static std::uniform_real_distribution<double> distrib(-1.0, 1.0);
    double sum = 0.0;
    q.w = distrib(engine);
    sum += q.w * q.w;
    q.x = sqrt(1 - sum) * distrib(engine);
    sum += q.x * q.x;
    q.y = sqrt(1 - sum) * distrib(engine);
    sum += q.y * q.y;
    q.z = sqrt(1 - sum) * (distrib(engine) < 0.0 ? -1.0 : 1.0);
  }

} }

