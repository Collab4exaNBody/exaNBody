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

#include <cstdlib>
#include <cmath>
#include <onika/cuda/cuda.h>

namespace exanb
{  
  struct IJK
  {
    ssize_t i = 0;
    ssize_t j = 0;
    ssize_t k = 0;
    IJK() = default;
    ONIKA_HOST_DEVICE_FUNC inline IJK(ssize_t _i, ssize_t _j, ssize_t _k) : i(_i), j(_j), k(_k) {}
  };

  struct GridBlock
  {
    IJK start;
    IJK end;
  };

  struct Complexd
  {
    double r=0.0;
    double i=0.0;
  };

  struct Vec3d
  {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    ONIKA_HOST_DEVICE_FUNC inline bool operator < (const Vec3d& p) const
    {
      if(x<p.x) return true;
      if(x>p.x) return false;
      if(y<p.y) return true;
      if(y>p.y) return false;
      if(z<p.z) return true;
      return false;
    }
  };

  struct Plane3d
  {
    Vec3d N;  // plane normal
    double D; // distance to the origin
  };

  struct Mat3d
  {
    /* m_l_c (l=line, c=column */
    double m11 = 0.0;
    double m12 = 0.0;
    double m13 = 0.0;
    double m21 = 0.0;
    double m22 = 0.0;
    double m23 = 0.0;
    double m31 = 0.0;
    double m32 = 0.0;
    double m33 = 0.0;
  };

  struct AABB
  {
    Vec3d bmin;
    Vec3d bmax;
  };

  // Fake Mat3d type, to avoid computing anything
  struct FakeMat3d
  {
    FakeMat3d()=default;
    FakeMat3d(const FakeMat3d&)=default;
    FakeMat3d(FakeMat3d&&)=default;
    ONIKA_HOST_DEVICE_FUNC inline FakeMat3d(const Mat3d&) {}
    template<class T> ONIKA_HOST_DEVICE_FUNC inline FakeMat3d& operator = (const T&) { return *this; }
  };

}

