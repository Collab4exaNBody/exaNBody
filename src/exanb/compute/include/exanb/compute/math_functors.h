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

#include <onika/cuda/cuda.h>
#include <exanb/core/basic_types_def.h>

// definition of a virtual field, a.k.a a field combiner
namespace exanb
{
  struct Vec3Norm2Functor
  {
    ONIKA_HOST_DEVICE_FUNC inline double operator () (double vx, double vy, double vz) const { return vx*vx+vy*vy+vz*vz; }
    ONIKA_HOST_DEVICE_FUNC inline double operator () (const Vec3d& v) const { return v.x*v.x+v.y*v.y+v.z*v.z; }
  };

  struct Vec3NormFunctor
  {
    ONIKA_HOST_DEVICE_FUNC inline double operator () (double vx, double vy, double vz) const { return sqrt(vx*vx+vy*vy+vz*vz); }
    ONIKA_HOST_DEVICE_FUNC inline double operator () (const Vec3d& v) const { return sqrt(v.x*v.x+v.y*v.y+v.z*v.z); }
  };

  struct Vec3FromXYZFunctor
  {
    ONIKA_HOST_DEVICE_FUNC inline Vec3d operator () (double vx, double vy, double vz) const { return {vx,vy,vz}; }
  };

  template<int A=1, int B=1>
  struct ConstRealFraction
  {
    using value_type = double;
    static constexpr double value = static_cast<double>(A) / static_cast<double>(B);
  };
  using ConstReal1 = ConstRealFraction<1,1>;

  template<class ConstType>
  struct ConstantFunctor
  {
    using type = typename ConstType::value_type;
    static constexpr type value = ConstType::value;
    ONIKA_HOST_DEVICE_FUNC inline type operator () () const
    {
      return value;
    }
  };

  template<class ValueType>
  struct UniformValueFunctor
  {
    const ValueType value = {};
    ONIKA_HOST_DEVICE_FUNC inline ValueType operator () () const
    {
      return value;
    }
  };

}


