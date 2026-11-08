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
#include <onika/math/quaternion.h>
#include <onika/oarray.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

namespace exanb
{

  // A copy of some per-particle fields, stored column-wise in their native
  // scalar type. Each column holds 'count' elements of 'shape' scalars each,
  // in C order (e.g. a Mat3d field has shape {3,3}, stored m11,m12,...,m33).
  // dtype follows numpy's notation: kind ('f','i','u','b') and byte size, e.g. "f8".
  struct ParticleFieldColumn
  {
    std::string dtype;
    std::vector<size_t> shape;
    std::vector<uint8_t> data;
  };

  struct ParticleFieldBundle
  {
    size_t count = 0;
    std::map<std::string, ParticleFieldColumn> columns;
  };

  // Describes how a particle field value type maps onto a flat array of scalars.
  // supported is false for types that cannot be exported as plain numbers.
  template<class T, class = void> struct ParticleFieldLayout
  {
    static constexpr bool supported = false;
  };

  template<class T> struct ParticleFieldLayout< T , std::enable_if_t< std::is_arithmetic_v<T> > >
  {
    static constexpr bool supported = true;
    using scalar_type = T;
    static std::vector<size_t> shape() { return {}; }
  };

  template<> struct ParticleFieldLayout< onika::math::Vec3d >
  {
    static constexpr bool supported = true;
    using scalar_type = double;
    static std::vector<size_t> shape() { return {3}; }
  };

  template<> struct ParticleFieldLayout< onika::math::Mat3d >
  {
    static constexpr bool supported = true;
    using scalar_type = double;
    static std::vector<size_t> shape() { return {3,3}; }
  };

  template<> struct ParticleFieldLayout< onika::math::Quaternion >
  {
    static constexpr bool supported = true;
    using scalar_type = double; // w, x, y, z
    static std::vector<size_t> shape() { return {4}; }
  };

  template<class T, size_t N> struct ParticleFieldLayout< onika::oarray_t<T,N> , std::enable_if_t< std::is_arithmetic_v<T> > >
  {
    static constexpr bool supported = true;
    using scalar_type = T;
    static std::vector<size_t> shape() { return {N}; }
  };

  template<class T> inline std::string particle_field_dtype()
  {
    const char kind = std::is_same_v<T,bool> ? 'b' : std::is_floating_point_v<T> ? 'f' : std::is_signed_v<T> ? 'i' : 'u';
    return std::string(1,kind) + std::to_string( sizeof(T) );
  }

}
