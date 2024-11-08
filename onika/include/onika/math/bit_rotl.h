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
#include <limits>

#if __cplusplus > 201703L // C++20 code
#include <bit>
#endif

namespace onika { namespace math
{

  template <typename _T>
  static inline constexpr _T bit_rotl (_T _v, int _b)
  {
#if __cplusplus > 201703L
    return std::rotl( v , int(b) );
#else
    using T = std::make_unsigned_t<_T>;
    T v = static_cast<T>(_v);

    static_assert(std::is_integral<T>::value, "rotate of non-integral type");
    static_assert(! std::is_signed<T>::value, "rotate of signed type");
    constexpr unsigned int num_bits {std::numeric_limits<T>::digits};
    static_assert(0 == (num_bits & (num_bits - 1)), "rotate value bit length not power of two");
    
    const unsigned int bdiv = 1 + ( std::abs(_b) / num_bits );
    const unsigned int b = static_cast<unsigned int>( _b + bdiv * num_bits );
    
    constexpr unsigned int count_mask {num_bits - 1};
    const unsigned int mb {b & count_mask};
    using promoted_type = typename std::common_type<int, T>::type;
    using unsigned_promoted_type = typename std::make_unsigned<promoted_type>::type;

    return static_cast<_T>(
             ((unsigned_promoted_type{v} << mb)
            | (unsigned_promoted_type{v} >> (-mb & count_mask)))
            );
#endif
  }

} }

