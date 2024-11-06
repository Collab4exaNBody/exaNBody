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

#include <cmath>
#include <onika/cuda/cuda.h>

namespace onika
{

  namespace cuda
  {

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)

#   define onika_fast_sincosf(t,s,c) __sincosf(t,s,c)

#else

#   define onika_fast_sincosf(t,s,c) do{ *(s) = std::sin(t); *(c) = std::cos(t); }while(false)

    static_assert( uint16_t(-1) == std::numeric_limits<uint16_t>::max() , "incorrect method for max uint16_t" );
    static_assert( uint32_t(-1) == std::numeric_limits<uint32_t>::max() , "incorrect method for max uint32_t" );
    static_assert( uint64_t(-1) == std::numeric_limits<uint64_t>::max() , "incorrect method for max uint64_t" );
    
#endif

    template<class T> ONIKA_HOST_DEVICE_FUNC inline const T& max( const T& a, const T& b ) { return a>b ? a : b ; }
    template<class T> ONIKA_HOST_DEVICE_FUNC inline const T& min( const T& a, const T& b ) { return a<b ? a : b ; }

    // numeric limits adapated as class static constexpr for warning less access in device code
    template<class T> struct numeric_limits;
    template<> struct numeric_limits<uint16_t>
    {
      static inline constexpr uint16_t max = std::numeric_limits<uint16_t>::max();
    };
    template<> struct numeric_limits<uint32_t>
    {
      static inline constexpr uint32_t max = std::numeric_limits<uint32_t>::max();
    };
    template<> struct numeric_limits<uint64_t>
    {
      static inline constexpr uint64_t max = std::numeric_limits<uint64_t>::max();
    };
    template<> struct numeric_limits<float>
    {
      static inline constexpr float infinity = std::numeric_limits<float>::infinity();
      static inline constexpr float quiet_NaN = std::numeric_limits<float>::quiet_NaN();
    };
    template<> struct numeric_limits<double>
    {
      static inline constexpr double infinity = std::numeric_limits<double>::infinity();
      static inline constexpr double quiet_NaN = std::numeric_limits<double>::quiet_NaN();
    };
    
    template<class T> ONIKA_HOST_DEVICE_FUNC inline const T& clamp( const T& x, const T& lo , const T& hi)
    {
      return x<lo ? lo : ( x>hi ? hi : x ) ;
    }

  }

}

