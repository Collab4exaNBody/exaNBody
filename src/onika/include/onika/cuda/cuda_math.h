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

    template<class T> struct numeric_limits;
    template<> struct numeric_limits<uint16_t> { static inline constexpr uint16_t max = std::numeric_limits<uint16_t>::max(); };
    template<> struct numeric_limits<uint32_t> { static inline constexpr uint32_t max = std::numeric_limits<uint32_t>::max(); };
    template<> struct numeric_limits<uint64_t> { static inline constexpr uint64_t max = std::numeric_limits<uint64_t>::max(); };
    template<> struct numeric_limits<double>   { static inline constexpr double infinity = std::numeric_limits<double>::infinity(); };
    
    template<class T> ONIKA_HOST_DEVICE_FUNC inline const T& clamp( const T& x, const T& lo , const T& hi)
    {
      return x<lo ? lo : ( x>hi ? hi : x ) ;
    }

  }

}

