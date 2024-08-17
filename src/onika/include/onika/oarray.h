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
#include <string_view>
#include <cstdlib>
#include <functional>
#include <onika/cuda/cuda.h>
#include <array>

namespace onika
{
  template <size_t Nd> using ConstDim = std::integral_constant<size_t,Nd>;

  template<class T, size_t N>
  struct oarray_t
  {
    static inline constexpr size_t array_size = N;
    using value_type = T;
    T x[N];
    ONIKA_HOST_DEVICE_FUNC inline T& operator [] (size_t i) { return x[i]; }
    ONIKA_HOST_DEVICE_FUNC inline const T& operator [] (size_t i) const { return x[i]; }
    ONIKA_HOST_DEVICE_FUNC inline T * data() { return x; }
    ONIKA_HOST_DEVICE_FUNC inline const T * data() const { return x; }
    static inline constexpr size_t size() { return N; }
    ONIKA_HOST_DEVICE_FUNC inline bool operator == (const oarray_t<T,N>& v) const
    {
      bool eq = true;
      for(size_t i=0;i<N;i++) if(x[i]!=v.x[i]) eq=false;
      return eq;
    }
    ONIKA_HOST_DEVICE_FUNC inline bool operator != (const oarray_t<T,N>& v) const
    {
      bool eq = false;
      for(size_t i=0;i<N;i++) if(x[i]!=v.x[i]) eq=true;
      return eq;
    }
    ONIKA_HOST_DEVICE_FUNC inline bool operator < (const oarray_t<T,N>& v) const
    {
      for(size_t i=0;i<N;i++)
      {
        if( x[i]<v.x[i] ) return true;
        else if( x[i]>v.x[i] ) return false;
      }
      return false;
    }
    ONIKA_HOST_DEVICE_FUNC inline bool operator > (const oarray_t<T,N>& v) const
    {
      for(size_t i=0;i<N;i++)
      {
        if( x[i]>v.x[i] ) return true;
        else if( x[i]<v.x[i] ) return false;
      }
      return false;
    }
    ONIKA_HOST_DEVICE_FUNC inline bool operator == (const T* v) const
    {
      bool eq = true;
      for(size_t i=0;i<N;i++) if(x[i]!=v[i]) eq=false;
      return eq;
    }
    ONIKA_HOST_DEVICE_FUNC inline const T* begin() const { return x; }
    ONIKA_HOST_DEVICE_FUNC inline T* begin() { return x; }
    ONIKA_HOST_DEVICE_FUNC inline const T* end() const { return x+N; }
    ONIKA_HOST_DEVICE_FUNC inline T* end() { return x+N; }
    
    template<class U, class = std::enable_if_t< std::is_convertible_v<U,T> > >
    static inline constexpr oarray_t from(const std::array<U,N>& sa)
    {
      oarray_t oa{}; for(size_t i=0;i<N;i++) oa.x[i] = sa[i]; return oa;
    }

    template<class U>
    static inline constexpr oarray_t from_ijk(const U& c)
    {
      oarray_t oa; const auto & [i,j,k] = c; oa.x[0]=T(i); oa.x[1]=T(j); oa.x[2]=T(k); return oa;
    }

    inline constexpr std::array<value_type,array_size> to_std_array() const
    {
      std::array<value_type,array_size> a = {};
      for(size_t i=0;i<array_size;i++) a[i] = x[i];
      return a;
    }

  };

  template<class T, size_t N> struct ZeroArray;
  template<class T> struct ZeroArray<T,0> { static inline constexpr oarray_t<T,0> zero {}; };
  template<class T> struct ZeroArray<T,1> { static inline constexpr oarray_t<T,1> zero { { static_cast<T>(0) } }; };
  template<class T> struct ZeroArray<T,2> { static inline constexpr oarray_t<T,2> zero { { static_cast<T>(0), static_cast<T>(0) } }; };
  template<class T> struct ZeroArray<T,3> { static inline constexpr oarray_t<T,3> zero { { static_cast<T>(0), static_cast<T>(0), static_cast<T>(0) } }; };
  template<class T> struct ZeroArray<T,4> { static inline constexpr oarray_t<T,3> zero { { static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0) } }; };

  template<class T, size_t N> static inline constexpr oarray_t<T,N> make_zero_array()
  {
    oarray_t<T,N> a{};
    if constexpr (N!=0) for(size_t i=0;i<N;i++) a.x[i] = static_cast<T>(0);
    return a;
  }

  template<class A> struct is_onika_array : public std::false_type {};
  template<class T, size_t N> struct is_onika_array< oarray_t<T,N> > : public std::true_type {};
  template<class A> static inline constexpr bool is_onika_array_v = is_onika_array<A>::value;

  using null_array_t = oarray_t<std::nullptr_t,0>;
  template<class T, size_t N, class = std::enable_if_t< std::is_trivially_copyable_v<T> > >
  static inline oarray_t<T,N>& initialize_array( oarray_t<T,N>& to , const oarray_t<T,N>& from )
  {
    return to=from ;
  }
  
  template<class T, size_t N, class = std::enable_if_t< std::is_arithmetic_v<T> > >
  static inline oarray_t<T,N>& initialize_array( oarray_t<T,N>& to , null_array_t from )
  {
    for(size_t i=0;i<N;i++) to.x[i] = static_cast<T>(0);
    return to;
  }

  template<class T, size_t N >
  static inline oarray_t<T,N>& initialize_array( oarray_t<T,N>& to , const T* from )
  {
    for(size_t i=0;i<N;i++) to.x[i] = from[i];
    return to;
  }

  template<class T, class U, size_t N, class = std::enable_if_t< std::is_convertible_v<T,U> > >
  static inline void convert_array( const oarray_t<T,N>& from , oarray_t<U,N>& to )
  {
    for(size_t i=0;i<N;i++) to.x[i] = static_cast<U>( from.x[i] );
  }

  template<class T, class U, size_t N, class = std::enable_if_t< std::is_convertible_v<T,U> > >
  static inline void convert_array( const T* from , oarray_t<U,N>& to )
  {
    if constexpr (N!=0) for(size_t i=0;i<N;i++) to.x[i] = static_cast<U>( from[i] );
  }

  template<class T, size_t N, class U=T, class = std::enable_if_t< std::is_integral_v<T> && std::is_integral_v<U> > >
  ONIKA_HOST_DEVICE_FUNC static inline size_t coord_to_index( const oarray_t<U,N>& coord , const oarray_t<T,N>& domain )
  {
    size_t S = 0;
    if constexpr (N!=0)
    {
      size_t M = 1;
      for(size_t i=0;i<N;i++)
      {
        S += coord.x[i] * M;
        M *= domain.x[i];
      }
    }
    return S;
  }

  template<class T, size_t N, class U, class = std::enable_if_t< std::is_integral_v<T> && std::is_integral_v<U> > >
  static inline size_t coord_to_index( const oarray_t<T,N>& coord , const U* domain )
  {
    oarray_t<T,N> a{}; convert_array(domain,a);
    return coord_to_index(coord,a);
  }


  template<class T, size_t N , class = std::enable_if_t< std::is_integral_v<T> > >
  ONIKA_HOST_DEVICE_FUNC static inline /*constexpr*/ oarray_t<T,N> index_to_coord( size_t i , const oarray_t<T,N>& domain )
  {
    oarray_t<T,N> c{};
    for(size_t k=0;k<N;k++)
    {
      auto j = i / domain.x[k];
      c[k] = i % domain.x[k];
      i = j;
    }
    return c;
  }

  template<class T, size_t N , class = std::enable_if_t< std::is_arithmetic_v<T> > >
  ONIKA_HOST_DEVICE_FUNC static inline /*constexpr*/ std::conditional_t< std::is_integral_v<T> , ssize_t , double > domain_size( const oarray_t<T,N>& domain )
  {
    if constexpr ( N > 0 )
    {
      std::conditional_t< std::is_integral_v<T> , ssize_t , double > r = domain.x[0];
      for(size_t i=1;i<N;i++) r *= domain.x[i];
      return r;
    }
    return 0;
  }

  template<class T, size_t N , class = std::enable_if_t< std::is_arithmetic_v<T> > >
  static inline constexpr oarray_t<T,N> array_min( const oarray_t<T,N>& A , const oarray_t<T,N>& B )
  {
    oarray_t<T,N> R{};
    if constexpr (N>0) for(size_t k=0;k<N;k++) R.x[k] = std::min( A.x[k] , B.x[k] );
    return R;
  }

  // alternative version that allows a 0D array to be a neutral element
  template<class T, size_t N , class = std::enable_if_t< std::is_arithmetic_v<T> && N!=0 > >
  static inline constexpr oarray_t<T,N> array_min( const oarray_t<T,N>& A , oarray_t<T,0> ) { return A; }

  template<class T, size_t N , class = std::enable_if_t< std::is_arithmetic_v<T> > >
  static inline constexpr oarray_t<T,N> array_max( const oarray_t<T,N>& A , const oarray_t<T,N>& B )
  {
    oarray_t<T,N> R{};
    if constexpr (N>0) for(size_t k=0;k<N;k++) R.x[k] = std::max( A.x[k] , B.x[k] );
    return R;
  }

  // alternative version that allows a 0D array to be a neutral element
  template<class T, size_t N , class = std::enable_if_t< std::is_arithmetic_v<T> && N!=0 > >
  static inline constexpr oarray_t<T,N> array_max( const oarray_t<T,N>& A , oarray_t<T,0> ) { return A; }

  template<class T, class U, size_t N , class = std::enable_if_t< std::is_arithmetic_v<T> && std::is_arithmetic_v<U> > >
  ONIKA_HOST_DEVICE_FUNC static inline oarray_t<decltype(T{}+U{}),N> array_add( const oarray_t<T,N>& A , const oarray_t<U,N>& B )
  {
    oarray_t<decltype(T{}+U{}),N> C{};
    for(size_t i=0;i<N;i++) C.x[i] = A.x[i] + B.x[i];
    return C;
  }

  template<class T, class U, size_t N , class = std::enable_if_t< std::is_arithmetic_v<T> && std::is_arithmetic_v<U> > >
  ONIKA_HOST_DEVICE_FUNC static inline oarray_t<decltype(T{}+U{}),N> array_add( const oarray_t<T,N>& A , const U* B )
  {
    oarray_t<decltype(T{}+U{}),N> C{};
    for(size_t i=0;i<N;i++) C.x[i] = A.x[i] + B[i];
    return C;
  }

  template<class T, class U, size_t N , class = std::enable_if_t< std::is_arithmetic_v<T> && std::is_arithmetic_v<U> > >
  static inline oarray_t<decltype(T{}-U{}),N> array_sub( const oarray_t<T,N>& A , const oarray_t<U,N>& B )
  {
    oarray_t<decltype(T{}+U{}),N> C{};
    for(size_t i=0;i<N;i++) C.x[i] = A.x[i] - B.x[i];
    return C;
  }

  template<class T, class U, size_t N , class = std::enable_if_t< std::is_arithmetic_v<T> && std::is_arithmetic_v<U> > >
  static inline oarray_t<decltype(T{}-U{}),N> array_sub( const oarray_t<T,N>& A , const U* B )
  {
    oarray_t<decltype(T{}+U{}),N> C{};
    for(size_t i=0;i<N;i++) C.x[i] = A.x[i] - B[i];
    return C;
  }


  template<class T, size_t N , class = std::enable_if_t< std::is_integral_v<T> > >
  static inline oarray_t<std::make_unsigned_t<T>,N> to_unsigned( const oarray_t<T,N>& from )
  {
    oarray_t<std::make_unsigned_t<T>,N> to{};
    convert_array( from , to );
    return to;
  }

  template<class T, size_t N , class = std::enable_if_t< std::is_integral_v<T> > >
  static inline oarray_t<std::make_signed_t<T>,N> to_signed( const oarray_t<T,N>& from )
  {
    oarray_t<std::make_signed_t<T>,N> to{};
    convert_array( from , to );
    return to;
  }

  template<class T, class U, size_t N , class = std::enable_if_t< std::is_arithmetic_v<T> && std::is_arithmetic_v<U> > >
  static inline bool in_range( const oarray_t<T,N>& p , const oarray_t<U,N>& dims )
  {
    for(size_t k=0;k<N;k++) if( p[k]<0 || p[k] >= static_cast<T>(dims[k]) ) return false;
    return true;
  }

  template<class T, size_t N , class = std::enable_if_t< std::is_arithmetic_v<T> > >
  static inline constexpr oarray_t<T,N> single_value_array( T x )
  {
    oarray_t<T,N> a{};
    for(size_t i=0;i<N;i++) a.x[i] = x;
    return a;
  }

  template<class T, size_t N>
  static inline constexpr size_t array_xor_hash ( const onika::oarray_t<T,N>& a )
  {
    size_t x = 0; // H(N); add this if you want to differentiate coordinates with different dimensionality
    if constexpr (N>0) { std::hash<T> H{}; for(size_t i=0;i<N;i++) x = x ^ H(a[i]); }
    return x;
  }

}

namespace std
{
  template<size_t N> struct hash< onika::oarray_t< int32_t,N> > { inline size_t operator () ( const onika::oarray_t< int32_t,N>& a ) const { return array_xor_hash(a); } };
  template<size_t N> struct hash< onika::oarray_t<uint32_t,N> > { inline size_t operator () ( const onika::oarray_t<uint32_t,N>& a ) const { return array_xor_hash(a); } };
  template<size_t N> struct hash< onika::oarray_t< int64_t,N> > { inline size_t operator () ( const onika::oarray_t< int64_t,N>& a ) const { return array_xor_hash(a); } };
  template<size_t N> struct hash< onika::oarray_t<uint64_t,N> > { inline size_t operator () ( const onika::oarray_t<uint64_t,N>& a ) const { return array_xor_hash(a); } };

  template<class T, size_t N > struct hash< onika::oarray_t<T,N> >
  {
    inline size_t operator () ( const onika::oarray_t<T,N>& v ) const { return std::hash<std::string_view>{} ( std::string_view(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T)) ); }
  };

}

