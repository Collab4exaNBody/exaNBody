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

#include <cstdint>
#include <type_traits>

#include <onika/cuda/cuda.h>

namespace onika
{
  // a short replacement of std::integral_constant for tuple indexing
  template <size_t I> struct tuple_index_t {};
  template <size_t I> static inline constexpr tuple_index_t<I> tuple_index = {};

#   define ADD_TUPLE_ELEMENT(i) \
      using type_##i = T##i; \
      T##i e##i; \
      ONIKA_HOST_DEVICE_FUNC inline T##i & get( tuple_index_t<i> ) { return e##i ; } \
      ONIKA_HOST_DEVICE_FUNC inline const T##i & get( tuple_index_t<i> ) const { return e##i ; } \
      ONIKA_HOST_DEVICE_FUNC inline T##i get_copy( tuple_index_t<i> ) const { return e##i ; }

#   define FLAT_TUPLE_COMMON \
      template<size_t i> ONIKA_HOST_DEVICE_FUNC inline auto& get_nth() { return get( tuple_index<i> ); } \
      template<size_t i> ONIKA_HOST_DEVICE_FUNC inline const auto& get_nth_const() const { return get( tuple_index<i> ); }

/****** generation code for flat_tuple_gen.h ******
python3 > flat_tuple_gen.h <<- EOF
def print_enum(n,pfx,sfx,sep=""):
  for i in range(n):
    if i != 0: c=sep
    else: c=""
    print("%s%s%d%s"%(c,pfx,i,sfx),end='')
print("template<class... T> struct FlatTuple;")
print("template<> struct FlatTuple<> { static inline constexpr size_t size(){return 0;} };")
print("template<class FTuple,size_t I> struct FlatTupleElement;")
for i in range(1,40):
  print("template<",end='')
  print_enum(i,"class T","",",")
  print("> struct FlatTuple<",end='')
  print_enum(i,"T","",",")
  print("> {")
  print("static inline constexpr size_t size(){return %d;}"%i)
  print("FLAT_TUPLE_COMMON")
  print_enum(i,"ADD_TUPLE_ELEMENT(",")\n")
  print("};")
  print("template<class... T> struct FlatTupleElement<FlatTuple<T...>,%d> { using type = typename FlatTuple<T...>::type_%d; };"%(i-1,i-1))
EOF
*****************************/
#   include <onika/flat_tuple_gen.h>

    /****************************
    *** end of generated code ***
    *****************************/

#   undef ADD_TUPLE_ELEMENT
#   undef FLAT_TUPLE_COMMON

    template<class FTuple,size_t I> using flat_tuple_element_t = typename FlatTupleElement<FTuple,I>::type;

    namespace flat_tuple_details
    {
      template <class T>
      struct unwrap_refwrapper
      {
          using type = T;
      };
       
      template <class T>
      struct unwrap_refwrapper<std::reference_wrapper<T>>
      {
          using type = T&;
      };
       
      template <class T>
      using unwrap_decay_t = typename unwrap_refwrapper<typename std::decay<T>::type>::type;
      // or use std::unwrap_ref_decay_t (since C++20)
    }
         
    template <class... Types>
    ONIKA_HOST_DEVICE_FUNC static inline constexpr
    FlatTuple<flat_tuple_details::unwrap_decay_t<Types>...> make_flat_tuple(Types&&... args)
    {
      using namespace flat_tuple_details;
      return FlatTuple<unwrap_decay_t<Types>...>{ std::forward<Types>(args)... };
    }

    template<class _Tuple,size_t... Is>
    static inline FlatTuple< flat_tuple_element_t<_Tuple,Is> ... > flat_tuple_subset( const _Tuple& v , std::integer_sequence<std::size_t,Is...> )
    {
      return { v.get(tuple_index<Is>) ... };
    }

    template<class FlatTupleT> struct TupleSizeConst { static inline constexpr size_t value = FlatTupleT::size(); };
    template<class FlatTupleT> static inline constexpr size_t tuple_size_const_v = TupleSizeConst<FlatTupleT>::value;
}



