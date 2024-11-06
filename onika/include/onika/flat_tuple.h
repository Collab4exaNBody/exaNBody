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
  template <size_t I> static inline constexpr tuple_index_t<I> tuple_index;

#   define ADD_TUPLE_ELEMENT(i) \
      using type_##i = T##i; \
      T##i e##i; \
      ONIKA_HOST_DEVICE_FUNC inline T##i & get( tuple_index_t<i> ) { return e##i ; } \
      ONIKA_HOST_DEVICE_FUNC inline const T##i & get( tuple_index_t<i> ) const { return e##i ; } \
      ONIKA_HOST_DEVICE_FUNC inline T##i get_copy( tuple_index_t<i> ) const { return e##i ; }

#   define FLAT_TUPLE_COMMON \
      template<size_t i> ONIKA_HOST_DEVICE_FUNC inline auto& get_nth() { return get( tuple_index<i> ); } \
      template<size_t i> ONIKA_HOST_DEVICE_FUNC inline const auto& get_nth_const() const { return get( tuple_index<i> ); }

/****** generation code ******
python3 <<- EOF
def print_enum(n,pfx,sfx,sep=""):
  for i in range(n):
    if i != 0: c=sep
    else: c=""
    print("%s%s%d%s"%(c,pfx,i,sfx),end='')
print("template<class... T> struct FlatTuple;")
print("template<> struct FlatTuple<> { static inline constexpr size_t size(){return 0;} };")
print("template<class FTuple,size_t I> struct FlatTupleElement;")
for i in range(1,30):
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

  template<class... T> struct FlatTuple;
  template<> struct FlatTuple<> { static inline constexpr size_t size(){return 0;} };
  template<class FTuple,size_t I> struct FlatTupleElement;
  template<class T0> struct FlatTuple<T0> {
  static inline constexpr size_t size(){return 1;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,0> { using type = typename FlatTuple<T...>::type_0; };
  template<class T0,class T1> struct FlatTuple<T0,T1> {
  static inline constexpr size_t size(){return 2;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,1> { using type = typename FlatTuple<T...>::type_1; };
  template<class T0,class T1,class T2> struct FlatTuple<T0,T1,T2> {
  static inline constexpr size_t size(){return 3;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,2> { using type = typename FlatTuple<T...>::type_2; };
  template<class T0,class T1,class T2,class T3> struct FlatTuple<T0,T1,T2,T3> {
  static inline constexpr size_t size(){return 4;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  ADD_TUPLE_ELEMENT(3)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,3> { using type = typename FlatTuple<T...>::type_3; };
  template<class T0,class T1,class T2,class T3,class T4> struct FlatTuple<T0,T1,T2,T3,T4> {
  static inline constexpr size_t size(){return 5;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  ADD_TUPLE_ELEMENT(3)
  ADD_TUPLE_ELEMENT(4)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,4> { using type = typename FlatTuple<T...>::type_4; };
  template<class T0,class T1,class T2,class T3,class T4,class T5> struct FlatTuple<T0,T1,T2,T3,T4,T5> {
  static inline constexpr size_t size(){return 6;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  ADD_TUPLE_ELEMENT(3)
  ADD_TUPLE_ELEMENT(4)
  ADD_TUPLE_ELEMENT(5)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,5> { using type = typename FlatTuple<T...>::type_5; };
  template<class T0,class T1,class T2,class T3,class T4,class T5,class T6> struct FlatTuple<T0,T1,T2,T3,T4,T5,T6> {
  static inline constexpr size_t size(){return 7;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  ADD_TUPLE_ELEMENT(3)
  ADD_TUPLE_ELEMENT(4)
  ADD_TUPLE_ELEMENT(5)
  ADD_TUPLE_ELEMENT(6)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,6> { using type = typename FlatTuple<T...>::type_6; };
  template<class T0,class T1,class T2,class T3,class T4,class T5,class T6,class T7> struct FlatTuple<T0,T1,T2,T3,T4,T5,T6,T7> {
  static inline constexpr size_t size(){return 8;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  ADD_TUPLE_ELEMENT(3)
  ADD_TUPLE_ELEMENT(4)
  ADD_TUPLE_ELEMENT(5)
  ADD_TUPLE_ELEMENT(6)
  ADD_TUPLE_ELEMENT(7)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,7> { using type = typename FlatTuple<T...>::type_7; };
  template<class T0,class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8> struct FlatTuple<T0,T1,T2,T3,T4,T5,T6,T7,T8> {
  static inline constexpr size_t size(){return 9;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  ADD_TUPLE_ELEMENT(3)
  ADD_TUPLE_ELEMENT(4)
  ADD_TUPLE_ELEMENT(5)
  ADD_TUPLE_ELEMENT(6)
  ADD_TUPLE_ELEMENT(7)
  ADD_TUPLE_ELEMENT(8)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,8> { using type = typename FlatTuple<T...>::type_8; };
  template<class T0,class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9> struct FlatTuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9> {
  static inline constexpr size_t size(){return 10;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  ADD_TUPLE_ELEMENT(3)
  ADD_TUPLE_ELEMENT(4)
  ADD_TUPLE_ELEMENT(5)
  ADD_TUPLE_ELEMENT(6)
  ADD_TUPLE_ELEMENT(7)
  ADD_TUPLE_ELEMENT(8)
  ADD_TUPLE_ELEMENT(9)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,9> { using type = typename FlatTuple<T...>::type_9; };
  template<class T0,class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9,class T10> struct FlatTuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10> {
  static inline constexpr size_t size(){return 11;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  ADD_TUPLE_ELEMENT(3)
  ADD_TUPLE_ELEMENT(4)
  ADD_TUPLE_ELEMENT(5)
  ADD_TUPLE_ELEMENT(6)
  ADD_TUPLE_ELEMENT(7)
  ADD_TUPLE_ELEMENT(8)
  ADD_TUPLE_ELEMENT(9)
  ADD_TUPLE_ELEMENT(10)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,10> { using type = typename FlatTuple<T...>::type_10; };
  template<class T0,class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9,class T10,class T11> struct FlatTuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11> {
  static inline constexpr size_t size(){return 12;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  ADD_TUPLE_ELEMENT(3)
  ADD_TUPLE_ELEMENT(4)
  ADD_TUPLE_ELEMENT(5)
  ADD_TUPLE_ELEMENT(6)
  ADD_TUPLE_ELEMENT(7)
  ADD_TUPLE_ELEMENT(8)
  ADD_TUPLE_ELEMENT(9)
  ADD_TUPLE_ELEMENT(10)
  ADD_TUPLE_ELEMENT(11)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,11> { using type = typename FlatTuple<T...>::type_11; };
  template<class T0,class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9,class T10,class T11,class T12> struct FlatTuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12> {
  static inline constexpr size_t size(){return 13;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  ADD_TUPLE_ELEMENT(3)
  ADD_TUPLE_ELEMENT(4)
  ADD_TUPLE_ELEMENT(5)
  ADD_TUPLE_ELEMENT(6)
  ADD_TUPLE_ELEMENT(7)
  ADD_TUPLE_ELEMENT(8)
  ADD_TUPLE_ELEMENT(9)
  ADD_TUPLE_ELEMENT(10)
  ADD_TUPLE_ELEMENT(11)
  ADD_TUPLE_ELEMENT(12)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,12> { using type = typename FlatTuple<T...>::type_12; };
  template<class T0,class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9,class T10,class T11,class T12,class T13> struct FlatTuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13> {
  static inline constexpr size_t size(){return 14;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  ADD_TUPLE_ELEMENT(3)
  ADD_TUPLE_ELEMENT(4)
  ADD_TUPLE_ELEMENT(5)
  ADD_TUPLE_ELEMENT(6)
  ADD_TUPLE_ELEMENT(7)
  ADD_TUPLE_ELEMENT(8)
  ADD_TUPLE_ELEMENT(9)
  ADD_TUPLE_ELEMENT(10)
  ADD_TUPLE_ELEMENT(11)
  ADD_TUPLE_ELEMENT(12)
  ADD_TUPLE_ELEMENT(13)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,13> { using type = typename FlatTuple<T...>::type_13; };
  template<class T0,class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9,class T10,class T11,class T12,class T13,class T14> struct FlatTuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14> {
  static inline constexpr size_t size(){return 15;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  ADD_TUPLE_ELEMENT(3)
  ADD_TUPLE_ELEMENT(4)
  ADD_TUPLE_ELEMENT(5)
  ADD_TUPLE_ELEMENT(6)
  ADD_TUPLE_ELEMENT(7)
  ADD_TUPLE_ELEMENT(8)
  ADD_TUPLE_ELEMENT(9)
  ADD_TUPLE_ELEMENT(10)
  ADD_TUPLE_ELEMENT(11)
  ADD_TUPLE_ELEMENT(12)
  ADD_TUPLE_ELEMENT(13)
  ADD_TUPLE_ELEMENT(14)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,14> { using type = typename FlatTuple<T...>::type_14; };
  template<class T0,class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9,class T10,class T11,class T12,class T13,class T14,class T15> struct FlatTuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15> {
  static inline constexpr size_t size(){return 16;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  ADD_TUPLE_ELEMENT(3)
  ADD_TUPLE_ELEMENT(4)
  ADD_TUPLE_ELEMENT(5)
  ADD_TUPLE_ELEMENT(6)
  ADD_TUPLE_ELEMENT(7)
  ADD_TUPLE_ELEMENT(8)
  ADD_TUPLE_ELEMENT(9)
  ADD_TUPLE_ELEMENT(10)
  ADD_TUPLE_ELEMENT(11)
  ADD_TUPLE_ELEMENT(12)
  ADD_TUPLE_ELEMENT(13)
  ADD_TUPLE_ELEMENT(14)
  ADD_TUPLE_ELEMENT(15)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,15> { using type = typename FlatTuple<T...>::type_15; };
  template<class T0,class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9,class T10,class T11,class T12,class T13,class T14,class T15,class T16> struct FlatTuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16> {
  static inline constexpr size_t size(){return 17;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  ADD_TUPLE_ELEMENT(3)
  ADD_TUPLE_ELEMENT(4)
  ADD_TUPLE_ELEMENT(5)
  ADD_TUPLE_ELEMENT(6)
  ADD_TUPLE_ELEMENT(7)
  ADD_TUPLE_ELEMENT(8)
  ADD_TUPLE_ELEMENT(9)
  ADD_TUPLE_ELEMENT(10)
  ADD_TUPLE_ELEMENT(11)
  ADD_TUPLE_ELEMENT(12)
  ADD_TUPLE_ELEMENT(13)
  ADD_TUPLE_ELEMENT(14)
  ADD_TUPLE_ELEMENT(15)
  ADD_TUPLE_ELEMENT(16)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,16> { using type = typename FlatTuple<T...>::type_16; };
  template<class T0,class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9,class T10,class T11,class T12,class T13,class T14,class T15,class T16,class T17> struct FlatTuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17> {
  static inline constexpr size_t size(){return 18;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  ADD_TUPLE_ELEMENT(3)
  ADD_TUPLE_ELEMENT(4)
  ADD_TUPLE_ELEMENT(5)
  ADD_TUPLE_ELEMENT(6)
  ADD_TUPLE_ELEMENT(7)
  ADD_TUPLE_ELEMENT(8)
  ADD_TUPLE_ELEMENT(9)
  ADD_TUPLE_ELEMENT(10)
  ADD_TUPLE_ELEMENT(11)
  ADD_TUPLE_ELEMENT(12)
  ADD_TUPLE_ELEMENT(13)
  ADD_TUPLE_ELEMENT(14)
  ADD_TUPLE_ELEMENT(15)
  ADD_TUPLE_ELEMENT(16)
  ADD_TUPLE_ELEMENT(17)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,17> { using type = typename FlatTuple<T...>::type_17; };
  template<class T0,class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9,class T10,class T11,class T12,class T13,class T14,class T15,class T16,class T17,class T18> struct FlatTuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18> {
  static inline constexpr size_t size(){return 19;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  ADD_TUPLE_ELEMENT(3)
  ADD_TUPLE_ELEMENT(4)
  ADD_TUPLE_ELEMENT(5)
  ADD_TUPLE_ELEMENT(6)
  ADD_TUPLE_ELEMENT(7)
  ADD_TUPLE_ELEMENT(8)
  ADD_TUPLE_ELEMENT(9)
  ADD_TUPLE_ELEMENT(10)
  ADD_TUPLE_ELEMENT(11)
  ADD_TUPLE_ELEMENT(12)
  ADD_TUPLE_ELEMENT(13)
  ADD_TUPLE_ELEMENT(14)
  ADD_TUPLE_ELEMENT(15)
  ADD_TUPLE_ELEMENT(16)
  ADD_TUPLE_ELEMENT(17)
  ADD_TUPLE_ELEMENT(18)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,18> { using type = typename FlatTuple<T...>::type_18; };
  template<class T0,class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9,class T10,class T11,class T12,class T13,class T14,class T15,class T16,class T17,class T18,class T19> struct FlatTuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19> {
  static inline constexpr size_t size(){return 20;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  ADD_TUPLE_ELEMENT(3)
  ADD_TUPLE_ELEMENT(4)
  ADD_TUPLE_ELEMENT(5)
  ADD_TUPLE_ELEMENT(6)
  ADD_TUPLE_ELEMENT(7)
  ADD_TUPLE_ELEMENT(8)
  ADD_TUPLE_ELEMENT(9)
  ADD_TUPLE_ELEMENT(10)
  ADD_TUPLE_ELEMENT(11)
  ADD_TUPLE_ELEMENT(12)
  ADD_TUPLE_ELEMENT(13)
  ADD_TUPLE_ELEMENT(14)
  ADD_TUPLE_ELEMENT(15)
  ADD_TUPLE_ELEMENT(16)
  ADD_TUPLE_ELEMENT(17)
  ADD_TUPLE_ELEMENT(18)
  ADD_TUPLE_ELEMENT(19)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,19> { using type = typename FlatTuple<T...>::type_19; };
  template<class T0,class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9,class T10,class T11,class T12,class T13,class T14,class T15,class T16,class T17,class T18,class T19,class T20> struct FlatTuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20> {
  static inline constexpr size_t size(){return 21;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  ADD_TUPLE_ELEMENT(3)
  ADD_TUPLE_ELEMENT(4)
  ADD_TUPLE_ELEMENT(5)
  ADD_TUPLE_ELEMENT(6)
  ADD_TUPLE_ELEMENT(7)
  ADD_TUPLE_ELEMENT(8)
  ADD_TUPLE_ELEMENT(9)
  ADD_TUPLE_ELEMENT(10)
  ADD_TUPLE_ELEMENT(11)
  ADD_TUPLE_ELEMENT(12)
  ADD_TUPLE_ELEMENT(13)
  ADD_TUPLE_ELEMENT(14)
  ADD_TUPLE_ELEMENT(15)
  ADD_TUPLE_ELEMENT(16)
  ADD_TUPLE_ELEMENT(17)
  ADD_TUPLE_ELEMENT(18)
  ADD_TUPLE_ELEMENT(19)
  ADD_TUPLE_ELEMENT(20)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,20> { using type = typename FlatTuple<T...>::type_20; };
  template<class T0,class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9,class T10,class T11,class T12,class T13,class T14,class T15,class T16,class T17,class T18,class T19,class T20,class T21> struct FlatTuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21> {
  static inline constexpr size_t size(){return 22;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  ADD_TUPLE_ELEMENT(3)
  ADD_TUPLE_ELEMENT(4)
  ADD_TUPLE_ELEMENT(5)
  ADD_TUPLE_ELEMENT(6)
  ADD_TUPLE_ELEMENT(7)
  ADD_TUPLE_ELEMENT(8)
  ADD_TUPLE_ELEMENT(9)
  ADD_TUPLE_ELEMENT(10)
  ADD_TUPLE_ELEMENT(11)
  ADD_TUPLE_ELEMENT(12)
  ADD_TUPLE_ELEMENT(13)
  ADD_TUPLE_ELEMENT(14)
  ADD_TUPLE_ELEMENT(15)
  ADD_TUPLE_ELEMENT(16)
  ADD_TUPLE_ELEMENT(17)
  ADD_TUPLE_ELEMENT(18)
  ADD_TUPLE_ELEMENT(19)
  ADD_TUPLE_ELEMENT(20)
  ADD_TUPLE_ELEMENT(21)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,21> { using type = typename FlatTuple<T...>::type_21; };
  template<class T0,class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9,class T10,class T11,class T12,class T13,class T14,class T15,class T16,class T17,class T18,class T19,class T20,class T21,class T22> struct FlatTuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22> {
  static inline constexpr size_t size(){return 23;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  ADD_TUPLE_ELEMENT(3)
  ADD_TUPLE_ELEMENT(4)
  ADD_TUPLE_ELEMENT(5)
  ADD_TUPLE_ELEMENT(6)
  ADD_TUPLE_ELEMENT(7)
  ADD_TUPLE_ELEMENT(8)
  ADD_TUPLE_ELEMENT(9)
  ADD_TUPLE_ELEMENT(10)
  ADD_TUPLE_ELEMENT(11)
  ADD_TUPLE_ELEMENT(12)
  ADD_TUPLE_ELEMENT(13)
  ADD_TUPLE_ELEMENT(14)
  ADD_TUPLE_ELEMENT(15)
  ADD_TUPLE_ELEMENT(16)
  ADD_TUPLE_ELEMENT(17)
  ADD_TUPLE_ELEMENT(18)
  ADD_TUPLE_ELEMENT(19)
  ADD_TUPLE_ELEMENT(20)
  ADD_TUPLE_ELEMENT(21)
  ADD_TUPLE_ELEMENT(22)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,22> { using type = typename FlatTuple<T...>::type_22; };
  template<class T0,class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9,class T10,class T11,class T12,class T13,class T14,class T15,class T16,class T17,class T18,class T19,class T20,class T21,class T22,class T23> struct FlatTuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23> {
  static inline constexpr size_t size(){return 24;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  ADD_TUPLE_ELEMENT(3)
  ADD_TUPLE_ELEMENT(4)
  ADD_TUPLE_ELEMENT(5)
  ADD_TUPLE_ELEMENT(6)
  ADD_TUPLE_ELEMENT(7)
  ADD_TUPLE_ELEMENT(8)
  ADD_TUPLE_ELEMENT(9)
  ADD_TUPLE_ELEMENT(10)
  ADD_TUPLE_ELEMENT(11)
  ADD_TUPLE_ELEMENT(12)
  ADD_TUPLE_ELEMENT(13)
  ADD_TUPLE_ELEMENT(14)
  ADD_TUPLE_ELEMENT(15)
  ADD_TUPLE_ELEMENT(16)
  ADD_TUPLE_ELEMENT(17)
  ADD_TUPLE_ELEMENT(18)
  ADD_TUPLE_ELEMENT(19)
  ADD_TUPLE_ELEMENT(20)
  ADD_TUPLE_ELEMENT(21)
  ADD_TUPLE_ELEMENT(22)
  ADD_TUPLE_ELEMENT(23)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,23> { using type = typename FlatTuple<T...>::type_23; };
  template<class T0,class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9,class T10,class T11,class T12,class T13,class T14,class T15,class T16,class T17,class T18,class T19,class T20,class T21,class T22,class T23,class T24> struct FlatTuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24> {
  static inline constexpr size_t size(){return 25;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  ADD_TUPLE_ELEMENT(3)
  ADD_TUPLE_ELEMENT(4)
  ADD_TUPLE_ELEMENT(5)
  ADD_TUPLE_ELEMENT(6)
  ADD_TUPLE_ELEMENT(7)
  ADD_TUPLE_ELEMENT(8)
  ADD_TUPLE_ELEMENT(9)
  ADD_TUPLE_ELEMENT(10)
  ADD_TUPLE_ELEMENT(11)
  ADD_TUPLE_ELEMENT(12)
  ADD_TUPLE_ELEMENT(13)
  ADD_TUPLE_ELEMENT(14)
  ADD_TUPLE_ELEMENT(15)
  ADD_TUPLE_ELEMENT(16)
  ADD_TUPLE_ELEMENT(17)
  ADD_TUPLE_ELEMENT(18)
  ADD_TUPLE_ELEMENT(19)
  ADD_TUPLE_ELEMENT(20)
  ADD_TUPLE_ELEMENT(21)
  ADD_TUPLE_ELEMENT(22)
  ADD_TUPLE_ELEMENT(23)
  ADD_TUPLE_ELEMENT(24)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,24> { using type = typename FlatTuple<T...>::type_24; };
  template<class T0,class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9,class T10,class T11,class T12,class T13,class T14,class T15,class T16,class T17,class T18,class T19,class T20,class T21,class T22,class T23,class T24,class T25> struct FlatTuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25> {
  static inline constexpr size_t size(){return 26;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  ADD_TUPLE_ELEMENT(3)
  ADD_TUPLE_ELEMENT(4)
  ADD_TUPLE_ELEMENT(5)
  ADD_TUPLE_ELEMENT(6)
  ADD_TUPLE_ELEMENT(7)
  ADD_TUPLE_ELEMENT(8)
  ADD_TUPLE_ELEMENT(9)
  ADD_TUPLE_ELEMENT(10)
  ADD_TUPLE_ELEMENT(11)
  ADD_TUPLE_ELEMENT(12)
  ADD_TUPLE_ELEMENT(13)
  ADD_TUPLE_ELEMENT(14)
  ADD_TUPLE_ELEMENT(15)
  ADD_TUPLE_ELEMENT(16)
  ADD_TUPLE_ELEMENT(17)
  ADD_TUPLE_ELEMENT(18)
  ADD_TUPLE_ELEMENT(19)
  ADD_TUPLE_ELEMENT(20)
  ADD_TUPLE_ELEMENT(21)
  ADD_TUPLE_ELEMENT(22)
  ADD_TUPLE_ELEMENT(23)
  ADD_TUPLE_ELEMENT(24)
  ADD_TUPLE_ELEMENT(25)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,25> { using type = typename FlatTuple<T...>::type_25; };
  template<class T0,class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9,class T10,class T11,class T12,class T13,class T14,class T15,class T16,class T17,class T18,class T19,class T20,class T21,class T22,class T23,class T24,class T25,class T26> struct FlatTuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26> {
  static inline constexpr size_t size(){return 27;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  ADD_TUPLE_ELEMENT(3)
  ADD_TUPLE_ELEMENT(4)
  ADD_TUPLE_ELEMENT(5)
  ADD_TUPLE_ELEMENT(6)
  ADD_TUPLE_ELEMENT(7)
  ADD_TUPLE_ELEMENT(8)
  ADD_TUPLE_ELEMENT(9)
  ADD_TUPLE_ELEMENT(10)
  ADD_TUPLE_ELEMENT(11)
  ADD_TUPLE_ELEMENT(12)
  ADD_TUPLE_ELEMENT(13)
  ADD_TUPLE_ELEMENT(14)
  ADD_TUPLE_ELEMENT(15)
  ADD_TUPLE_ELEMENT(16)
  ADD_TUPLE_ELEMENT(17)
  ADD_TUPLE_ELEMENT(18)
  ADD_TUPLE_ELEMENT(19)
  ADD_TUPLE_ELEMENT(20)
  ADD_TUPLE_ELEMENT(21)
  ADD_TUPLE_ELEMENT(22)
  ADD_TUPLE_ELEMENT(23)
  ADD_TUPLE_ELEMENT(24)
  ADD_TUPLE_ELEMENT(25)
  ADD_TUPLE_ELEMENT(26)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,26> { using type = typename FlatTuple<T...>::type_26; };
  template<class T0,class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9,class T10,class T11,class T12,class T13,class T14,class T15,class T16,class T17,class T18,class T19,class T20,class T21,class T22,class T23,class T24,class T25,class T26,class T27> struct FlatTuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27> {
  static inline constexpr size_t size(){return 28;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  ADD_TUPLE_ELEMENT(3)
  ADD_TUPLE_ELEMENT(4)
  ADD_TUPLE_ELEMENT(5)
  ADD_TUPLE_ELEMENT(6)
  ADD_TUPLE_ELEMENT(7)
  ADD_TUPLE_ELEMENT(8)
  ADD_TUPLE_ELEMENT(9)
  ADD_TUPLE_ELEMENT(10)
  ADD_TUPLE_ELEMENT(11)
  ADD_TUPLE_ELEMENT(12)
  ADD_TUPLE_ELEMENT(13)
  ADD_TUPLE_ELEMENT(14)
  ADD_TUPLE_ELEMENT(15)
  ADD_TUPLE_ELEMENT(16)
  ADD_TUPLE_ELEMENT(17)
  ADD_TUPLE_ELEMENT(18)
  ADD_TUPLE_ELEMENT(19)
  ADD_TUPLE_ELEMENT(20)
  ADD_TUPLE_ELEMENT(21)
  ADD_TUPLE_ELEMENT(22)
  ADD_TUPLE_ELEMENT(23)
  ADD_TUPLE_ELEMENT(24)
  ADD_TUPLE_ELEMENT(25)
  ADD_TUPLE_ELEMENT(26)
  ADD_TUPLE_ELEMENT(27)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,27> { using type = typename FlatTuple<T...>::type_27; };
  template<class T0,class T1,class T2,class T3,class T4,class T5,class T6,class T7,class T8,class T9,class T10,class T11,class T12,class T13,class T14,class T15,class T16,class T17,class T18,class T19,class T20,class T21,class T22,class T23,class T24,class T25,class T26,class T27,class T28> struct FlatTuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28> {
  static inline constexpr size_t size(){return 29;}
  FLAT_TUPLE_COMMON
  ADD_TUPLE_ELEMENT(0)
  ADD_TUPLE_ELEMENT(1)
  ADD_TUPLE_ELEMENT(2)
  ADD_TUPLE_ELEMENT(3)
  ADD_TUPLE_ELEMENT(4)
  ADD_TUPLE_ELEMENT(5)
  ADD_TUPLE_ELEMENT(6)
  ADD_TUPLE_ELEMENT(7)
  ADD_TUPLE_ELEMENT(8)
  ADD_TUPLE_ELEMENT(9)
  ADD_TUPLE_ELEMENT(10)
  ADD_TUPLE_ELEMENT(11)
  ADD_TUPLE_ELEMENT(12)
  ADD_TUPLE_ELEMENT(13)
  ADD_TUPLE_ELEMENT(14)
  ADD_TUPLE_ELEMENT(15)
  ADD_TUPLE_ELEMENT(16)
  ADD_TUPLE_ELEMENT(17)
  ADD_TUPLE_ELEMENT(18)
  ADD_TUPLE_ELEMENT(19)
  ADD_TUPLE_ELEMENT(20)
  ADD_TUPLE_ELEMENT(21)
  ADD_TUPLE_ELEMENT(22)
  ADD_TUPLE_ELEMENT(23)
  ADD_TUPLE_ELEMENT(24)
  ADD_TUPLE_ELEMENT(25)
  ADD_TUPLE_ELEMENT(26)
  ADD_TUPLE_ELEMENT(27)
  ADD_TUPLE_ELEMENT(28)
  };
  template<class... T> struct FlatTupleElement<FlatTuple<T...>,28> { using type = typename FlatTuple<T...>::type_28; };

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



