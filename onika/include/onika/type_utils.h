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

#include <string>
#include <type_traits>
#include <tuple>
#include <utility>
#include <cassert>
#include <typeinfo>

namespace onika
{
  // just hold type information as a parameter
  template<typename T> struct TypePlaceHolder {};

  std::string demangle_type_string(std::string s);

  template<typename T>
  static inline std::string type_as_string()
  {
      return demangle_type_string( typeid(T).name() );
  }

  std::string remove_known_namespaces(std::string s);
  std::string strip_type_spaces(std::string s);
  std::string remove_shared_ptr(std::string s);
  std::string simplify_std_vector(std::string s);
  std::string pretty_short_type(std::string s);

  template<typename T>
  static inline std::string pretty_short_type() { return pretty_short_type( typeid(T).name() ); }


  // tells if T is a complete type, e.g. has a definition and not only a declaration.
  // IsComplete solution from https://stackoverflow.com/questions/1625105/how-to-write-is-complete-template
  // Thanks to Bat-Ulzii Luvsanbat: https://blogs.msdn.microsoft.com/vcblog/2015/12/02/partial-support-for-expression-sfinae-in-vs-2015-update-1/
  template <class T, class = void>
  struct IsComplete : std::false_type
  {};

  template <class T>
  struct IsComplete< T, decltype(void(sizeof(T))) > : std::true_type
  {};
  
  template <class T> static inline constexpr bool is_complete_v = IsComplete<T>::value ;

  namespace __IsCompleteTest__
  {
    template <class T> struct A;
    template <> struct A<int> {};
    static_assert( is_complete_v< A<int> > , "is_complete_v template failed (positive case)" );
    static_assert( ! is_complete_v< A<double> > , "is_complete_v template failed (negative case)" );
  }

  template<typename T, bool = std::is_default_constructible<T>::value > struct DefaultNewOrNull { static inline T* alloc() { return new T(); } };
  template<typename T> struct DefaultNewOrNull<T,false> { static inline T* alloc() { return nullptr; } };
  
  template<typename T, std::enable_if_t< std::is_convertible<T,bool>::value , int > = 0 >
  static inline bool convert_to_bool( const T& x, bool) { return x; }

  template<typename T, std::enable_if_t< !std::is_convertible<T,bool>::value , int > = 0 >
  static inline bool convert_to_bool( const T&, bool value_if_not_convertible) { return value_if_not_convertible; }
  
/*
  struct TypeDescriptor
  {
    std::type_index m_info;
  };

  template<typename T>
  static inline TypeDescriptor
*/

  // apply a function on all tuple members
  template<class TupleT, class = std::make_index_sequence<std::tuple_size<TupleT>::value> >
  struct TupleApplyHelper {};
  template<class TupleT, size_t... S> struct TupleApplyHelper< TupleT, std::integer_sequence<size_t,S...> >
  {
    template<class FuncT> static inline void apply( const TupleT& t, FuncT f ) { ( ... , f(std::get<S>(t)) ); }
  };
  template<class FuncT, class... T>
  static inline void tuple_apply( const std::tuple<T...>& t , FuncT f )
  {
     TupleApplyHelper< std::tuple<T...> >::apply(t,f);
  }


  // utilities for method returning either references or non references
  template<class R, class... A> static inline constexpr bool function_return_ref( R(A...) ){return false;}
  template<class R, class... A> static inline constexpr bool function_return_ref( R&(A...) ){return true;}


  template<class T> struct remove_pointer { typedef T type; };
  template<class T> struct remove_pointer<T*> { typedef T type; };
  template<class T> struct remove_pointer<T* const> { typedef T type; };
  template<class T> struct remove_pointer<T* volatile> { typedef T type; };
  template<class T> struct remove_pointer<T* const volatile> { typedef T type; };
  template<class T> struct remove_pointer<T* __restrict__ > { typedef T type; };
  template<class T> struct remove_pointer<T* __restrict__ const> { typedef T type; };
  template<class T> struct remove_pointer<T* __restrict__ volatile> { typedef T type; };
  template<class T> struct remove_pointer<T* __restrict__ const volatile> { typedef T type; };
  template<class T> using remove_pointer_t = typename ::onika::remove_pointer<T>::type ;
}

