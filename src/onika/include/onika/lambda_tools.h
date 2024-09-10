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

#include <functional>
#include <cstring>
#include <cassert>
#include <cstdlib>
#include <iostream>

#include <onika/flat_tuple.h>

namespace onika
{

  struct LambdaCallInfo
  {
    const void * const call_address = nullptr;
    const void * const data_address = nullptr;
    const size_t data_size = 0;
  };

  // sample class serving as a base pattern for all lambda closure types
  struct AbstractLambda
  {
    unsigned long some_data;
    template<class... Args> inline void operator () (Args ...) const;
  };

  template<class F> struct FunctionCallArgsHash
  {
    static inline size_t value() { return typeid(F).hash_code(); }
  };
  template<class R, class... Args>
  struct FunctionCallArgsHash< std::function<R(Args...)> >
  {
    using function_type = std::function<R(Args...)>;
    static inline size_t value() { return typeid(function_type).hash_code(); }
  };

  template<class F>
  static inline auto lambda_to_function( const F& lambda )
  {
#   ifdef __CUDACC__
    return lambda;
#   else
    std::function func = lambda; // uses deduction guides
    return func;
#   endif
  }

  template<class F>
  static inline size_t lambda_call_args_hash(const F&)
  {
    return FunctionCallArgsHash< decltype( lambda_to_function( std::declval<F>() ) ) >::value();
  }



  /********** detect functor call compatibilty ************************/ 
  template <class R, class F , class ArgsTp , class = void >
  struct FunctorSupportsCallSignature : public std::false_type {};  
  // this version will always be considered more specialized than base defintion
  // so it will be used, unless function cannot be called with arguments, or return type doesn't match with requested one
  template <class R, class F , class... Args >
  struct FunctorSupportsCallSignature<R,F,onika::FlatTuple<Args...>, std::enable_if_t< std::is_same_v<decltype(std::declval<F>()(std::declval<Args>()...)),R> > > : public std::true_type {};
  template<class F, class R, class... Args> static inline constexpr bool lambda_is_compatible_with_v = FunctorSupportsCallSignature<R,F,onika::FlatTuple<Args...> >::value;

  // without return type, only checks call args
  template <class F , class ArgsTp , class = void >
  struct FunctorSupportsCallArgs : public std::false_type {};  
  // this version will always be considered more specialized than base defintion
  // so it will be used, unless function cannot be called with arguments, or return type doesn't match with requested one
  template <class F , class... Args >
  struct FunctorSupportsCallArgs<F,onika::FlatTuple<Args...>, std::enable_if_t<sizeof(decltype(std::declval<F>()(std::declval<Args>()...)))>=0> > : public std::true_type {};
  template<class F,class... Args> static inline constexpr bool lambda_is_callable_with_args_v = FunctorSupportsCallArgs<F,onika::FlatTuple<Args...> >::value;
  /********************************************************************/


/*
  template<class F, class R, class... Args>
  struct LambdaIsCompatibleWith
  {
#   ifdef __CUDACC__
    static inline constexpr bool value = std::is_same_v< decltype( std::declval<F>()( std::declval<Args>() ... ) ) , R > ;
#   else
    static inline constexpr bool value = std::is_same_v< decltype( lambda_to_function(std::declval<F>()) ) , std::function<R(Args...)> > ;
#   endif
  };
  template<class F, class R, class... Args>
  static inline constexpr bool lambda_is_compatible_with_v = functor_supports_call_signature_v<F,R,Args...>; //LambdaIsCompatibleWith<std::remove_reference_t<F>,R,Args...>::value;
*/

  template<class F>
  static inline LambdaCallInfo lambda_call_info( const F& f )
  {      
    decltype( & F::operator () ) call_addr;
    static_assert( sizeof(call_addr) >= sizeof(void*) , "strange function pointer size");
    static_assert( sizeof(char) == 1 , "sizeof(char) <> 1" );
#   ifndef NDEBUG
    std::memset( &call_addr , 0 , sizeof(call_addr) );
#   endif
    call_addr = & F::operator ();
    char* call_addr_bytes = reinterpret_cast<char*>( & call_addr );
#   ifndef NDEBUG
    for(size_t i=sizeof(void*);i<sizeof(call_addr);i++) { assert( call_addr_bytes[i]==0 ); }
#   endif
    const void* generic_call_addr = * reinterpret_cast<void const * const *>( call_addr_bytes );
    const void* obj_ptr = reinterpret_cast<const void*>( &f );
    size_t obj_size = sizeof(F);
    return { generic_call_addr , obj_ptr , obj_size };
  }
  
  template<class... Args>
  static inline void lambda_call( const void* call_address , const void* data , Args... args)
  {
    static_assert( ( ... && (!std::is_reference_v<Args>) ) , "unexpected reference type in parameter list" );
    using lambda_fn_t = void (AbstractLambda::*)(Args...) const;
    const void* fn_ptr_data[2] = { call_address , nullptr };
    lambda_fn_t f;
    static_assert( sizeof(fn_ptr_data) == sizeof(lambda_fn_t) , "lambda pointer has unexepected size" );
    std::memcpy( &f , fn_ptr_data , sizeof(f) );
    const auto& obj = * reinterpret_cast<const AbstractLambda*>( data );
    std::invoke(f,obj, args... );
  }

  template<class... Args> inline void AbstractLambda::operator () (Args ...) const
  {
    std::cerr << "Fatal error: AbstractLambda::operator () () not meant to be called" << std::endl;
    std::abort();
    std::cerr << "You shall never see this : some_data = " << some_data << std::endl;
  }
}

