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

#include <onika/macro_utils.h>

namespace onika
{

  namespace soatl
  {

    template<class FuncT , class... fids> struct FieldCombiner;
/*
    {
      FuncT m_func;
      using value_type = decltype( m_func( typename FieldId<fids>::value_type {} ... ) );
      static const char* short_name() { return "combiner"; }
      static const char* name() { return "combiner"; }
    };
*/

  } // namespace soatl
  
}

#define _ONIKA_GET_TYPE_FROM_FIELD_ID(x) typename FieldId<x>::value_type {}

#define ONIKA_DECLARE_FIELD_COMBINER(ns,CombT,combiner,FuncT,...) \
namespace onika { \
namespace soatl { \
template<> struct FieldCombiner<FuncT OPT_COMMA_VA_ARGS(__VA_ARGS__)> { \
  FuncT m_func; \
  using value_type = decltype( m_func( EXPAND_WITH_FUNC(_ONIKA_GET_TYPE_FROM_FIELD_ID OPT_COMMA_VA_ARGS(__VA_ARGS__)) ) ); \
  static const char* short_name() { return #combiner; } \
  static const char* name() { return #combiner; } \
  }; \
} } \
namespace ns { using CombT = onika::soatl::FieldCombiner<FuncT OPT_COMMA_VA_ARGS(__VA_ARGS__)>; }

