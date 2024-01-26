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

#include <exanb/core/yaml_utils.h>
#include <onika/macro_utils.h>
#include <map>
#include <vector>
#include <string>
#include <iostream>

#define EXANB_YAML_ENUM_strmatch(x) if( key == #x ) return MyEnumType{ MyEnumType::x }; else
#define EXANB_YAML_ENUM_idmatch(x) if( value.m_value == MyEnumType::x ) return #x; else
#define EXANB_YAML_ENUM(ns,name,...) \
namespace ns { \
  struct name \
  { \
    enum name##Enum : int { UNDEFINED=-1 OPT_COMMA_VA_ARGS(__VA_ARGS__) , COUNT }; \
    name##Enum m_value = UNDEFINED; \
    name()=default; \
    name(const name&)=default; \
    name(name&&)=default; \
    name(const name##Enum& v):m_value(v){} \
    name& operator = (const name&)=default; \
    name& operator = (name&&)=default; \
    name& operator = (const name##Enum& v) { m_value=v; return *this; } \
    inline int value() const { return m_value; } \
    inline bool operator == ( const name##Enum & x ) const { return m_value == x; } \
    inline bool operator == ( const name & x ) const { return m_value == x.m_value; } \
    inline bool operator != ( const name##Enum & x ) const { return m_value != x; } \
    inline bool operator != ( const name & x ) const { return m_value != x.m_value; } \
  }; \
  inline name name##_from_str(const std::string& key) \
  { \
    using MyEnumType = name ; \
    EXPAND_WITH_FUNC_NOSEP(EXANB_YAML_ENUM_strmatch OPT_COMMA_VA_ARGS(__VA_ARGS__) ) \
    return MyEnumType{ name::UNDEFINED }; \
  } \
  inline const char* to_cstr( const name & value ) \
  { \
    using MyEnumType = name ; \
    EXPAND_WITH_FUNC_NOSEP(EXANB_YAML_ENUM_idmatch OPT_COMMA_VA_ARGS(__VA_ARGS__) ) \
    return "<undefined>"; \
  } \
  inline std::ostream& operator << ( std::ostream& out , const name & value ) { out << to_cstr(value); return out; } \
} \
namespace YAML \
{ \
  template<> struct convert< ns::name > \
  { \
    static inline Node encode(const ns::name& v) \
    { \
      Node node; \
      node = to_cstr(v); \
      return node; \
    } \
    static inline bool decode(const Node& node, ns::name& v) \
    { \
      v = ns::name##_from_str( node.as<std::string>() ); \
      return true; \
    } \
  }; \
}

