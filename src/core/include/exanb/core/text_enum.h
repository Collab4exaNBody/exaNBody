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
#include <yaml-cpp/yaml.h>

# define XNB_TEXT_ENUM_START(name) \
  namespace exanb { struct name { \
    inline name() = default; \
    inline name(const name&) = default; \
    inline name(int x) : value(x) {} \
    inline name(const std::string& txt) { value=-1; for(int i=0;i<s_item_count;i++) { if( txt == s_text[i] ) { value = i; break; } } } \
    inline bool valid() const { return value>=0 && value<s_item_count; } \
    inline const char* str() const { if(valid()) return s_text[value]; else return nullptr; }\
    inline operator int() const { return value; } \
    static constexpr int MAX_ITEMS = 256; \
    static inline int s_item_count = 0; \
    static inline const char* s_text[MAX_ITEMS]; \
    static inline int register_enum(const char* str) { s_text[s_item_count] = str; return s_item_count++; } \
    int value = 0

# define XNB_TEXT_ENUM_ITEM(itname) static inline const int itname = register_enum(#itname)

# define XNB_TEXT_ENUM_END() }; } using __exanb_##name=int

#define XNB_TEXT_ENUM_YAML(name) \
namespace YAML { template<> struct convert< ::exanb::name > { \
    static inline Node encode(const ::exanb::name& v) { Node node; if(v.valid()) node = std::string( ::exanb::name::s_text[ v.value ] ); return node; } \
    static inline bool decode(const Node& node, ::exanb::name& v) { v = ::exanb::name(node.as<std::string>()); return v.valid(); } \
    }; } using __exanb_yaml_##name=int


