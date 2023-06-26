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

