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

