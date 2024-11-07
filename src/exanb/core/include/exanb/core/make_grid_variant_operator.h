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

#include <exanb/core/operator.h>
#include <exanb/core/operator_factory.h>
#include <exanb/field_sets.h>
#include <exanb/core/grid.h>
#include <yaml-cpp/yaml.h>
#include <memory>

#include <onika/type_utils.h>
#include <exanb/core/log.h>

namespace exanb
{
  
  /*
    Internal template utilities
  */
  namespace details
  {
  
    template< template<class> class _OperatorTemplate, class GridT , class = void >
    struct instantiable_grid_template : public std::false_type {};

    template< template<class> class _OperatorTemplate, class GridT >
    struct instantiable_grid_template< _OperatorTemplate , GridT , decltype(void(sizeof(_OperatorTemplate<GridT>))) > : public std::true_type {};

    template<template<class> class _OperatorTemplate, class FSS >
    struct valid_template_field_sets {};
    
    template<template<class> class _OperatorTemplate, class fs_head, class... other_fss >
    struct valid_template_field_sets< _OperatorTemplate , FieldSets<fs_head,other_fss...> >
    {
      static constexpr bool instantiable_head = instantiable_grid_template< _OperatorTemplate , GridFromFieldSet<fs_head> >::value ;
      using type = typename PrependFieldSetIf<
        typename valid_template_field_sets< _OperatorTemplate , FieldSets<other_fss...> >::type
        , fs_head , instantiable_head >::type;
    };
    
    template< template<class> class _OperatorTemplate >
    struct valid_template_field_sets< _OperatorTemplate , FieldSets<> >
    {
      using type = FieldSets<>;
    };
    
    template< template<class> class _OperatorTemplate, class FSS > using valid_template_field_sets_t = typename valid_template_field_sets<_OperatorTemplate,FSS>::type;

    template< template<typename> typename _OperatorTemplate , typename _fs>
    struct MakeGridVariantOperatorHelper {}; 
    
    template< template<typename> typename _OperatorTemplate , typename... _fs>
    struct MakeGridVariantOperatorHelper<_OperatorTemplate , FieldSets<_fs...> >
    {
      static inline std::shared_ptr<OperatorNode> make_operator( const YAML::Node& node, const OperatorNodeFlavor& flavor )
      {
        static constexpr bool all_have_memory_bytes_method = ( ... && (onika::memory::has_memory_bytes_method_v< GridFromFieldSet<_fs> > ) );
        static_assert( all_have_memory_bytes_method , "a valid memory_bytes method is needed for proper memory usage accounting" );
        return make_compatible_operator < _OperatorTemplate< GridFromFieldSet<_fs> > ... > (node,flavor);
      }
    };

    template< typename FSS > struct PrintFieldSets;
    template< typename... FS > struct PrintFieldSets< FieldSets<FS...> >
    {
      template<class StreamT>
      static inline void print_field_sets(StreamT& out , const std::string& prefix="- ", const std::string& mid="", const std::string& sufix="\n" )
      {
        std::string m="";
        ( ... , (  out << prefix << m << onika::pretty_short_type< GridFromFieldSet<FS> > () << sufix , m=mid ) );
      }
    };

  }
  
  template< template<class> class _OperatorTemplate , class FieldSetsT  >
  struct make_grid_variant_operator_t
  {
    static inline OperatorNodeCreateFunction make_factory(const std::string& opname)
    {
      using instantiable_field_sets = details::valid_template_field_sets_t< _OperatorTemplate , FieldSetsT >;
      static constexpr bool empty_instantiable_field_sets = FieldSetsEmpty<instantiable_field_sets>::value;

      if( empty_instantiable_field_sets )
      {
        ldbg << "ignore factory for "<<opname <<", field sets : ";
        details::PrintFieldSets< FieldSetsT >::print_field_sets( ldbg , "" , " , " , "" );
        ldbg << std::endl;
        return nullptr;
      }
      else if( OperatorNodeFactory::debug_verbose_level() >= 3 )
      {
        ldbg << "generate factory for "<<opname <<", instanciable field sets : ";
        details::PrintFieldSets< instantiable_field_sets >::print_field_sets( ldbg , "" , " , " , "" );
        ldbg << std::endl;
      }

      OperatorNodeCreateFunction factory = [] (const YAML::Node& node, const OperatorNodeFlavor& flavor) -> std::shared_ptr<OperatorNode>
        {
          if( OperatorNodeFactory::debug_verbose_level() >= 3 )
          {
            ldbg << "instantiable_field_sets :" << std::endl;
            details::PrintFieldSets< instantiable_field_sets >::print_field_sets( ldbg );
          }
          std::shared_ptr<OperatorNode> op = details::MakeGridVariantOperatorHelper< _OperatorTemplate , instantiable_field_sets >::make_operator(node,flavor);
          return op;        
        };
        
      return factory;
    }
  };
  
  template< template<class> class _OperatorTemplate , class FieldSetsT >
  struct OperatorNodeFactoryGenerator< make_grid_variant_operator_t<_OperatorTemplate,FieldSetsT> >
  {
    static inline OperatorNodeCreateFunction make_factory(const std::string& opname) { return make_grid_variant_operator_t<_OperatorTemplate,FieldSetsT>::make_factory(opname) ; }
  };

  template<class FieldSetsT>
  struct make_grid_variant_operator_helper_t
  {
    template< template<class> class _OperatorTemplate > static inline constexpr OperatorNodeFactoryGenerator< make_grid_variant_operator_t<_OperatorTemplate,FieldSetsT> > _make_grid_variant_operator = {} ;
  };

# ifdef XSTAMPV2_OVERRIDE_DEFAULT_FIELDS_SETS
# define make_grid_variant_operator make_grid_variant_operator_helper_t<XSTAMPV2_OVERRIDE_DEFAULT_FIELDS_SETS>::_make_grid_variant_operator
# else
# define make_grid_variant_operator make_grid_variant_operator_helper_t<XSTAMP_ENABLED_FIELD_SETS>::_make_grid_variant_operator
# endif

}


