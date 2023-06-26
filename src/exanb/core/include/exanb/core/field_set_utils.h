#pragma once

#include <exanb/core/type_utils.h>
#include "exanb/field_sets_macro.h"

namespace exanb
{

# ifdef XSTAMP_ENABLED_FIELD_SETS
  using EnabledFieldSets = XSTAMP_ENABLED_FIELD_SETS ;
# endif

  // ====================== utlity functions =========================
  template<class... _ids>
  static inline constexpr FieldSet<_ids...> make_field_set( onika::soatl::FieldId<_ids> ... ) { return {}; }

  // ====================== utlity templates =========================

  // ====================================================
  // determine if a field is rx, ry or rz, which are implicitly included in all field sets
  // ====================================================
  template<typename field_id> struct FieldContainDefaultField : std::false_type {};
  template<> struct FieldContainDefaultField<field::_rx> : std::true_type {};
  template<> struct FieldContainDefaultField<field::_ry> : std::true_type {};
  template<> struct FieldContainDefaultField<field::_rz> : std::true_type {};
  using DefaultFields = FieldSet< field::_rx, field::_ry, field::_rz >;

  
  // ====================================================
  // determine if a field set contain a specific field
  // ====================================================
  template<typename field_set, typename field_id>
  struct FieldSetContainField : std::false_type {};

  // an empty FieldSet does not contain any field, except those that are implicitly included in all field sets (rx, ry and rz)
  template<typename field_id>
  struct FieldSetContainField< FieldSet<> , field_id > 
  : std::integral_constant<bool, FieldContainDefaultField<field_id>::value > {};
  
  // for a list of fields in the FieldSet, either the field we search is the First, or it may be in the Others fields
  template<typename field_id, typename First, typename... Others>
  struct FieldSetContainField< FieldSet<First,Others...> , field_id >
  : std::integral_constant<bool, std::is_same<First,field_id>::value || FieldSetContainField< FieldSet<Others...>, field_id >::value > {};


  // ====================================================
  // determine if a field set contain another field set.
  // usefull to test requirements for field sets
  // ====================================================
  template<typename field_set, typename required_field_set>
  struct FieldSetContainFieldSet : std::false_type {};

  template<typename field_set, typename... required_fields>
  struct FieldSetContainFieldSet<field_set,FieldSet<required_fields...> >
  : std::integral_constant<bool, AndT< FieldSetContainField<field_set,required_fields>::value ... >::value > {};


  // =======================================
  // test if empty FieldSets
  // =======================================
  template<typename FS> struct FieldSetsEmpty :  std::false_type {};
  template<> struct FieldSetsEmpty< FieldSets<> > : std::true_type {};


  // =======================================
  // conditionaly add a field to a FieldSet 
  // =======================================
  template<typename field_set, typename prepend_field_id, bool cond>
  struct PrependFieldIdIf;
  
  template<typename prepend_field_id, typename... field_ids>
  struct PrependFieldIdIf< FieldSet<field_ids...> , prepend_field_id , true >
  {
    using type = FieldSet<prepend_field_id, field_ids...>;
  };

  template<typename prepend_field_id, typename... field_ids>
  struct PrependFieldIdIf< FieldSet<field_ids...> , prepend_field_id , false >
  {
    using type = FieldSet<field_ids...>;
  };

  template<typename field_set, typename prepend_field_id>
  using PrependField = typename PrependFieldIdIf<field_set,prepend_field_id,true>::type;


  // =======================================
  // unconditionaly add a field (at the end) to a FieldSet 
  // =======================================
  template<class field_set, class field_id> struct _AppendFieldId;
  template<class field_id, class... field_ids> struct _AppendFieldId< FieldSet<field_ids...> , field_id > { using field_set = FieldSet<field_ids... , field_id>; };
  template<class field_set, class field_id> using AppendFieldId = typename _AppendFieldId<field_set,field_id>::field_set;


  // ==============================================
  // remove fields in FieldSet b from FieldSet a
  // ==============================================
  template<typename field_set_a, typename field_set_b> struct RemoveFieldsHelper;

  template<typename field_set_b>
  struct RemoveFieldsHelper< FieldSet<> , field_set_b >
  {
    using type = FieldSet<>;
  };

  template<typename field_set_b, typename field_id_a, typename... other_field_ids_a>
  struct RemoveFieldsHelper< FieldSet<field_id_a,other_field_ids_a...> , field_set_b >
  {
    using type = typename PrependFieldIdIf<
      typename RemoveFieldsHelper< FieldSet<other_field_ids_a...> , field_set_b >::type
      , field_id_a , ! FieldSetContainField<field_set_b,field_id_a>::value >::type;
  };

  template<typename fsA, typename fsB>
  using RemoveFields = typename RemoveFieldsHelper<fsA,fsB>::type;



  // =============================================
  // add default fields (rx,ry,rz) to a  FieldSet
  // =============================================
  template<typename field_set_a> struct AddDefaultFieldsHelper;

  template<typename... field_ids>
  struct AddDefaultFieldsHelper< FieldSet<field_ids...> >
  {
    using type = FieldSet<field::_rx, field::_ry, field::_rz, field_ids...>;
  };
  
  template<typename fs>
  using AddDefaultFields = typename AddDefaultFieldsHelper< RemoveFields<fs,DefaultFields> >::type;
  

  // ========================
  // concat 2 FieldSets
  // ========================
  template<class field_set_a, class field_set_b> struct _ConcatFieldSet;
  template<class field_set_a > struct _ConcatFieldSet< field_set_a , FieldSet<> > { using field_set = field_set_a; };
  template<class fsa, class fb1, class... fbn >
  struct _ConcatFieldSet< fsa , FieldSet<fb1,fbn...> >
  {
    using field_set = typename _ConcatFieldSet< AppendFieldId<fsa,fb1> , FieldSet<fbn...> >::field_set;
  };
  template<class fsa, class fsb> using ConcatFieldSet = typename _ConcatFieldSet<fsa,fsb>::field_set;
  template<class fsa, class fsb> using MergeFieldSet = typename _ConcatFieldSet< fsa , RemoveFields<fsb,fsa> >::field_set;

  
  // ==============================================
  // filter FieldSets with requirements (FieldSet)
  // the resulting FieldSets contain only those
  // FieldSet which contain required fields
  // ==============================================

  template<typename _field_id, typename _field_set>
  struct AppendFieldSetT;

  template<typename _field_id, typename... _field_set_field_ids>
  struct AppendFieldSetT< _field_id, FieldSet<_field_set_field_ids...> >
  {
    using type = FieldSet<_field_set_field_ids...,_field_id>;
  };
  template<typename _field_id, typename _field_set> using AppendFieldSet = typename AppendFieldSetT<_field_id,_field_set>::type ;
  
  
  
  template<typename field_sets, typename prepend_field_set, bool cond>
  struct PrependFieldSetIf {};

  template<typename prepend_field_set, typename... field_sets>
  struct PrependFieldSetIf< FieldSets<field_sets...>, prepend_field_set, true>
  {
    using type = FieldSets<prepend_field_set, field_sets...>;
  };
  
  template<typename field_sets, typename prepend_field_set>
  struct PrependFieldSetIf< field_sets, prepend_field_set, false>
  {
    using type = field_sets;
  };

  template<typename field_sets , typename requirement >
  struct FilterFieldSets {};

  template< typename requirement >
  struct FilterFieldSets< FieldSets<> , requirement >
  {
    using type = FieldSets<>;
  };
  
  template< typename requirement, typename first_field_set , typename... other_field_sets >
  struct FilterFieldSets< FieldSets<first_field_set,other_field_sets...> , requirement >
  {
    using type = typename PrependFieldSetIf<
      typename FilterFieldSets< FieldSets<other_field_sets...> , requirement >::type ,
      first_field_set , FieldSetContainFieldSet<first_field_set,requirement>::value >::type;
  };


  // select a subset of StandardFieldSets choosing only those which contain given fields
  template<typename FS , typename... T>
  using FieldSetsWith = typename FilterFieldSets< FS, FieldSet<T...> >::type ;
  //using FieldSetsWith = typename FilterFieldSets< StandardFieldSets, FieldSet<T...> >::type ;


  // =======================================
  // intersection of two FieldSet 
  // =======================================
  template<class field_set_a, class field_set_b> struct _FieldSetIntersection;
  template<class field_set_a>
  struct _FieldSetIntersection< field_set_a , FieldSet<>  >
  {
    using type = FieldSet<>;
  };  
  template<class field_set_a, class field_id0_b, class...field_ids_b>
  struct _FieldSetIntersection< field_set_a , FieldSet<field_id0_b,field_ids_b...>  >
  {
    using type = typename PrependFieldIdIf<
      typename _FieldSetIntersection<field_set_a,FieldSet<field_ids_b...> >::type ,
      field_id0_b ,
      FieldSetContainField<field_set_a,field_id0_b>::value
      >::type;
  };
  template<class field_set_a, class field_set_b> using FieldSetIntersection = typename _FieldSetIntersection<field_set_a,field_set_b>::type;

  
  template<class field_set, class field_id> using AppendFieldId = typename _AppendFieldId<field_set,field_id>::field_set;

}

