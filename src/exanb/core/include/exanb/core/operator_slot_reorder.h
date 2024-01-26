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

#include <exanb/core/operator_slot.h>
#include <utility>

namespace exanb
{

  namespace operator_slot_details
  {
    template<class T> struct IsOperatorSlot { static constexpr bool value = false; };
    
    template<class T, bool IsInputOnly, bool HasYAMLConversion>
    struct IsOperatorSlot< OperatorSlot<T,IsInputOnly,HasYAMLConversion> > { static constexpr bool value = true; };
    
    template<class T> struct OperatorSlotCountInputOnly { static constexpr int value = 0; };
    template<class T, bool HasYAMLConversion>
    struct OperatorSlotCountInputOnly< OperatorSlot<T,true,HasYAMLConversion> > { static constexpr int value = 1; };

    template<class... T> struct OperatorSlotPackCountInputOnly { static constexpr int value = ( ... + ( OperatorSlotCountInputOnly<T>::value ) ); };
    template<> struct OperatorSlotPackCountInputOnly<> { static constexpr int value = 0; };

    // concatenate integer sequences
    template<class A, class B> struct ConcatenateIndexSequence;
    template<int... As, int... Bs>
    struct ConcatenateIndexSequence< std::integer_sequence<int,As...> , std::integer_sequence<int,Bs...> >
    {
      using type = std::integer_sequence<int,As...,Bs...>;
    };

    // get index at position in sequence
    template<int I, class S> struct IndexSequenceAt { };
    template<int s0, int... si>
    struct IndexSequenceAt<0,std::integer_sequence<int,s0,si...> >
    {
      static constexpr int value = s0;
    };
    template<int i, int s0, int... si>
    struct IndexSequenceAt<i,std::integer_sequence<int,s0,si...> >
    {
      static constexpr int value = IndexSequenceAt<i-1,std::integer_sequence<int,si...> >::value;
    };
    template<int I, class S> static inline int index_at_v = IndexSequenceAt<I,S>::value;


    // associate each OperatorSlot with its index
    template<class... OpT> struct OpList {};
    template<int I,class OpT> struct IndexedOp {};

    template<class IS,class OpS> struct IndexedOpList;
    template<int... i,class... OpT> 
    struct IndexedOpList< std::integer_sequence<int,i...> , OpList<OpT...> >
    {
      using type = OpList< IndexedOp<i,OpT> ... >;
    };
    
    // build integer sequences for input_only and in_out slots
    template<int IStart, int OStart, class... IOps> struct InputOutputIndexList;
    template<int IStart, int OStart> struct InputOutputIndexList<IStart,OStart>
    {
      static constexpr int NextIn = IStart;
      static constexpr int NextOut = OStart;
      using in_list = std::integer_sequence<int>;
      using out_list = std::integer_sequence<int>;
      using all_placement = std::integer_sequence<int>;
    };
    // specialization for input_only slot
    template<int IStart, int OStart, int I, class T, bool HasYAMLConversion>
    struct InputOutputIndexList< IStart , OStart , IndexedOp< I , OperatorSlot<T,true,HasYAMLConversion> > > 
    {
      static constexpr int NextIn = IStart+1;
      static constexpr int NextOut = OStart;
      using in_list = std::integer_sequence<int,I>;
      using out_list = std::integer_sequence<int>;
      using all_placement = std::integer_sequence<int,IStart>;
    };
    // specialization for in_out slot
    template<int IStart, int OStart, int I, class T, bool HasYAMLConversion>
    struct InputOutputIndexList< IStart , OStart , IndexedOp< I , OperatorSlot<T,false,HasYAMLConversion> > >
    {
      static constexpr int NextIn = IStart;
      static constexpr int NextOut = OStart+1;
      using in_list = std::integer_sequence<int>;
      using out_list = std::integer_sequence<int,I>;
      using all_placement = std::integer_sequence<int,OStart>;
    };
    // specialization for recursion rule
    template<int IStart, int OStart, class T0, class... T>
    struct InputOutputIndexList<IStart,OStart,T0,T...>
    {
      using header_t = InputOutputIndexList<IStart,OStart,T0>;
      using tail_t = InputOutputIndexList<header_t::NextIn,header_t::NextOut,T...>;
      using in_list  = typename ConcatenateIndexSequence< typename header_t::in_list  , typename tail_t::in_list  >::type ;
      using out_list = typename ConcatenateIndexSequence< typename header_t::out_list , typename tail_t::out_list >::type ;
      using all_placement = typename ConcatenateIndexSequence< typename header_t::all_placement , typename tail_t::all_placement >::type ;
    };

    // final helpers for index lists
    template<int IStart, int OStart, class OpListT> struct InputOutputIndexListHelper;
    template<int IStart, int OStart, class... OpT> struct InputOutputIndexListHelper< IStart , OStart , OpList<OpT...> > { using type = InputOutputIndexList<IStart,OStart,OpT...>; };

    template<int IStart, int OStart, class... OpT> using input_output_index_list_helper_t =
      typename InputOutputIndexListHelper< IStart , OStart , typename IndexedOpList< std::make_integer_sequence<int,sizeof...(OpT)> , OpList<OpT...> >::type >::type;

    // debug: prints content of a sequence
    template<int... i> static inline void dbg_print_sequence( std::integer_sequence<int,i...> ) { ( ... , ( std::cout<<' '<<i ) ); }

    // shortcuts
    template<class T> static inline constexpr bool is_operator_slot_v = IsOperatorSlot<T>::value;
    template<class... T> static inline constexpr int count_input_only_operator_slots_v = OperatorSlotPackCountInputOnly<T...>::value;

  }

  // parameters info returned as a list containing : Nb of input only args, placement of args (in first, inout after)
  template< typename... OpT >
  static inline constexpr auto slot_args_reorder( OpT& ... )
  {
    using namespace operator_slot_details;
    constexpr int in_list_size = count_input_only_operator_slots_v<OpT...>;
    using index_list_t = input_output_index_list_helper_t< 0 , in_list_size , OpT... >;
    using in_list_t = typename index_list_t::in_list;
    using out_list_t = typename index_list_t::out_list;
    using reorder_list_t = typename ConcatenateIndexSequence< in_list_t , out_list_t >::type;
    using size_and_reorder_list_t = typename ConcatenateIndexSequence< std::integer_sequence<int,in_list_size> , reorder_list_t >::type;
    return size_and_reorder_list_t{};
  }

  template<typename... OpSlotT>
  static inline auto forward_slot_data_pointers( OpSlotT& ... args ) { return std::make_tuple( args.get_pointer() ... ); }
}


