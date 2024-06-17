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

#include <cstdlib> // for size_t
#include <type_traits>

namespace onika
{
  namespace soatl
  {

    template<typename _field_id> struct FieldId
    /*{
      using value_type = void;
      using Id = _field_id;
      static const char* name() { return "<uknown>"; }
    }
    */;

    template<> struct FieldId<void>
    {
      using value_type = void;
      using Id = void;
      static const char* name() { return "<uknown>"; }
    };


    // field to index utilities
    static constexpr size_t bad_field_index = 1ull << 30; // no more than 1 billion field ids are allowed

    template<typename... ids> struct FieldIds {};

    namespace details
    {
      template<size_t I> struct increment_field_index { static constexpr size_t value = I+1; };
      template<> struct increment_field_index<bad_field_index> { static constexpr size_t value = bad_field_index; };

      template<bool B, size_t I> struct zero_or_increment { static constexpr size_t value = 0; };
      template<size_t I> struct zero_or_increment<false,I> { static constexpr size_t value = increment_field_index<I>::value; };

      template<typename k, typename... ids> struct find_index_of_id {};

      template<typename k, typename f, typename... ids>
      struct find_index_of_id<k,f,ids...>
      {
        static constexpr size_t index = zero_or_increment<
            std::is_same<k,f>::value ,
            find_index_of_id<k,ids...>::index
          >::value;
      };

      template<typename k>
      struct find_index_of_id<k> { static constexpr size_t index = bad_field_index; };


      template<class k, class Fids> struct find_index_of_id_in_field_ids;
      template<class k, class... ids> struct find_index_of_id_in_field_ids< k, FieldIds<ids...> > { static constexpr size_t index = find_index_of_id<k,ids...>::index; };
    }

    template<typename k, typename... ids> using find_index_of_id = details::find_index_of_id<k,ids...>;
    template<typename k, typename... ids> static inline constexpr size_t find_index_of_id_v = find_index_of_id<k,ids...>::index ;
    template<class k, class Fids> static inline constexpr size_t find_index_of_id_in_field_ids_v = details::find_index_of_id_in_field_ids<k,Fids>::index;

    // add a field id in front of an existing FieldIds
    template<typename k, typename Fids> struct PrependId {};
    template<typename k, typename... ids> struct PrependId<k,FieldIds<ids...> > { using type=FieldIds<k,ids...>; };
    template<typename k, typename Fids> using prepend_field_id_t = typename PrependId<k,Fids>::type;


    // sub list of field ids preceding k  
    template<typename k, typename Fids> struct PrecedingIds {};
    template<typename k> struct PrecedingIds<k,FieldIds<> >
    {
      using type = FieldIds<>;
    };
    template<typename k, typename id1, typename... ids> struct PrecedingIds<k,FieldIds<id1,ids...> >
    {
      using type = std::conditional_t<
        std::is_same_v<k,id1> ,
          FieldIds<> ,
          prepend_field_id_t<id1 , typename PrecedingIds<k,FieldIds<ids...> >::type > 
          >;
    };
    template<typename k, typename... ids> using preceding_field_ids_t = typename PrecedingIds< k, FieldIds<ids...> >::type;

    // find n first contiguous field ids that includes a specific fielSet
    template<size_t N, class Fids> struct NFirstFields { using type = FieldIds<>; };
    template<size_t N, class id1, class... ids> struct NFirstFields<N, FieldIds<id1,ids...> >
    {
      using type = std::conditional_t< N==0 , FieldIds<> , prepend_field_id_t<id1 , typename NFirstFields<N-1,FieldIds<ids...> >::type > >;
    };

    template<class FidsRef , class FidsSubSet> struct FieldIdsIncludingSequence;    
    template<class FidsRef , class... SubSetIds>
    struct FieldIdsIncludingSequence< FidsRef , FieldIds<SubSetIds...> >
    {
      static inline constexpr size_t n_fields()
      {
	auto MAX = [](size_t a,size_t b) ->size_t { return a>=b ? a : b ; };
        size_t i=0; ( ... , ( i=MAX(i,size_t(1)+find_index_of_id_in_field_ids_v<SubSetIds,FidsRef>) ) );
        return i;
      }
      using type = typename NFirstFields< n_fields() , FidsRef >::type;
    };
  
  }
  
}

