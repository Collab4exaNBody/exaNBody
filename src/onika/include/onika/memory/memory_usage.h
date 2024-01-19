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

#include <cstdlib>
#include <vector>
#include <array>
#include <utility>
#include <tuple>
#include <type_traits>
#include <cassert>
#include <unordered_map>
#include <string>

#include <onika/oarray.h>
#include <onika/aggregate.h>

/*
  MemoryUsage template can be specialized to help onika guess how much memory an object uses
*/

namespace onika { namespace memory
{

  struct MemoryResourceCounters
  {
    static constexpr unsigned int N_COUNTERS = 18;
    enum CounterEnum
    {
      TOTAL_PROG_MB = 0,
      RESIDENT_SET_MB,
      SHARED_PAGES,
      TEXT_MB,
      LIB_MB,
      DATA_STACK_MB,
      DIRTY_PAGES,
      PAGE_RECLAIMS,
      PAGE_FAULTS,
      SWAPS,
      BLOCK_INPUT_OPS,
      BLOCK_OUTPUT_OPS,
      MESSAGES_SENT,
      MESSAGES_RECV,
      SIGNALS_RECV,
      VOL_CONTEXT_SW,
      INVOL_CONTEXT_SW,
      PROG_BRK_INC,
      COUNTER_ENUM_END
    };
    static_assert( COUNTER_ENUM_END == N_COUNTERS , "inconsistent counter enums" );
    
    static const char * labels[N_COUNTERS];
    long page_size = 4096;
    long stats[N_COUNTERS];
    
    static inline constexpr unsigned int nb_counters() { return N_COUNTERS; }
    void read();
  };

  template<class... T> static inline size_t memory_bytes(const T&... x);

  // tells if an object class has a member method called memory_bytes
  // TODO: check it's a member method (not static), it takes no argument and it returns an integral type
  template<class T, class=void> struct has_memory_bytes_method : public std::false_type {};
  template<class T> struct has_memory_bytes_method< T , decltype(void( sizeof( std::declval<std::remove_reference_t<T> >().memory_bytes() ) )) > : public std::true_type {};
  template<class T> static inline constexpr bool has_memory_bytes_method_v = has_memory_bytes_method<T>::value;

  template<class T>
  struct MemoryUsage
  {
    static inline constexpr size_t memory_bytes(const T& obj)
    {
      if constexpr ( has_memory_bytes_method_v<T> ) { return obj.memory_bytes(); }
      else if constexpr (std::is_empty_v<T>) { return 0; }
      else if constexpr (std::is_aggregate_v<T>)
      {
        if constexpr ( aggregate_members_at_least_v<T,20>) { /* give up */ return sizeof(T); }
        else if constexpr ( aggregate_members_at_least_v<T,19>) { const auto& [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19]=obj; return onika::memory::memory_bytes(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19); }
        else if constexpr ( aggregate_members_at_least_v<T,18>) { const auto& [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18]=obj; return onika::memory::memory_bytes(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18); }
        else if constexpr ( aggregate_members_at_least_v<T,17>) { const auto& [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17]=obj; return onika::memory::memory_bytes(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17); }
        else if constexpr ( aggregate_members_at_least_v<T,16>) { const auto& [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16]=obj; return onika::memory::memory_bytes(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16); }
        else if constexpr ( aggregate_members_at_least_v<T,15>) { const auto& [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15]=obj; return onika::memory::memory_bytes(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15); }
        else if constexpr ( aggregate_members_at_least_v<T,14>) { const auto& [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14]=obj; return onika::memory::memory_bytes(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14); }
        else if constexpr ( aggregate_members_at_least_v<T,13>) { const auto& [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13]=obj; return onika::memory::memory_bytes(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13); }
        else if constexpr ( aggregate_members_at_least_v<T,12>) { const auto& [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12]=obj; return onika::memory::memory_bytes(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12); }
        else if constexpr ( aggregate_members_at_least_v<T,11>) { const auto& [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11]=obj; return onika::memory::memory_bytes(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11); }
        else if constexpr ( aggregate_members_at_least_v<T,10>) { const auto& [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]=obj; return onika::memory::memory_bytes(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10); }
        else if constexpr ( aggregate_members_at_least_v<T,9> ) { const auto& [x1,x2,x3,x4,x5,x6,x7,x8,x9]=obj; return onika::memory::memory_bytes(x1,x2,x3,x4,x5,x6,x7,x8,x9); }
        else if constexpr ( aggregate_members_at_least_v<T,8> ) { const auto& [x1,x2,x3,x4,x5,x6,x7,x8]=obj; return onika::memory::memory_bytes(x1,x2,x3,x4,x5,x6,x7,x8); }
        else if constexpr ( aggregate_members_at_least_v<T,7> ) { const auto& [x1,x2,x3,x4,x5,x6,x7]=obj; return onika::memory::memory_bytes(x1,x2,x3,x4,x5,x6,x7); }
        else if constexpr ( aggregate_members_at_least_v<T,6> ) { const auto& [x1,x2,x3,x4,x5,x6]=obj; return onika::memory::memory_bytes(x1,x2,x3,x4,x5,x6); }
        else if constexpr ( aggregate_members_at_least_v<T,5> ) { const auto& [x1,x2,x3,x4,x5]=obj; return onika::memory::memory_bytes(x1,x2,x3,x4,x5); }
        else if constexpr ( aggregate_members_at_least_v<T,4> ) { const auto& [x1,x2,x3,x4]=obj; return onika::memory::memory_bytes(x1,x2,x3,x4); }
        else if constexpr ( aggregate_members_at_least_v<T,3> ) { const auto& [x1,x2,x3]=obj; return onika::memory::memory_bytes(x1,x2,x3); }
        else if constexpr ( aggregate_members_at_least_v<T,2> ) { const auto& [x1,x2]=obj; return onika::memory::memory_bytes(x1,x2); }
        else if constexpr ( aggregate_members_at_least_v<T,1> ) { const auto& [x1]=obj; return onika::memory::memory_bytes(x1); }
      }
      return sizeof(T);
    }
  };

  template<class T, class A>
  struct MemoryUsage< std::vector<T,A> >
  {
    static inline size_t memory_bytes(const std::vector<T,A>& obj)
    {
      assert( obj.capacity() >= obj.size() );
      size_t nbytes = sizeof(obj) + (obj.capacity()-obj.size())*sizeof(T);
      for(const auto& x:obj) nbytes += onika::memory::memory_bytes(x);
      return nbytes;
    }
  };

  template<class T, size_t N>
  struct MemoryUsage< oarray_t<T,N> >
  {
    static inline size_t memory_bytes(const oarray_t<T,N>& obj)
    {
      size_t nbytes = 0;
      for(const auto& x:obj) nbytes += onika::memory::memory_bytes(x);
      return nbytes;
    }
  };

  template<class T, size_t N>
  struct MemoryUsage< std::array<T,N> >
  {
    static inline size_t memory_bytes(const std::array<T,N>& obj)
    {
      size_t nbytes = 0;
      for(const auto& x:obj) nbytes += onika::memory::memory_bytes(x);
      return nbytes;
    }
  };

  template<class T, size_t N>
  struct MemoryUsage< T[N] >
  {
    static inline size_t memory_bytes(const T(&obj)[N] )
    {
      size_t nbytes = 0;
      for(size_t i=0;i<N;i++) nbytes += onika::memory::memory_bytes(obj[i]);
      return nbytes;
    }
  };

  template<class T1, class T2>
  struct MemoryUsage< std::pair<T1,T2> >
  {
    static inline size_t memory_bytes(const std::pair<T1,T2>& obj )
    {
      return onika::memory::memory_bytes( obj.first , obj.second );
    }
  };

  template<class... T>
  struct MemoryUsage< std::tuple<T...> >
  {
    static inline size_t memory_bytes(const std::tuple<T...>& obj )
    {
      return std::apply( onika::memory::memory_bytes , obj );
    }
  };

  template<class Key,class T,class Hash,class KeyEqual,class Allocator>
  struct MemoryUsage< std::unordered_map<Key,T,Hash,KeyEqual,Allocator> >
  {
    using ObjT = std::unordered_map<Key,T,Hash,KeyEqual,Allocator>;
    static inline size_t memory_bytes(const ObjT& obj )
    {
      size_t nbytes = sizeof(ObjT);
      for(const auto& p:obj) { nbytes += sizeof(typename ObjT::node_type) - sizeof(Key) - sizeof(T) + onika::memory::memory_bytes(p); }
      return nbytes;
    }
  };

  template<class CharT,class Traits,class Allocator>
  struct MemoryUsage< std::basic_string<CharT,Traits,Allocator> >
  {
    using ObjT = std::basic_string<CharT,Traits,Allocator>;
    static inline size_t memory_bytes(const ObjT& obj )
    {
      const uint8_t* sptr = reinterpret_cast<const uint8_t*>( obj.data() );
      const uint8_t* objptr = reinterpret_cast<const uint8_t*>( &obj );
      auto pdiff = sptr - objptr;
      size_t nbytes = sizeof(ObjT);
      if( pdiff<0 || pdiff >= static_cast<decltype(pdiff)>(sizeof(ObjT)) ) // detect in-object allocation of small strings
      {
        nbytes += (obj.capacity()) * sizeof(CharT);
      }
      return nbytes;
    }
  };

  template<class... T>
  static inline size_t memory_bytes(const T&... x) { return ( ... + ( MemoryUsage<T>::memory_bytes(x) ) ); }

} }

