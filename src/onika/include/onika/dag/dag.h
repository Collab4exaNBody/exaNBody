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
#include <cassert>
#include <onika/oarray.h>
#include <numeric>
#include <vector>
#include <algorithm>
#include <type_traits>
#include <unordered_set>
#include <unordered_map>

#include <onika/oarray.h>
#include <onika/dac/dac.h>
#include <onika/range.h>

#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <onika/cuda/stl_adaptors.h>

namespace onika
{

  namespace dag
  {

    // virtual interface
    struct AbstractWorkShareDAG
    {
      virtual size_t number_of_dimensions() const { return 0; }
      virtual size_t number_of_items() const { return 0; }
      virtual size_t item_dep_count( size_t ) const { return 0; }
      virtual const size_t* item_coord( size_t ) const { return nullptr; } 
      virtual const size_t* item_dep( size_t , size_t ) const { return nullptr; } 
      virtual bool empty() const { return true; }
      virtual void clear() {}
      template<size_t Nd> inline oarray_t<size_t,Nd> to_coord_nd( const size_t* __restrict__ ap ) const
      {
        assert( Nd == number_of_dimensions() );
        oarray_t<size_t,Nd> a{};
        for(size_t k=0;k<Nd;k++) a[k] = ap[k];
        return a;
      }
      template<size_t Nd> inline oarray_t<size_t,Nd> item_coord_nd( size_t i ) const { return to_coord_nd<Nd>( item_coord(i) ); }
      template<size_t Nd> inline oarray_t<size_t,Nd> item_dep_nd( size_t i , size_t j ) const { return to_coord_nd<Nd>(item_dep(i,j)); }
      virtual ~AbstractWorkShareDAG() = default;
/*
      inline bool equals( const AbstractWorkShareDAG& rhs )
      {
        const size_t nd = number_of_dimensions();
        const size_t n = number_of_items();
        if( n != rhs.number_of_items() ) return false;
        for(size_t i=0;i<n;i++)
        {
          const size_t ndeps = item_dep_count(i);
          if( ndeps != rhs.item_dep_count(i) ) return false;
          for(size_t j=0;j<ndeps;j++)
          {
            const size_t* id1 = item_dep(i,j);
            const size_t* id2 = rhs.item_dep(i,j);
            for(size_t k=0;k<nd;k++) if(id1[k]!=id2[k]) return false;
          }
        }
        return true;
      }
*/ 
    };


    // ----------------------------------------------------------------

    template<size_t Nd , class IndexT=size_t >
    struct WorkShareDAG
    {
      static constexpr size_t n_dims = Nd;
      using index_t = IndexT;
      using coord_t = oarray_t<index_t,n_dims>;
      using dependency_t = coord_t;
      using const_iterator_t = typename std::vector<coord_t>::const_iterator;
      static inline constexpr size_t number_of_dimensions() { return n_dims; }
      inline size_t number_of_items() const { if(m_start.empty()) return 0; else return m_start.size()-1; }
      inline const coord_t& item_coord( size_t i ) const
      {
        assert( m_rem_first_dep ); 
        assert( i < m_start.size() );
        return m_deps[m_start[i]];
      }      
      inline const coord_t& item_dep(size_t i, size_t j) const
      {
        assert( i < m_start.size() );
        size_t k = m_start[i] + m_rem_first_dep + j;
        assert( k < m_deps.size() );
        return m_deps[ k ];
      }      
      inline std::pair<size_t,size_t> item_deps_irange( size_t i ) const { return { m_start[i]+m_rem_first_dep , m_start[i+1] }; }
      inline size_t item_dep_count( size_t i ) const { auto [s,e]=item_deps_irange(i); assert(s<=e); return e-s; }
      inline IteratorRangeView<const_iterator_t> item_deps( size_t i ) const { return { m_deps.begin()+m_start[i]+m_rem_first_dep , m_deps.begin()+m_start[i+1] }; }
      inline size_t total_dep_count() const { return m_deps.size() - ( m_rem_first_dep * number_of_items() ); }
      inline bool empty() const { return m_start.empty(); }
      inline void clear() { m_start.clear(); m_deps.clear(); }
      inline bool operator == ( const WorkShareDAG<Nd,IndexT>& rhs ) const
      {
        return m_rem_first_dep == rhs.m_rem_first_dep
            && m_start == rhs.m_start
            && m_deps == rhs.m_deps;
      }

      std::vector<index_t> m_start; // size = n_cells + 1 
      std::vector<coord_t> m_deps; // size = total number of dependencies. first coord is the coord of item to process
      bool m_rem_first_dep = true; // if true first element is reference item coordinate, and it is removed from dependency list
    };


    // ----------------------------------------------------------------

    template<size_t Nd, class IndexT=size_t >
    struct WorkShareDAG2
    {
      static constexpr size_t n_dims = Nd;
      using index_t = IndexT;
      using coord_t = oarray_t<index_t,n_dims>;
      using dependency_t = index_t;
      using const_iterator_t = typename std::vector<IndexT>::const_iterator;
      static inline constexpr size_t number_of_dimensions() { return n_dims; }
      inline size_t number_of_items() const { if(m_start.empty()) return 0; else return m_start.size()-1; }
      inline const coord_t& item_coord(size_t i) const { return m_coords[i]; }
      // separate method used in mixed host/device functions
      ONIKA_HOST_DEVICE_FUNC inline coord_t item_coord_cu(size_t i) const { return cuda::vector_data(m_coords)[i]; }
      inline size_t item_dep_idx(size_t i, size_t j) const
      {
        assert( i < m_start.size() );
        size_t k = m_start[i] + j;
        assert( k < m_deps.size() );
        return m_deps[k];
      }
      inline const coord_t& item_dep(size_t i, size_t j) const { return m_coords[ item_dep_idx(i,j) ]; }
      inline std::pair<size_t,size_t> item_deps_irange( size_t i ) const { return { m_start[i] , m_start[i+1] }; }
      inline size_t item_dep_count( size_t i ) const { auto [s,e]=item_deps_irange(i); assert(s<=e); return e-s; }
      inline IteratorRangeView<const_iterator_t> item_deps( size_t i ) const { return { m_deps.begin()+m_start[i] , m_deps.begin()+m_start[i+1] }; }
      inline size_t total_dep_count() const { return m_deps.size(); }
      inline bool empty() const { return m_start.empty(); }
      inline void clear() { m_start.clear(); m_deps.clear(); m_coords.clear(); }
      inline bool operator == ( const WorkShareDAG2<Nd,IndexT>& rhs ) const { return m_start == rhs.m_start && m_deps == rhs.m_deps && m_coords==rhs.m_coords; }
      std::vector<index_t> m_start; // size = number of tasks + 1 
      std::vector<index_t> m_deps; // size = total number of dependencies
      memory::CudaMMVector<coord_t> m_coords; // size = total number of tasks.
    };

    template<class IndexT>
    struct WorkShareDAG2<0,IndexT>
    {
      static constexpr size_t Nd = 0;
      static constexpr size_t n_dims = 0;
      using index_t = IndexT;
      using coord_t = oarray_t<index_t,n_dims>;
      using dependency_t = index_t;
      using const_iterator_t = const coord_t *;
      static inline constexpr size_t number_of_dimensions() { return n_dims; }
      static inline constexpr size_t number_of_items() { return 0; }
      static inline constexpr coord_t item_coord(size_t) { return {}; }
      static inline constexpr size_t item_dep_idx(size_t, size_t) { return 0; }
      static inline constexpr coord_t item_dep(size_t, size_t) { return {}; }
      static inline constexpr std::pair<size_t,size_t> item_deps_irange( size_t ) { return {0,0}; }
      static inline constexpr size_t item_dep_count( size_t ) { return 0; }
      static inline constexpr IteratorRangeView<const_iterator_t> item_deps( size_t ) { return { nullptr,nullptr}; }
      static inline constexpr size_t total_dep_count() { return 0; }
      static inline constexpr bool empty() { return true; }
      static inline constexpr void clear() { }
      inline constexpr bool operator == ( const WorkShareDAG2<0,IndexT>& ) { return true; }
    };


    // ----------------------------------------------------------------

    // helper template to detect OperatorTaskGraph compatible types
    template<class T> struct IsWorkShareDag : public std::false_type {};
    template<size_t N, class I> struct IsWorkShareDag< WorkShareDAG<N,I> > : public std::true_type {};
    template<size_t N, class I> struct IsWorkShareDag< WorkShareDAG2<N,I> > : public std::true_type {};
    template<class T> static inline constexpr bool is_workshare_dag_v = IsWorkShareDag<T>::value;


    // ----------------------------------------------------------------

    // workshare dag reference wrapper
    template<class WSDag , class = std::enable_if_t< is_workshare_dag_v<WSDag> > >
    struct WorkShareDAGAdapter final : public AbstractWorkShareDAG
    {
      using coord_t = typename WSDag::coord_t;
      WSDag m_dag;
      inline size_t number_of_dimensions() const override final
      {
        return m_dag.number_of_dimensions();
      }
      inline size_t number_of_items() const override final
      {
        return m_dag.number_of_items();
      }            
      inline size_t item_dep_count( size_t i ) const override final
      {
        return m_dag.item_dep_count(i);
      }
      inline const size_t* item_coord( size_t i ) const override final
      {
        if constexpr ( WSDag::n_dims > 0 ) return m_dag.item_coord(i).data();
        return nullptr;
      }      
      inline const size_t* item_dep( size_t i, size_t j) const override final
      {
        if constexpr ( WSDag::n_dims > 0 ) return m_dag.item_dep(i,j).data();
        return nullptr;
      }
      inline bool empty() const override final
      {
        return m_dag.empty();
      }
      virtual void clear() override final { m_dag.clear(); }
    };


    // ----------------------------------------------------------------

    template<class T> struct IsWorkShareDagAdaptor : public std::false_type {};
    template<class T> struct IsWorkShareDagAdaptor< WorkShareDAGAdapter<T> > : public std::true_type {};
    template<class T> static inline constexpr bool is_workshare_dag_adaptor_v = IsWorkShareDagAdaptor<T>::value;
    template<class T> static inline constexpr bool is_workshare_dag_or_adaptor_v = is_workshare_dag_v<T> || is_workshare_dag_adaptor_v<T>; 

  }
}

