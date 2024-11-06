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
#include <onika/dac/stencil.h>
#include <onika/dac/box_span.h>
#include <onika/dag/dag.h>
#include <onika/debug.h>

namespace onika
{

  namespace dag
  {

    // ****************************************************
    // ************* DAG construction kit *****************
    // ****************************************************

    struct DagBuildStats
    {
      static constexpr size_t SCRATCH_PEAK_MEMORY    = 0;
      static constexpr size_t DEP_CONSTRUCTION_TIME  = 1;
      static constexpr size_t NODEP_TASK_COUNT       = 2;
      static constexpr size_t MAX_DEP_DEPTH          = 3;
      static constexpr size_t STENCIL_DAG_STAT_COUNT = 4;
    };
    using stencil_dag_sats_t = size_t[ DagBuildStats::STENCIL_DAG_STAT_COUNT ];

    template<class DagT , class = std::enable_if_t< is_workshare_dag_or_adaptor_v<DagT> > >
    inline void assert_schedule_ordering( const DagT& ONIKA_DEBUG_ONLY(dag) )
    {
#     ifndef NDEBUG
      using coord_t = typename DagT::coord_t;
      std::unordered_set< coord_t > scheduled_set;
      size_t n = dag.number_of_items();
      for(size_t i=0;i<n;i++)
      {
        auto c = dag.item_coord(i);
        size_t ndeps = dag.item_dep_count(i);
        for(size_t j=0;j<ndeps;j++)
        {
          auto d = dag.item_dep(i,j);
          //std::cout << "check "<<format_array(c)<<" -> "<<format_array(d)<<std::endl;
          assert( c != d );
          assert( scheduled_set.find(d) != scheduled_set.end() );
        }
        scheduled_set.insert( c );
        //std::cout << "insert "<<format_array(c)<<std::endl;
      }
#     endif
    }

    // ----------------------------------------------------------------

    template<class Dag1T, class Dag2T , class = std::enable_if_t< is_workshare_dag_or_adaptor_v<Dag1T> && is_workshare_dag_or_adaptor_v<Dag2T> > >
    inline void assert_equal_dag( const Dag1T& lhs , const Dag2T& rhs )
    {
#     ifndef NDEBUG
      const size_t nd = lhs.number_of_dimensions();
      assert( nd == rhs.number_of_dimensions() );
      const size_t n = lhs.number_of_items();
      assert( n == rhs.number_of_items() );
      for(size_t i=0;i<n;i++)
      {
        const size_t ndeps = lhs.item_dep_count(i);
        assert( ndeps == rhs.item_dep_count(i) );
        for(size_t j=0;j<ndeps;j++)
        {
          auto id1 = lhs.item_dep(i,j);
          auto id2 = rhs.item_dep(i,j);
          if constexpr ( std::is_pointer_v<decltype(id1)> && std::is_pointer_v<decltype(id2)> ) { for(size_t k=0;k<nd;k++) assert( id1[k] == id2[k] ); }
          else { assert( id1 == id2 ); }
        }
      }
#     endif
    }

    template<size_t Nd>
    extern std::unordered_map<oarray_t<size_t,Nd>,size_t> dag_item_index_map( const WorkShareDAG<Nd>& dag );

    template<size_t Nd>
    extern std::vector<uint16_t> dag_dependency_depth( const WorkShareDAG<Nd>& dag );

    template<size_t Nd>
    extern std::vector<uint16_t> dag_dependency_depth( const WorkShareDAG2<Nd>& dag );
    
    template<size_t Nd>
    extern WorkShareDAG<Nd> make_stencil_dag_legacy( const dac::abstract_box_span_t & span , const dac::AbstractStencil & stencil );

    template<size_t Nd>
    extern WorkShareDAG<Nd> make_stencil_dag( const dac::abstract_box_span_t & span , 
                                              const dac::AbstractStencil & stencil , 
                                              size_t* stats = nullptr , 
                                              uint16_t* max_depth = nullptr ,
                                              bool patch_traversal = true ,
                                              bool transitive_reduction = true);

    template<size_t Nd>
    extern WorkShareDAG2<Nd> make_stencil_dag2( const dac::abstract_box_span_t & span ,
                                                const dac::AbstractStencil & stencil , 
                                                size_t* stats = nullptr , 
                                                uint16_t* max_depth = nullptr ,
                                                bool patch_traversal = true ,
                                                bool transitive_reduction = true );

    extern WorkShareDAG<3> make_3d_neighborhood_exclusion_dag( const oarray_t<size_t,3>& dims );

    template<size_t Nd>
    extern WorkShareDAG<Nd> make_co_stencil_dag(const dac::abstract_box_span_t & co_span, const WorkShareDAG<Nd>& dag, const std::unordered_set< oarray_t<int,Nd> >& co_dep_graph );

    template<size_t Nd>
    extern void shift_dag_coords( WorkShareDAG<Nd>& dag , const oarray_t<size_t,Nd>& v);

    template<size_t Nd>
    extern void dag_closure( WorkShareDAG<Nd>& dag , size_t i, std::unordered_set< oarray_t<size_t,Nd> > & closure );

    template<size_t Nd>
    inline WorkShareDAG<1> flatten_nd_dag( const oarray_t<size_t,Nd>& dims , const WorkShareDAG<Nd>& nd_dag )
    {
      WorkShareDAG<1> result;
      result.m_start = nd_dag.m_start;
      result.m_deps.clear();
      result.m_deps.reserve( nd_dag.m_deps.size() );
      for(const auto& c : nd_dag.m_deps) { result.m_deps.push_back( { coord_to_index(c,dims) } ); }
      assert_schedule_ordering(result);
      return result;
    }
    
  }
}

