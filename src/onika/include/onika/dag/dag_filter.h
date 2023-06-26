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
#include <onika/dag/dag_algorithm.h>
#include <onika/debug.h>

namespace onika
{

  namespace dag
  {

    template<size_t Nd, class FilterFunc >
    static inline WorkShareDAG2<Nd> filter_dag( const WorkShareDAG2<Nd>& dag , const FilterFunc& filter )
    {
      std::vector<ssize_t> reduce_map;
      return filter_dag( dag , filter , reduce_map );
    }

    template<size_t Nd>
    inline bool mark_subgraph(const WorkShareDAG2<Nd>& dag , size_t i , std::vector<bool>& marks )
    {
      if( marks[i] ) return false;
      marks[i] = true;
      size_t ndeps = dag.item_dep_count(i);
      for(size_t j=0;j<ndeps;j++)
      {
        mark_subgraph( dag , dag.item_dep_idx(i,j) , marks );
      }
      return true;
    }

    template<size_t Nd>
    inline void unmark_subgraph(const WorkShareDAG2<Nd>& dag , size_t i , std::vector<bool>& marks )
    {
      if( marks[i] )
      {
        marks[i] = false;
        size_t ndeps = dag.item_dep_count(i);
        for(size_t j=0;j<ndeps;j++)
        {
          unmark_subgraph( dag , dag.item_dep_idx(i,j) , marks );
        }
      }
    }

    template<size_t Nd>
    inline void append_deps_subgraph(const WorkShareDAG2<Nd>& dag , const std::vector<ssize_t>& reduce_map , size_t i , WorkShareDAG2<Nd>& out , std::vector<bool>& marks )
    {
      if( mark_subgraph(dag,i,marks) )
      {
        if( reduce_map[i] != -1 )
        {
          assert( reduce_map[i] < ssize_t(out.m_coords.size()) );
          out.m_deps.push_back( reduce_map[i] );
        }
        else
        {
          size_t ndeps = dag.item_dep_count(i);
          for(size_t j=0;j<ndeps;j++)
          {
            append_deps_subgraph( dag , reduce_map , dag.item_dep_idx(i,j) , out , marks );
          }
        }
      }
    }

    template<size_t Nd, class FilterFunc >
    static inline WorkShareDAG2<Nd> filter_dag( const WorkShareDAG2<Nd>& dag , const FilterFunc& filter , std::vector<ssize_t>& reduce_map )
    {
      const size_t n_cells = dag.number_of_items();    

#     ifndef NDEBUG
      /*
        assertion 1 : dependences are sorted according to their descending 'dependence depth'
        => this assertion ensures that progressively marking the dependency graph for transitive reduction works
      */
      {
        auto dep_depth = dag_dependency_depth(dag);
        for(size_t i=0;i<n_cells;i++)
        {
          size_t ndeps = dag.item_dep_count(i);
          unsigned int dd = std::numeric_limits<unsigned int>::max();
          for(size_t j=0;j<ndeps;j++)
          {
            auto di = dag.item_dep_idx(i,j);
            assert( dep_depth[di] <= dd );
            dd = dep_depth[di];
          }
        }
      } // frees dep_depth
#     endif
    
      reduce_map.resize( n_cells );
      std::vector<bool> marks( n_cells , false );
      
      size_t n_filtered_cells = 0;
      for(size_t i=0;i<n_cells;i++)
      {
        if( filter(i) ) reduce_map[i] = n_filtered_cells++;
        else reduce_map[i] = -1;
      }

      WorkShareDAG2<Nd> result;
      if( n_filtered_cells == 0 ) return result;
      
      result.m_start.reserve( n_filtered_cells+1 );
      result.m_coords.reserve( n_filtered_cells );
      result.m_deps.reserve( ( dag.m_deps.size() * n_filtered_cells ) / n_cells );
      for(size_t i=0;i<n_cells;i++)
      {
        if( reduce_map[i] != -1 )
        {
          assert( reduce_map[i] == ssize_t(result.m_coords.size()) );
          result.m_start.push_back( result.m_deps.size() );
          size_t ndeps = dag.item_dep_count(i);
          result.m_coords.push_back( dag.item_coord(i) );
          for(size_t j=0;j<ndeps;j++)
          {
            append_deps_subgraph( dag, reduce_map , dag.item_dep_idx(i,j) , result , marks );
          }
          for(size_t j=0;j<ndeps;j++)
          {
            unmark_subgraph( dag , dag.item_dep_idx(i,j) , marks );
          }
        }
      }
      
      if( !result.m_start.empty() )
      {
        result.m_start.push_back( result.m_deps.size() );
      }
      assert_schedule_ordering( result );
      return result;
    }
    
  }
}

