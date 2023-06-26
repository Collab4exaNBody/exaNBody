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
#include <onika/dag/dag.h>
#include <onika/dag/dag_algorithm.h>
#include <onika/dag/dag_execution.h>

#include <onika/debug.h>

namespace onika
{

  namespace dag
  {

    // priority_cmp takes two element coordinates as input and returns a bool. returns true if first element has a higher priority than the second
    template<size_t Nd, class CostFunc, class FinalIdxVector>
    inline void reorder_fast_dep_unlock( const WorkShareDAG2<Nd>& g , WorkShareDAGExecution<Nd>& dag_exe , FinalIdxVector& final_idx , const CostFunc& fcost )
    {
      //assert_schedule_ordering(g);
      size_t n = g.number_of_items();

      final_idx.clear();
      final_idx.reserve(n);

      assert( dag_exe.number_of_items() == n );

      dag_exe.reset_dep_countdown( g , std::memory_order_relaxed );
      
      std::vector<size_t> heap_idx;
      heap_idx.reserve( n );
      for(size_t i=0;i<n;i++)
      {
        if( dag_exe.dep_counter(i,std::memory_order_relaxed) == 0 )
        {
          heap_idx.push_back(i);
        }
      }
      
      auto idx_cmp = [&fcost,&dag_exe]( size_t a , size_t b ) -> bool
        {
          ssize_t unlock_time_a;
          ssize_t unlock_time_b;
          
          int oa = dag_exe.item_out_dep_count(a); // m_out_deps[a].size();
          if(oa>0) unlock_time_a = fcost(a) / oa;
          else     unlock_time_a = fcost(a); //std::numeric_limits<ssize_t>::max();
          
          int ob = dag_exe.item_out_dep_count(b); //m_out_deps[b].size();
          if(ob>0) unlock_time_b = fcost(b) / ob;
          else     unlock_time_b = fcost(b); //std::numeric_limits<ssize_t>::max();
          
          return unlock_time_a > unlock_time_b;
        };
      
      std::make_heap( heap_idx.begin() , heap_idx.end() , idx_cmp );
      
      while( final_idx.size() < n )
      {
        assert( ! heap_idx.empty() );
        std::pop_heap( heap_idx.begin() , heap_idx.end() , idx_cmp );
        auto i = heap_idx.back();
        heap_idx.pop_back();
        final_idx.push_back( i );
        for(auto d : dag_exe.item_out_deps(i) /*m_out_deps[i]*/ )
        {
          auto depcount = dag_exe.decrease_dep_counter_unsafe(d);
          assert( depcount > 0 );
          -- depcount;
          if( depcount == 0 )
          {
            heap_idx.push_back(d);
            std::push_heap( heap_idx.begin() , heap_idx.end() , idx_cmp );
          }
        }
      }

      
    }



    // priority_cmp takes two element coordinates as input and returns a bool. returns true if first element has a higher priority than the second
    template<class DagT , class PriorityFunc , class = std::enable_if_t< is_workshare_dag_or_adaptor_v<DagT> > >
    inline WorkShareDAG<DagT::n_dims> reorder_dag( const DagT& dag , PriorityFunc priority )
    {
      using coord_t = typename DagT::coord_t;

      assert( check_schedule_ordering(dag) );
      size_t n = dag.number_of_items();

      std::vector<size_t> final_idx;
      final_idx.reserve(n);
      size_t total_deps = 0;
      {
        std::vector< std::unordered_set<size_t> > out_deps( n );
        std::vector<size_t> dep_countdown( n , 0 );
        {
          std::unordered_map< coord_t , size_t > coord_to_idx;
          for(size_t i=0;i<n;i++)
          {
            coord_to_idx[ dag.item_coord(i) ] = i;
          }
          for(size_t i=0;i<n;i++)
          {
            //assert( dag.item_dep_count(i) >= 1 );
            dep_countdown[i] = dag.item_dep_count(i);
            total_deps += dep_countdown[i];
            for(auto d:dag.item_deps(i))
            {
              size_t j = coord_to_idx[d];
              assert( j != i );
              out_deps[j].insert(i);
            }
          }
        }
        
        std::vector<size_t> heap_idx;
        
        auto idx_cmp = [&dag,&priority](size_t a, size_t b) -> bool { return priority(dag.item_coord(a)) < priority(dag.item_coord(b)); };
        size_t n = dag.number_of_items();
        for(size_t i=0;i<n;i++)
        {
          if( dep_countdown[i]==0 )
          {
            heap_idx.push_back(i);
          }
        }
        std::make_heap( heap_idx.begin() , heap_idx.end() , idx_cmp );
        
        while( final_idx.size() < n )
        {
          assert( ! heap_idx.empty() );
          std::pop_heap( heap_idx.begin() , heap_idx.end() , idx_cmp );
          auto i = heap_idx.back();
          heap_idx.pop_back();
          final_idx.push_back( i );
          for(auto d:out_deps[i])
          {
            assert( dep_countdown[d] > 0 );
            -- dep_countdown[d];
            if( dep_countdown[d] == 0 )
            {
              heap_idx.push_back(d);
              std::push_heap( heap_idx.begin() , heap_idx.end() , idx_cmp );
            }
          }
        }
      }
      
      WorkShareDAG<DagT::n_dims> result;
      result.m_start.reserve(n+1);
      result.m_deps.reserve(n+total_deps);
      result.m_start.push_back( result.m_deps.size() ); // 0 = offset of first element
      for(auto i:final_idx)
      {
        result.m_deps.push_back( dag.item_coord(i) );
        for(const auto& d:dag.item_deps(i)) result.m_deps.push_back(d);
        result.m_start.push_back( result.m_deps.size() );
      }
      assert_schedule_ordering(result);
      return result;
    }

    
  }
}

