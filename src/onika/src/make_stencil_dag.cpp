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
#include <onika/dag/dag_algorithm.h>

#include <onika/dac/stencil.h>
#include <onika/dac/box_span.h>

#include <vector>
#include <onika/oarray.h>
#include <algorithm>
#include <cassert>
#include <limits>
#include <chrono>

#include <onika/range.h>

#ifndef NDEBUG
#include <unordered_set>
#endif

namespace onika
{

  namespace dag
  {

    namespace dag_algorithm_opt_details
    {

      template<class _IndexT = int64_t>
      struct CellGraphTmpl
      {
        static_assert( std::is_integral_v<_IndexT> , "Indexing type must be an integral" );
        using IndexT = std::make_signed_t<_IndexT>;
        using SizeT = std::make_unsigned_t<_IndexT>;
        std::vector<IndexT> m_idx_storage;
        std::vector<IndexT> m_storage;
        IndexT * __restrict__ m_max_dist = nullptr;
        IndexT * __restrict__ m_start = nullptr;
        IndexT * __restrict__ m_task_indices = nullptr;
        SizeT m_n_nodes = 0;
        
        inline CellGraphTmpl( SizeT n_nodes )
          : m_n_nodes( n_nodes )
        {
          //m_storage.clear();
          m_storage.reserve(m_n_nodes);
          m_idx_storage.resize( m_n_nodes * 4 );
          m_max_dist = m_idx_storage.data();
          m_task_indices = m_idx_storage.data() + m_n_nodes;
          m_start = m_idx_storage.data() + m_n_nodes*2;
          for(SizeT i=0;i<m_n_nodes;++i)
          {
            m_max_dist[i] = -1;
            m_task_indices[i] = -1;
            m_start[i*2] = 0;
            m_start[i*2+1] = 0;
          }
        }

        inline size_t memory_bytes() const
        {
          return sizeof(CellGraphTmpl) + sizeof(IndexT)*m_idx_storage.capacity() + sizeof(IndexT)*m_storage.capacity() ;
        }
        
        inline size_t size() const { return m_n_nodes; }

        inline auto task_indices() { return make_iterator_range_view(m_task_indices,m_task_indices+m_n_nodes); }

        inline void start_insertion(IndexT i) { m_start[i*2] = m_storage.size(); }
        inline void end_insertion(IndexT i) { m_start[i*2+1] = m_storage.size(); }
        inline void insert(IndexT d) { m_storage.push_back( d ); }

        inline auto deps_range(SizeT i)
        {
          assert( i < size() );
          auto end_it = m_storage.end();
          if( size_t(m_start[i*2+1]) < m_storage.size() ) end_it = m_storage.begin() + m_start[i*2+1];
          return make_iterator_range_view( m_storage.begin() + m_start[i*2] , end_it );
        }

        inline bool has_cycle( SizeT i )
        {
          auto md = m_max_dist[i];
          if( md == -1 ) return true;
          m_max_dist[i] = -1;
          for( auto d : deps_range(i) )
          {
            if( has_cycle(d) ) { m_max_dist[i]=md; return true; }
          }
          m_max_dist[i] = md;
          return false;
        }
        
        static constexpr IndexT mark_bit = IndexT(1) << (sizeof(IndexT)*8-1);
        inline bool node_marked(SizeT i) const { return ( m_max_dist[i] & mark_bit ) != 0 ; }
        inline void node_mark_set(SizeT i) const { m_max_dist[i] |= mark_bit; }
        inline void node_mark_clear(SizeT i) const { m_max_dist[i] &= ~mark_bit; }
        inline void mark_sub_graph(SizeT i)
        {
          if( node_marked(i) ) return;
          node_mark_set(i);
          for( auto d : deps_range(i) )
          {
            mark_sub_graph(d);
          }
        }
        inline void unmark_sub_graph( SizeT i )
        {
          if( ! node_marked(i) ) return;
          node_mark_clear(i);
          for( auto d : deps_range(i) )
          {
            unmark_sub_graph(d);
          }
        }

        inline void compact( SizeT cell_idx )
        {
          auto s = m_start[cell_idx*2];
          auto e = m_start[cell_idx*2+1];
          
          if( (e-s) > 1 )             
          {
            auto dep_maxd = m_max_dist[ m_storage[s] ];
            auto dep_mind = m_max_dist[ m_storage[e-1] ];
            assert( dep_mind <= dep_maxd );
            assert( dep_maxd <= m_max_dist[cell_idx] );              
            if( dep_maxd != dep_mind )
            {
              for(auto j=s;j<e;j++)
              {
                if( node_marked( m_storage[j] ) ) m_storage[j] = -1;
                else mark_sub_graph( m_storage[j] );
              }
              auto k = m_start[cell_idx*2];
              for(auto j=s;j<e;j++)
              {
                if( m_storage[j] != -1 ) m_storage[k++] = m_storage[j];
              }
              m_start[cell_idx*2+1] = k;              
              s = m_start[cell_idx*2]; e = m_start[cell_idx*2+1];
              for(auto j=s;j<e;j++) { unmark_sub_graph( m_storage[j] ); }
            }
          }
        }

        inline IndexT max_dep_dist( SizeT i )
        {
          auto md = m_max_dist[i];
          if( md == -1 )
          {
            for( auto d : deps_range(i) )
            {
              md = std::max( md , max_dep_dist(d) );
            }
            ++ md;
          }
          m_max_dist[i] = md;
          return md;
        }
                
      };
      
      //using CellGraph = CellGraphTmpl<>;
    }

    using namespace dag_algorithm_opt_details;

    // ***************************** Access pattern to DAG *********************************
    // span information is NOT missing, because we only treat intra parallel task dependecies
    template<class CellGraph, class WorkShareDAGT >
    static inline void make_stencil_dag_internal( WorkShareDAGT& result,
                                                  const dac::abstract_box_span_t & span , 
                                                  const dac::AbstractStencil & stencil , 
                                                  size_t* stats , 
                                                  uint16_t* max_depth ,
                                                  bool use_patch_traversal ,
                                                  bool enable_transitive_reduction )
    {
      static constexpr size_t Nd = WorkShareDAGT::n_dims;
      static constexpr bool is_wsdag2 = std::is_same_v< WorkShareDAGT , WorkShareDAG2<Nd> >;
      using IndexT = typename CellGraph::IndexT;

      //static constexpr std::integral_constant<size_t,Nd> ndims{};
      assert( Nd == stencil.m_ndims );
      assert( Nd == span.ndims );

      // reset statistics counters
      if( stats != nullptr )
      {
        for(size_t i=0;i<DagBuildStats::STENCIL_DAG_STAT_COUNT;i++) stats[i] = 0;
      }

      // timer for performance statistics
      auto dep_construction_start = std::chrono::high_resolution_clock::now();
          
      oarray_t<size_t,Nd> domain; initialize_array( domain , span.coarse_domain );
      const size_t n_cells = domain_size( domain );
      const size_t grainsize = span.grainsize * stencil.m_scaling;

      // convert stencil masks to list of neighbor dependencies
      auto nbh_deps = dac::stencil_dep_graph<Nd>( stencil , grainsize );

      oarray_t<size_t,Nd> stencil_dep_box_size = ZeroArray<size_t,Nd>::zero;
      for(const auto& d : nbh_deps)
      {
        for(size_t k=0;k<Nd;k++)
        {
          stencil_dep_box_size[k] = std::max( stencil_dep_box_size[k] , size_t(std::abs(d[k])) );
        }
      }
      for(size_t k=0;k<Nd;k++) stencil_dep_box_size[k] += 1;
      size_t stencil_dep_box_elements = domain_size( stencil_dep_box_size );
      assert( stencil_dep_box_elements > 0 );

      // if DAG happens to be trivial, even though it is not based on a local stencil, we inform caller that this DAG is trivial
      if( stencil_dep_box_elements == 1 )
      {
        //std::cout<<"no neighborhodd dependencies, graph is trivial\n";
        result.clear();
        return ;
      }

      // stencil box filling curve
      // compute a traversal pattern for a rectangular area as large as the stencil bounding box
      // std::cout<<"traversal patch size = "<<format_array(stencil_dep_box_size)<<std::endl;
      oarray_t<size_t,Nd> patch_traversal[stencil_dep_box_elements];
      for(size_t i=0;i<stencil_dep_box_elements;i++)
      {
        auto c = index_to_coord( i, stencil_dep_box_size );
        bool rev = false;
        for(ssize_t j=(Nd-1);j>=0;j--)
        {
          if( rev ) c[j] = stencil_dep_box_size[j] - 1 - c[j];
          if( c[j]%2 != 0 ) rev = ! rev;
        }
        // std::cout<<std::setw(2)<< i<<" :" << format_array(c) <<std::endl;
        patch_traversal[i] = c;
      }
      
      // not really usefull, mainly for debugging and to produce exemples
      if( ! use_patch_traversal )
      {
        stencil_dep_box_elements = 1;
        for(size_t k=0;k<Nd;k++)
        {
          stencil_dep_box_size[k] = 1;
          patch_traversal[0][k] = 0;
        }
      }

      // dependences graph temporary structure
      CellGraph graph_tmp( n_cells );

      // indirection table counter
      size_t index_counter = 0;

      // sort function for task dependences
      auto sort_descending_maxdist = [&graph_tmp] ( size_t a , size_t b ) -> bool
        {
          assert(a<graph_tmp.size());
          assert(b<graph_tmp.size());
          auto da = graph_tmp.m_max_dist[a];
          auto db = graph_tmp.m_max_dist[b];
          return ( da > db ) || ( da == db && a < b );
        };


      // ************ building waves of tasks with corresponding dependences *************
      
      auto cdims = domain;
      for(size_t i=0;i<Nd;i++) cdims[i] = (cdims[i]/stencil_dep_box_size[i])+1;
      size_t cncells = domain_size(cdims);
      
      IndexT all_max_dist = -1;
      IndexT current_dep_dist = -1;
      bool strictly_increasing_dep_dist = true;
      
      for(size_t p=0;p<stencil_dep_box_elements;p++)
      {
        for(size_t i=0;i<cncells;i++)
        {
          auto c = to_signed( index_to_coord( i , cdims ) );
          for(size_t k=0;k<Nd;k++) c[k] = c[k] * stencil_dep_box_size[k] + patch_traversal[p][k];
          if( in_range(c,domain) )
          {
            size_t cidx = coord_to_index(c,domain);
            assert( graph_tmp.m_max_dist[cidx] == -1 ); // not already processed

            assert( index_counter < graph_tmp.size() );
            graph_tmp.m_task_indices[index_counter++] = cidx;
            
            // look for dependencies around
            graph_tmp.start_insertion( cidx );
            IndexT md = 0;
            for( const auto& dpos : nbh_deps )
            {
              auto nbh_c = dpos;
              for(size_t k=0;k<Nd;k++) nbh_c[k] += c[k];
              if( in_range(nbh_c,domain) )
              {
                size_t d = coord_to_index( nbh_c , domain );
                if( graph_tmp.m_max_dist[d] >=0 ) // it means item has been visited already, hence we build a dependence from it
                {
                  graph_tmp.insert( d );
                  md = std::max( md , graph_tmp.m_max_dist[d] + 1 );
                }
              }
            }
            graph_tmp.end_insertion(cidx);
            graph_tmp.m_max_dist[cidx] = md;
            strictly_increasing_dep_dist = strictly_increasing_dep_dist && ( md >= current_dep_dist );
            current_dep_dist = md;
            all_max_dist = std::max( all_max_dist , md );
            
            // sort and reduce dependences list
            assert( graph_tmp.m_storage.size() == size_t(graph_tmp.m_start[cidx*2+1]) );
            auto r = graph_tmp.deps_range( cidx );
            std::sort( r.begin() , r.end() , sort_descending_maxdist );
            if( enable_transitive_reduction )
            {
              graph_tmp.compact( cidx );
              graph_tmp.m_storage.resize( graph_tmp.m_start[cidx*2+1] );
            }
          }
        }
      }
      assert( graph_tmp.size() == n_cells );
      assert( index_counter == n_cells );
//      std::cout<<"wave time="<<std::chrono::nanoseconds(std::chrono::high_resolution_clock::now()-T0).count()/1000000.0<<" ms"<<std::endl; T0=std::chrono::high_resolution_clock::now();
      // *******************************************************

#     ifndef NDEBUG
      size_t total_deps = 0;
      for(size_t cell_idx=0; cell_idx<n_cells; ++cell_idx )
      {
        total_deps += graph_tmp.deps_range(cell_idx).size();
        assert( graph_tmp.m_max_dist[cell_idx] >= 0 );
        std::unordered_set<size_t> s;
        for( auto d : graph_tmp.deps_range(cell_idx) )
        {
          assert( d>=0 );
          assert( s.find(d) == s.end() );
          s.insert(d);
        }
      }
      
//      std::cout<<"all_max_dist="<<all_max_dist<<", inc_dist="<<std::boolalpha<<strictly_increasing_dep_dist<< ", deps="<<total_deps <<", bytes="<< graph_tmp.memory_bytes()<< std::endl<<std::flush;

      std::vector<ssize_t> md_backup( graph_tmp.m_max_dist , graph_tmp.m_max_dist+graph_tmp.size() );
      for(size_t cell_idx=0;cell_idx<n_cells;cell_idx++)
      {
        graph_tmp.m_max_dist[cell_idx] = -1;
      }
      for(auto cell_idx : graph_tmp.task_indices())
      {
        graph_tmp.max_dep_dist( cell_idx );
        assert( graph_tmp.m_max_dist[cell_idx] == md_backup[cell_idx] );
      }
#     endif
      // *******************************************************


      // ************** reorder tasks (min deps first) ************
      if( ! strictly_increasing_dep_dist )
      {
        std::cout << "Warning, non increasing dependency depth in depend list" << std::endl;
        size_t mdhist[all_max_dist+1];
        for(IndexT i=0;i<=all_max_dist;i++) mdhist[i]=0;
        for(size_t i=0;i<n_cells;i++)
        {
          auto md = graph_tmp.m_max_dist[i];
          assert( md >= 0 && md <= all_max_dist );
          ++ mdhist[ md ];
        }
        size_t sum=0; for(IndexT i=0;i<=all_max_dist;i++) { auto h=mdhist[i]; mdhist[i]=sum; sum+=h; }
        assert( sum == n_cells );
        
#       ifndef NDEBUG
        for(size_t i=0;i<n_cells;i++) { graph_tmp.m_task_indices[i] = -1; }
#       endif

        for(size_t i=0;i<n_cells;i++)
        {
          auto md = graph_tmp.m_max_dist[i];
          graph_tmp.m_task_indices[ mdhist[ md ] ++ ] = i;
        }
      }

      // initialiaze statistics counters
      if( stats != nullptr )
      {
        auto nanosecs = std::chrono::nanoseconds( std::chrono::high_resolution_clock::now() - dep_construction_start ).count();
        stats[DagBuildStats::SCRATCH_PEAK_MEMORY] = graph_tmp.memory_bytes();
        stats[DagBuildStats::DEP_CONSTRUCTION_TIME] = static_cast<size_t>( nanosecs * 0.001 );
        stats[DagBuildStats::NODEP_TASK_COUNT] = 0;
        stats[DagBuildStats::MAX_DEP_DEPTH] = 0;
      }

#     ifndef NDEBUG
      for(size_t i=0;i<n_cells;i++) { assert( graph_tmp.m_task_indices[i] != -1 ); }
      size_t last_md = 0;
      for(auto cell_idx : graph_tmp.task_indices())
      {
        assert( graph_tmp.m_max_dist[cell_idx] >= ssize_t(last_md) );
        last_md = graph_tmp.m_max_dist[cell_idx];
      }
#     endif

      // count total number of dependencies
      size_t n_deps = 0;
      for(size_t cell_idx=0;cell_idx<n_cells;cell_idx++)
      {
        auto nd = graph_tmp.deps_range(cell_idx).size();
        assert( ( graph_tmp.m_max_dist[cell_idx]>0 && nd>0 ) || ( graph_tmp.m_max_dist[cell_idx]==0 && nd==0 ) );
        n_deps += nd;
      }

      // final copy to DAG standardized structure
      result.m_start.clear();
      result.m_start.reserve( n_cells + 1 );
      result.m_deps.clear();
      result.m_deps.reserve( n_deps );
      result.m_start.push_back( result.m_deps.size() ); // push first offset ( 0 )
      if constexpr ( is_wsdag2 ) { result.m_coords.clear(); result.m_coords.reserve(n_cells); }

      size_t task_count = 0;
      for(auto i : graph_tmp.task_indices())
      {
        if constexpr ( is_wsdag2 ) result.m_coords.push_back( array_add( index_to_coord(i,domain) , span.lower_bound ) );
        if constexpr ( ! is_wsdag2 ) result.m_deps.push_back( array_add( index_to_coord(i,domain) , span.lower_bound ) );
        
        size_t n_deps = 0;
        for(auto d : graph_tmp.deps_range(i))
        {
          if constexpr ( is_wsdag2 ) result.m_deps.push_back( graph_tmp.m_max_dist[d] ); // 
          if constexpr ( ! is_wsdag2 ) result.m_deps.push_back( array_add( index_to_coord(d,domain) , span.lower_bound ) );
          ++ n_deps;
        }
        result.m_start.push_back( result.m_deps.size() );
        
        size_t mdepth = graph_tmp.m_max_dist[i];
        if( max_depth != nullptr )
        {
            assert( mdepth < std::numeric_limits<uint16_t>::max() );
            max_depth[ task_count ] = mdepth;
        }
        if( stats != nullptr )
        {
          if( n_deps == 0 ) ++ stats[DagBuildStats::NODEP_TASK_COUNT];
          assert( stats[DagBuildStats::MAX_DEP_DEPTH] <= mdepth );
          stats[DagBuildStats::MAX_DEP_DEPTH] = std::max( stats[DagBuildStats::MAX_DEP_DEPTH] , mdepth );
        }
        
        // re-use graph_tmp.m_max_dist to store index in final result
        // so that we have a map table from index 'd' (dependency index in graph_tmp) to dependency index in final result graph
        graph_tmp.m_max_dist[i] = task_count;

        ++ task_count;
      }
      if( stats != nullptr )
      {
        assert( stats[DagBuildStats::MAX_DEP_DEPTH] == size_t(all_max_dist) );
      }
      assert_schedule_ordering( result );
    }
    
    // ---------------------------------------------------------------------------------
    
    template<size_t Nd>
    WorkShareDAG<Nd> make_stencil_dag( const dac::abstract_box_span_t & span ,
                                       const dac::AbstractStencil & stencil , 
                                       size_t* stats , 
                                       uint16_t* max_depth, 
                                       bool patch_traversal , 
                                       bool transitive_reduction )
    {
      oarray_t<size_t,Nd> domain; initialize_array( domain , span.coarse_domain );
      const size_t n_cells = domain_size( domain );
      WorkShareDAG<Nd> result;
      if( n_cells < (1ull<<24) )
      {
        make_stencil_dag_internal< CellGraphTmpl<int32_t> , WorkShareDAG<Nd> >( result, span, stencil, stats, max_depth, patch_traversal, transitive_reduction );
      }
      else
      {
        make_stencil_dag_internal< CellGraphTmpl<int64_t> , WorkShareDAG<Nd> >( result, span, stencil, stats, max_depth, patch_traversal, transitive_reduction );
      }
      return result;
    }

    template WorkShareDAG<1> make_stencil_dag<1>( const dac::abstract_box_span_t & , const dac::AbstractStencil & , size_t* , uint16_t* ,bool, bool );
    template WorkShareDAG<2> make_stencil_dag<2>( const dac::abstract_box_span_t & , const dac::AbstractStencil & , size_t* , uint16_t* ,bool, bool );
    template WorkShareDAG<3> make_stencil_dag<3>( const dac::abstract_box_span_t & , const dac::AbstractStencil & , size_t* , uint16_t* ,bool, bool );


    // ---------------------------------------------------------------------------------

    template<size_t Nd>
    WorkShareDAG2<Nd> make_stencil_dag2( const dac::abstract_box_span_t & span ,
                                         const dac::AbstractStencil & stencil ,
                                         size_t* stats ,
                                         uint16_t* max_depth, 
                                         bool patch_traversal , 
                                         bool transitive_reduction )
    {
      oarray_t<size_t,Nd> domain; initialize_array( domain , span.coarse_domain );
      const size_t n_cells = domain_size( domain );
      WorkShareDAG2<Nd> result;
      if( n_cells < (1ull<<24) )
      {
        make_stencil_dag_internal< CellGraphTmpl<int32_t> , WorkShareDAG2<Nd> >( result, span, stencil, stats, max_depth, patch_traversal, transitive_reduction );
      }
      else
      {
        make_stencil_dag_internal< CellGraphTmpl<int64_t> , WorkShareDAG2<Nd> >( result, span, stencil, stats, max_depth, patch_traversal, transitive_reduction );
      }
      return result;
    }

    template WorkShareDAG2<1> make_stencil_dag2<1>( const dac::abstract_box_span_t & , const dac::AbstractStencil & , size_t* , uint16_t* ,bool, bool );
    template WorkShareDAG2<2> make_stencil_dag2<2>( const dac::abstract_box_span_t & , const dac::AbstractStencil & , size_t* , uint16_t* ,bool, bool );
    template WorkShareDAG2<3> make_stencil_dag2<3>( const dac::abstract_box_span_t & , const dac::AbstractStencil & , size_t* , uint16_t* ,bool, bool );
  }
}

