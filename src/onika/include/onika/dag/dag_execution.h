#pragma once

#include <cstdlib>
#include <cassert>
#include <vector>
#include <unordered_set>
#include <unordered_set>
#include <atomic>

#include <onika/dag/dag.h>
#include <onika/memory/allocator.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/stl_adaptors.h>
#include <onika/force_assert.h>

namespace onika
{

  namespace dag
  {

    template<size_t Nd, class IndexT = size_t, class DepCountT = uint32_t , class DepsStorageT = memory::CudaMMVector< IndexT > , class DepCountArrayT = memory::CudaMMVector<DepCountT> >
    struct WorkShareDAGExecutionT
    {
      using onika_cu_memory_order_t = onika::cuda::onika_cu_memory_order_t;
    
      static_assert( sizeof(DepCountT) == sizeof(std::atomic<DepCountT>) && alignof(DepCountT) == alignof(std::atomic<DepCountT>) , "atomic alias cannot work" );

      DepsStorageT m_out_deps_storage;
      DepCountArrayT m_dep_countdown;

      inline WorkShareDAGExecutionT<Nd,IndexT,DepCountT,cuda::VectorShallowCopy<IndexT>,cuda::VectorShallowCopy<DepCountT> > shallow_copy()
      {
        return { { m_out_deps_storage.data(), m_out_deps_storage.size() } , { m_dep_countdown.data() , m_dep_countdown.size() } };
      }

      ONIKA_HOST_DEVICE_FUNC
      inline void set_dep_counter(size_t i, DepCountT value , onika_cu_memory_order_t mo = ONIKA_CU_MEM_ORDER_RELEASE /*std::memory_order_release*/ )
      {
        auto * dep_count_ptr = cuda::vector_data( m_dep_countdown );
        ONIKA_CU_ATOMIC_STORE( dep_count_ptr[i] , value , mo );
      }

      ONIKA_HOST_DEVICE_FUNC
      inline DepCountT decrease_dep_counter(size_t i , onika_cu_memory_order_t mo = ONIKA_CU_MEM_ORDER_RELEASE /*std::memory_order mo = std::memory_order_release*/ )
      {
        auto * dep_count_ptr = cuda::vector_data( m_dep_countdown );
        return ONIKA_CU_ATOMIC_SUB( dep_count_ptr[i] , 1 , mo );
      }

      inline DepCountT decrease_dep_counter_unsafe(size_t i)
      {
        auto dc = ONIKA_CU_ATOMIC_LOAD( m_dep_countdown[i] , ONIKA_CU_MEM_ORDER_RELAXED ); //.load(std::memory_order_relaxed);
        ONIKA_CU_ATOMIC_STORE( m_dep_countdown[i] , dc - 1 , ONIKA_CU_MEM_ORDER_RELAXED ); //.store( dc - 1 , std::memory_order_relaxed );
        return dc;
      }
      
      ONIKA_HOST_DEVICE_FUNC inline DepCountT dep_counter(size_t i , onika_cu_memory_order_t mo = ONIKA_CU_MEM_ORDER_ACQUIRE /*std::memory_order_acquire*/ ) const
      { 
        // return m_dep_countdown[i].load( mo );
        return ONIKA_CU_ATOMIC_LOAD( cuda::vector_data(m_dep_countdown)[i] , mo ); // .load( mo );
      }

      inline void rebuild_from_dag(const WorkShareDAG2<Nd>& g)
      {
        size_t n = g.number_of_items();
        std::vector< std::unordered_set<IndexT> > out_deps( n );
        //m_out_deps.resize(n);
        //for(size_t i=0;i<n;i++) m_out_deps[i].clear();
        size_t total_out_deps = 0;
        for(size_t i=0;i<n;i++)
        {
          size_t ndeps = g.item_dep_count(i);
          total_out_deps += ndeps;
          for(size_t j=0;j<ndeps;j++)
          {
            auto d = g.item_dep_idx(i,j);
            assert( d != i );
            assert( out_deps[d].find(i) == out_deps[d].end() );
            out_deps[d].insert(i);
          }
        }
        
        m_out_deps_storage.resize( 1 + n+1 + total_out_deps );
        m_out_deps_storage[ 0 ] = n;
        const size_t out_index_offset = 1 ;
        const size_t out_deps_offset = 1 + n+1;
        
        //m_out_index.resize( n + 1 );
        //m_out_deps.resize( total_out_deps );
        size_t out_dep_count = 0;
        m_out_deps_storage[ out_index_offset + 0 ] = out_dep_count;
        for(size_t i=0;i<n;i++)
        {
          size_t prev_count = out_dep_count;
          for(IndexT odep : out_deps[i]) m_out_deps_storage[ out_deps_offset + (out_dep_count++) ] = odep;
          assert( (out_dep_count-prev_count) == out_deps[i].size() );
          std::sort( m_out_deps_storage.data()+prev_count+out_deps_offset , m_out_deps_storage.data()+out_dep_count+out_deps_offset );
          m_out_deps_storage[ out_index_offset + i+1 ] = out_dep_count;
        }
      }

      inline void reset_dep_countdown(const WorkShareDAG2<Nd>& g , onika_cu_memory_order_t mo = ONIKA_CU_MEM_ORDER_RELEASE /*std::memory_order mo = std::memory_order_release*/ )
      {
        size_t n = g.number_of_items();
        
        //if( n != m_dep_countdown.size() ) { m_dep_countdown = std::move( std::vector< std::atomic<DepCountT> >(n) ); }
        m_dep_countdown.resize( n );
        
        for(size_t i=0;i<n;i++)
        {
          size_t ndeps = g.item_dep_count(i);
          set_dep_counter( i , ndeps , mo );
        }
        assert( m_dep_countdown.size() == n );
      }

      inline void assert_consistency(const WorkShareDAG2<Nd>& g) const
      {
#       ifndef NDEBUG
        assert_consistency_forced(g);
#       endif
      }
      
      inline void assert_consistency_forced(const WorkShareDAG2<Nd>& g) const
      {
        size_t n = g.number_of_items();
        ONIKA_FORCE_ASSERT( m_out_deps_storage.size() >= (n+1) );
        ONIKA_FORCE_ASSERT( n == m_out_deps_storage[0] );
        ONIKA_FORCE_ASSERT( n == m_dep_countdown.size() );
        size_t total_in_deps = 0;
        size_t total_out_deps = 0;
        for(size_t i=0;i<n;i++)
        {
          size_t ndeps = g.item_dep_count(i);
          total_in_deps += ndeps;
          total_out_deps += item_out_dep_count(i); //m_out_deps[i].size();
          ONIKA_FORCE_ASSERT( dep_counter(i) /*m_dep_countdown[i].load(std::memory_order_acquire)*/ == ndeps );
          for(size_t j=0;j<ndeps;j++)
          {
            unsigned int d = g.item_dep_idx(i,j);
            ONIKA_FORCE_ASSERT( /*d>=0 &&*/ d<n );
            ONIKA_FORCE_ASSERT( d != i );
            auto [ rstart , rend ] = item_out_deps(d);
            ONIKA_FORCE_ASSERT( std::lower_bound(rstart, rend, i) != rend );
          }
        }
        ONIKA_FORCE_ASSERT( total_in_deps == total_out_deps );
      }

      ONIKA_HOST_DEVICE_FUNC
      inline size_t number_of_items() const
      {
        //assert( cuda::vector_size(m_dep_countdown) /*.size()*/ == cuda::vector_data(m_out_deps_storage)[0] );
        return cuda::vector_data(m_out_deps_storage)[0]; // cuda::vector_size(m_dep_countdown); //m_dep_countdown.size();
      }

      ONIKA_HOST_DEVICE_FUNC
      inline size_t item_out_dep_count(size_t i) const
      {
        const auto * graph_storage = cuda::vector_data( m_out_deps_storage );
        const size_t out_index_offset = 1;
        return graph_storage[out_index_offset + i+1] - graph_storage[out_index_offset + i]; //m_out_deps[i].size();
      }

      ONIKA_HOST_DEVICE_FUNC
      inline IteratorRangeView<const IndexT*> item_out_deps(size_t i) const
      {
        const auto * graph_storage = cuda::vector_data( m_out_deps_storage );
        const size_t n = graph_storage[0];
        const size_t out_deps_offset = 1 + n+1;
        const size_t out_index_offset = 1;
        return { graph_storage + out_deps_offset + graph_storage[out_index_offset+i]
               , graph_storage + out_deps_offset + graph_storage[out_index_offset+i+1] };
      }

    };

    template<size_t Nd> using WorkShareDAGExecution = WorkShareDAGExecutionT<Nd>;
  }
}

