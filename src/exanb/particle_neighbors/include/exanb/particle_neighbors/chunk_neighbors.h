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

#include <cstdint>
#include <onika/memory/allocator.h>
#include <onika/memory/memory_partition.h>
#include <onika/memory/memory_usage.h>
#include <onika/cuda/cuda.h>
#include <onika/integral_constant.h>

#include <exanb/particle_neighbors/chunk_neighbors_specializations.h>
#include <exanb/core/basic_types_def.h>
#include <exanb/core/log.h>

#include <algorithm>
#include <utility>

namespace exanb
{

  // number of cells => for each cell : number of chunks => chunks
  //using GridChunkNeighborsData = onika::memory::CudaMMVector< onika::memory::CudaMMVector<uint16_t> >;

  using GridChunkNeighborsData = uint16_t const * const * ;

  struct GridChunkNeighbors
  {  
    static inline constexpr unsigned int MAX_STREAM_OFFSET_TABLES = 2;
    static inline constexpr uint16_t CELL_ENCODED_REL_COORD_0 = ( ( ( 16 << 5 ) + 16 ) << 5 ) + 16;
  
    onika::memory::MemoryPartionnerMT m_fixed_stream_pool = {};
    onika::memory::CudaMMVector<uint16_t *> m_cell_stream;
    onika::memory::CudaMMVector<uint32_t> m_cell_stream_size;
    size_t m_stream_pool_hint = 0; // guessed best value for m_fixed_stream_pool capacity
    size_t m_nb_dyn_alloc = 0;
    unsigned int m_chunk_size = 4;
    onika::memory::GenericHostAllocator m_alloc = {};
    
    GridChunkNeighborsData data() const ;
    const uint16_t * cell_stream(size_t i) const ;
    size_t cell_stream_size(size_t i) const ;
    size_t number_of_cells() const;
    void update_stream_pool_hint();
    void set_number_of_cells(size_t n);
    void set_chunk_size( unsigned int cs );

    inline ~GridChunkNeighbors()
    {
      m_stream_pool_hint = 0;
      realloc_stream_pool( 0.0 );
    }

    inline void realloc_stream_pool(double allocation_hint_factor = 1.1 )
    {
      if( m_fixed_stream_pool.capacity() > 0 )
      {
        m_alloc.deallocate( m_fixed_stream_pool.base_ptr() , m_fixed_stream_pool.memory_bytes() );
      }
      else if( m_fixed_stream_pool.m_base_ptr != nullptr )
      {
        lerr<<"Internal Error : m_fixed_stream_pool.m_base_ptr must be null here"<<std::endl;
        std::abort();
      }
      
      m_fixed_stream_pool.set_allocated_memory( nullptr , 0 );

      const size_t desired_size = static_cast<size_t>( allocation_hint_factor * m_stream_pool_hint );
      const size_t alloc_size = m_fixed_stream_pool.adjust_allocation_size( desired_size );
      if( alloc_size < desired_size )
      {
        lerr << "WARNING: adpated pre-alloc capacity from "<<desired_size<<" to "<< alloc_size << " bytes" << std::endl;
      }

      if( alloc_size > 0 )
      {
        const size_t al = std::max( alignof(uint16_t) , static_cast<size_t>(onika::memory::MINIMUM_CUDA_ALIGNMENT) );
        m_fixed_stream_pool.set_allocated_memory( (uint8_t*) m_alloc.allocate(alloc_size,al) , alloc_size );
      }
    }

    template<class AllocatorT = onika::memory::CudaManagedAllocator<uint16_t> >
    inline void clear( AllocatorT alloc = {} )
    {
      size_t n = m_cell_stream.size();
      assert( n == m_cell_stream_size.size() );
      for(size_t cell_i=0;cell_i<n;cell_i++)
      {
        if( m_cell_stream[cell_i] != nullptr )
        {
          if( ! m_fixed_stream_pool.contains( m_cell_stream[cell_i] ) )
          {
            m_alloc.deallocate( m_cell_stream[cell_i] , m_cell_stream_size[cell_i] );
          }
        }
        m_cell_stream[cell_i] = nullptr;
        m_cell_stream_size[cell_i] = 0;
      }
      m_fixed_stream_pool.clear();
      m_cell_stream.clear();
      m_cell_stream_size.clear();
    }

  };
  
  template<bool Symmetric=false>
  struct GridChunkNeighborsLightWeightIt
  {
    using is_symmetrical_t = onika::BoolConst<Symmetric>;
    GridChunkNeighborsData m_nbh_streams = nullptr;
    unsigned int m_chunk_size = 4;
    inline GridChunkNeighborsLightWeightIt(const GridChunkNeighbors& chnbh) : m_nbh_streams(chnbh.m_cell_stream.data()) , m_chunk_size(chnbh.m_chunk_size) {}
    ONIKA_HOST_DEVICE_FUNC inline static constexpr is_symmetrical_t is_symmetrical() { return {}; }
  };

  static constexpr size_t GRID_CHUNK_NBH_MAX_CHUNK_SIZE = 64;
  static constexpr size_t GRID_CHUNK_NBH_MAX_AMR_RES = 31;
  static constexpr uint16_t GRID_CHUNK_NBH_MIN_CELL_ENC_VALUE = ( ( (1u<<5) + 1u ) << 5 ) + 1u;
  static_assert( GRID_CHUNK_NBH_MIN_CELL_ENC_VALUE == 1057 , "bad computed value for minimum encoded cell value" );

  ONIKA_HOST_DEVICE_FUNC inline uint16_t encode_cell_index(const IJK& relloc)
  {
    // 3 groups of 5 bits, each represent a position of the neighbor cell relative to cell of interest
    assert( relloc.i >= -15 && relloc.i <= 15 );
    assert( relloc.j >= -15 && relloc.j <= 15 );
    assert( relloc.k >= -15 && relloc.k <= 15 );
    uint16_t c = relloc.k + 16;
    c = c << 5;
    c += relloc.j + 16;
    c = c << 5;
    c += relloc.i + 16;
    assert( c!=0 && c>=GRID_CHUNK_NBH_MIN_CELL_ENC_VALUE ); // must be guaranteed for safe particle offset injection
    return c;
  }

  ONIKA_HOST_DEVICE_FUNC inline IJK decode_cell_index(uint16_t x)
  {
    assert( x!=0 && x>=GRID_CHUNK_NBH_MIN_CELL_ENC_VALUE ); // must be guaranteed for safe particle offset injection
    IJK relloc;
    relloc.i = static_cast<ssize_t>( x & 31 ) - 16;
    x = x >> 5; 
    relloc.j = static_cast<ssize_t>( x & 31 ) - 16;
    x = x >> 5; 
    relloc.k = static_cast<ssize_t>( x & 31 ) - 16;
    return relloc;
  }

  struct ChunkNeighborsStreamInfo
  {
    const uint16_t* stream = nullptr;
    const uint32_t* offset = nullptr;
    const uint32_t* dual_offset = nullptr;
    int32_t shift = 0;
  };
  
  ONIKA_HOST_DEVICE_FUNC inline ChunkNeighborsStreamInfo chunknbh_stream_info(const uint16_t* stream , int32_t n_particles )
  {
    if(stream==nullptr || n_particles==0) return { nullptr , nullptr , nullptr , 0 };
    if( stream[0] <= GridChunkNeighbors::MAX_STREAM_OFFSET_TABLES && stream[1] == 0 )
    {
      const int num_offset_tables = stream[0];
      const int offset_table_size = ( num_offset_tables >= 1 ) ? ( ( n_particles * num_offset_tables + 1 ) * 2 ) : 0 ;
      return {
        stream + offset_table_size , 
        ( num_offset_tables >= 1 ) ? reinterpret_cast<const uint32_t*>(stream) : nullptr , 
        ( num_offset_tables >= 2 ) ? ( reinterpret_cast<const uint32_t*>(stream) + n_particles + 1 ) : nullptr ,
        - num_offset_tables };
    }
    return { stream , nullptr , nullptr , 0 };
  }

  ONIKA_HOST_DEVICE_FUNC inline const uint16_t* chunknbh_stream_to_next_particle(const uint16_t* stream , unsigned int chunk, unsigned int nchunks , unsigned int cg , unsigned int cell_groups )
  {
    assert( chunk <= nchunks );
    stream += nchunks - chunk;
    for(;cg<cell_groups;cg++)
    {
      ++ stream;
      stream += (*stream) + 1;
    }
    return stream;
  }

  template<class CellT, class FuncT>
  ONIKA_HOST_DEVICE_FUNC inline void chunknbh_apply_cell_stream(const CellT* cells, const GridChunkNeighbors& nbh, FuncT func)
  {
    size_t n_cells = nbh.number_of_cells();
    for(size_t i=0;i<n_cells;i++)
    {
      size_t np = cells[i].size();
      auto sinfo = chunknbh_stream_info( nbh.m_cell_stream[i] , np );
      func( i, sinfo.stream , np );
    }
  }

}


// *************** memory consumption ****************
namespace onika
{
  namespace memory
  {
    template<>
    struct MemoryUsage< exanb::GridChunkNeighbors >
    {
      static inline size_t memory_bytes(const exanb::GridChunkNeighbors& v )
      {
        size_t mem_bytes = sizeof( v );
        mem_bytes += v.m_cell_stream.capacity() * sizeof(uint16_t*);
        mem_bytes += v.m_cell_stream_size.capacity() * sizeof(uint32_t);
        mem_bytes += v.m_fixed_stream_pool.capacity();
        size_t n = v.m_cell_stream_size.size();
        assert( n == v.m_cell_stream.size() );
        for(size_t i=0;i<n;i++)
        {
          if( ! v.m_fixed_stream_pool.contains(v.m_cell_stream[i]) ) mem_bytes += v.m_cell_stream_size[i]*sizeof(uint16_t);
        }
        return mem_bytes;
      }
    };
  }
}
//****************************************************


