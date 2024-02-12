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
#include <vector>
#include <exanb/particle_neighbors/chunk_neighbors.h>

namespace exanb
{
  

  // not deprecated but rarely used, prefere apply_cell_particle_neighbors
  template<typename CellT, bool Symmetric=false>
  struct ChunkParticleNeighborsIterator
  {
    inline void start_cell( size_t cell_a , size_t n_particles )
    {
      m_nbh_stream = chunknbh_stream_info( m_nbh[cell_a] /*.data()*/ , n_particles ).stream;
      m_current_particle = -1;
      m_loc_a = grid_index_to_ijk( m_dims , cell_a );
      m_cell_a = cell_a;
    }

    inline void start_particle( size_t p_a )
    {
      assert( p_a == static_cast<size_t>(m_current_particle+1) );
      m_current_particle = p_a;
      m_cell_groups = *(m_nbh_stream++);
      m_current_cell_group = -1;
      m_nbh_cell = 0;
      m_chunks = 0;
      m_current_chunk = 0;
      m_current_chunk_element = 0;
      m_nbh_cell_particles = 0;
      m_nbh_chunk = 0;
      next();
    }

    inline void next()
    {
      next_internal();
      if( m_nbh_cell==m_cell_a && ( m_nbh_chunk + m_current_chunk_element ) == m_current_particle && ! end() )
      {
        next_internal();
      }
    }

    inline void next_internal()
    {
      bool reload_chunk = false;
    
      ++ m_current_chunk_element;
      
      if( ( m_current_chunk_element >= m_chunk_size ) || ( ( m_nbh_chunk + m_current_chunk_element ) >= m_nbh_cell_particles ) )
      {
        m_current_chunk_element = 0;
        ++ m_current_chunk;
        reload_chunk = true;
      }

      if( m_current_chunk >= m_chunks )
      {
        m_current_chunk = 0;
        ++ m_current_cell_group;
        if( m_current_cell_group < m_cell_groups )
        {
          uint16_t cell_b_enc = *(m_nbh_stream++);
          IJK loc_b = m_loc_a + decode_cell_index(cell_b_enc);
          m_nbh_cell = grid_ijk_to_index( m_dims , loc_b );
          m_nbh_cell_particles = m_cells[m_nbh_cell].size();
          m_chunks = *(m_nbh_stream++);
        }
        else
        {
          reload_chunk = false;
        }
      }

      if( reload_chunk )
      {
        m_nbh_chunk = *(m_nbh_stream++);
        m_nbh_chunk *= m_chunk_size;
      }

    }

    inline bool end() const
    {
      bool sym_over = m_nbh_cell>m_cell_a || ( m_nbh_cell==m_cell_a && ( m_nbh_chunk + m_current_chunk_element ) >= m_current_particle );
      return m_current_cell_group >= m_cell_groups || ( Symmetric && sym_over );
    }
    
    inline void go_to_next_particle()
    {
      if( Symmetric )
      {
        ++ m_current_chunk;
        for(;m_current_chunk<m_chunks;m_current_chunk++) { ++m_nbh_stream; }
        ++ m_current_cell_group;
        for(;m_current_cell_group<m_cell_groups;m_current_cell_group++)
        {
          ++ m_nbh_stream; // cell id
          m_chunks = *(m_nbh_stream++); // then number of chunks
          for(m_current_chunk=0;m_current_chunk<m_chunks;m_current_chunk++) { ++m_nbh_stream; }
        }
      }
    }

    inline void get_nbh(size_t& cell_b, size_t& p_b) const
    {
      cell_b = m_nbh_cell;
      p_b = m_nbh_chunk + m_current_chunk_element;
      //return encode_cell_particle( cell_b , p_b );
    }

    // input neighbor data
    const CellT* m_cells = nullptr;
    GridChunkNeighborsData m_nbh;
    const IJK m_dims;
    const unsigned int m_chunk_size;

    // temporary variables for neighborhood traversal
    const uint16_t* __restrict__ m_nbh_stream = nullptr;
    IJK m_loc_a;
    size_t m_cell_a = 0;
    size_t m_nbh_cell = 0;
    size_t m_nbh_chunk = 0;
    unsigned int m_nbh_cell_particles = 0;
    unsigned int m_current_particle = 0;
    int m_cell_groups = 0;
    int m_current_cell_group = 0;
    unsigned int m_chunks = 0; // in current cell group
    unsigned int m_current_chunk = 0; // in current cell group
    unsigned int m_current_chunk_element = 0; // in current cell group
  };

}


