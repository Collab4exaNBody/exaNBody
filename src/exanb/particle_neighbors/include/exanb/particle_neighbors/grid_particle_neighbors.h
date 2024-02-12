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
#include <cstdlib>

#include <exanb/core/particle_id_codec.h>
#include <onika/memory/allocator.h>

namespace exanb
{
  

  struct CellParticleNeighbors
  {
    // same size as number of particle in cell
    onika::memory::CudaMMVector< uint32_t > nbh_start;
    // size = total number of bonds starting at particles in cell
    onika::memory::CudaMMVector< uint64_t > neighbors;
  };

  using GridParticleNeighbors = std::vector< CellParticleNeighbors >;

  template<int NA=1>
  struct GridParticleNeighborsIterator
  {
    inline void start_cell( size_t cell_a , size_t n_particles )
    {
      // upon direction change
      // size_t n_particles = m_nbh[0][cell_a].nbh_start.size();
      for(int i=0;i<NA;i++)
      {
        assert( n_particles == m_nbh[i][cell_a].nbh_start.size() );
        m_nbh_start[i] = m_nbh[i][cell_a].nbh_start.data();
        m_neighbors[i] = m_nbh[i][cell_a].neighbors.data();
      }
      //return n_particles;
    }

    inline void start_particle( size_t p_a )
    {
      m_pa = p_a;
      m_dir = -1;
      m_neighbor_index = 0;
      m_neighbor_end = 0;
      next();
    }

    inline void next()
    {
      ++ m_neighbor_index;
      while( m_neighbor_index >= m_neighbor_end && m_dir < (NA-1) )
      {
        ++ m_dir;
        m_neighbor_index = 0;
        if( m_pa > 0 ) { m_neighbor_index = m_nbh_start[m_dir][m_pa-1]; }
        m_neighbor_end = m_nbh_start[m_dir][m_pa];
      }
    }
    
    inline bool end() const
    {
      return m_neighbor_index >= m_neighbor_end;
    }

    inline void get_nbh(size_t& cell_b, size_t& p_b) const
    {
      assert( ! end() );
      uint64_t nbh = m_neighbors[m_dir][m_neighbor_index];
      decode_cell_particle(nbh, cell_b, p_b );            
    }


    // input neighbor data
    const CellParticleNeighbors * m_nbh[NA];

    // temporary variables
    const uint32_t* __restrict__ m_nbh_start[NA];
    const uint64_t* __restrict__ m_neighbors[NA];
    ssize_t m_pa;
    ssize_t m_neighbor_index;
    ssize_t m_neighbor_end;
    int m_dir;
  };

}


