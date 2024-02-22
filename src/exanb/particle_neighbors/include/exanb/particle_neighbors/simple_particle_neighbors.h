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
#include <onika/integral_constant.h>
#include <type_traits>

namespace exanb
{
  
  struct SimpleNeighbor
  {
    uint32_t m_cell_idx = 0;
    uint32_t m_particle_idx = 0;
    inline bool operator < ( const SimpleNeighbor& rhs ) { return m_cell_idx<rhs.m_cell_idx || ( m_cell_idx==rhs.m_cell_idx && m_particle_idx<rhs.m_particle_idx ); }
  };
  
  struct CellSimpleParticleNeighbors
  {
    // same size as number of particle in cell
    onika::memory::CudaMMVector< uint32_t > nbh_start;
    // size = total number of bonds starting at particles in cell
    onika::memory::CudaMMVector< SimpleNeighbor > neighbors;
  };

  struct GridSimpleParticleNeighbors
  {
    using is_symmetrical_t = onika::BoolConst<false>;
    onika::memory::CudaMMVector< CellSimpleParticleNeighbors > m_cell_neighbors;
    inline size_t number_of_cells() const { return m_cell_neighbors.size(); }
  };

}


