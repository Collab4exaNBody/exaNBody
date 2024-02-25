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

#include <onika/memory/allocator.h>

namespace exanb
{
  template<class _NeighborOffsetT = uint64_t , class _ParticleIndexT = uint32_t , class _NeighborCountT = uint16_t >
  struct FlatPartNbhListT
  {
    using NeighborOffset = _NeighborOffsetT;
    using ParticleIndex = _ParticleIndexT;
    using NeighborCount = _NeighborCountT;
    onika::memory::CudaMMVector< NeighborOffset > m_neighbor_offset; // size = number of particles + 1 , ast one is the total size
    onika::memory::CudaMMVector< ParticleIndex > m_neighbor_list;    // size = total number of particle pairs
    onika::memory::CudaMMVector< NeighborCount > m_half_count;       // size = number of particles    
  };
  using FlatPartNbhList = FlatPartNbhListT<>;

}

