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

#include <exanb/particle_neighbors/grid_particle_neighbors.h>
#include <exanb/particle_neighbors/parallel_build_neighbors.h>
#include <exanb/core/particle_id_codec.h>

#ifndef NDEBUG
#define ABORT_ON_CHECK_FAIL() abort()
#else
#define ABORT_ON_CHECK_FAIL() (void)0
#endif

namespace exanb
{
  
  template<class GridT>
  inline bool check_neighbors(const GridT& grid, GridParticleNeighbors& pb )
  {
    
    size_t n_cells = pb.size();
    if( n_cells != grid.number_of_cells() ) { ABORT_ON_CHECK_FAIL(); return false; }
    for(size_t cell_a=0;cell_a<n_cells;cell_a++)
    {
      size_t n_particles = pb[cell_a].nbh_start.size();
      if( grid.cell_number_of_particles(cell_a) != n_particles ) { ABORT_ON_CHECK_FAIL(); return false; }
      if( n_particles > 0 )
      {
        if( pb[cell_a].nbh_start[n_particles-1] != pb[cell_a].neighbors.size() ) { ABORT_ON_CHECK_FAIL(); return false; }
      }
      for(size_t p_a=0;p_a<n_particles;p_a++)
      {
          size_t neighbor_index = 0;
          if( p_a > 0 ) { neighbor_index = pb[cell_a].nbh_start[p_a-1]; }
          size_t neighbor_end = pb[cell_a].nbh_start[p_a];
          size_t last_cell_b = cell_a;
          size_t last_p_b = p_a;
          for(;neighbor_index<neighbor_end;neighbor_index++)
          {
            size_t cell_b=0, p_b=0;
            exanb::decode_cell_particle(pb[cell_a].neighbors[neighbor_index], cell_b, p_b);
            if( cell_b >= n_cells ) { ABORT_ON_CHECK_FAIL(); return false; }
            if( p_b >= grid.cell_number_of_particles(cell_b) ) { ABORT_ON_CHECK_FAIL(); return false; }
            if( cell_b < last_cell_b ) { ABORT_ON_CHECK_FAIL(); return false; }
            if( cell_b == last_cell_b )
            {
              if( p_b <= last_p_b ) { ABORT_ON_CHECK_FAIL(); return false; }
            }
            last_p_b = p_b;
            last_cell_b = cell_b;
          }
      }
    }
    return true;
  }


} // namespace exanb

