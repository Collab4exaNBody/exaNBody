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
#include <exanb/particle_neighbors/parallel_build_dual_neighbors.h>

#include <exanb/core/particle_id_codec.h>

#include <cassert>
#include <cstdlib>

#ifndef NDEBUG
#define ABORT_ON_CHECK_FAIL() abort()
#else
#define ABORT_ON_CHECK_FAIL() (void)0
#endif

namespace exanb
{

  bool check_dual_neighbors(GridParticleNeighbors& db)
  {
    size_t n_cells = db.size();
    for(size_t cell_a=0;cell_a<n_cells;cell_a++)
    {
      size_t n_particles = db[cell_a].nbh_start.size();
      for(size_t p_a=0;p_a<n_particles;p_a++)
      {
          size_t bond_index = 0;
          if( p_a > 0 ) { bond_index = db[cell_a].nbh_start[p_a-1]; }
          size_t bond_end = db[cell_a].nbh_start[p_a];
          ssize_t last_cell_b = -1;
          ssize_t last_p_b = -1;
          for(;bond_index<bond_end;bond_index++)
          {
            size_t cell_b=0, p_b=0;
            exanb::decode_cell_particle(db[cell_a].neighbors[bond_index], cell_b, p_b );
            
            // in dual pairs (or neighbors) (a,b) we expect b < a, b is 'before' a in the sens of lexicographic order (cell,particle) 
            if( cell_b > cell_a ) { ABORT_ON_CHECK_FAIL(); return false; }
            if( cell_b == cell_a && p_b >= p_a ) { ABORT_ON_CHECK_FAIL(); return false; }
            
            // for i in [0;N[, N being the number of neibors of particle A, we expect that for successive pairs (A,B_i-1) (A,B_i), B_i-1 < B_i
            if( static_cast<ssize_t>(cell_b) < last_cell_b ) { ABORT_ON_CHECK_FAIL(); return false; }
            if( static_cast<ssize_t>(cell_b) == last_cell_b )
            {
              if( static_cast<ssize_t>(p_b) <= last_p_b ) { ABORT_ON_CHECK_FAIL(); return false; }
            }
            last_p_b = p_b;
            last_cell_b = cell_b;
          }
      }
    }
    return true;
  }

}

