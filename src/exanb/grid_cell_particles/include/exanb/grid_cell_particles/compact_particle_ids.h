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
#include <exanb/core/grid.h>
#include <exanb/core/parallel_grid_algorithm.h>

#include <exanb/core/basic_types_yaml.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/log.h>
#include <exanb/core/particle_type_id.h>

#include <mpi.h>

namespace exanb
{
  // Generate Orthorhombic lattice, i.e. a, b, c may have different lengths but alpha, beta and gamma equal 90 degrees.
  // Requirements :
  // structure : SC, BCC, FCC, HCP, DIAMOND, ROCKSALT, FLUORITE, C15, PEROVSKITE, ST, BCT, FCT, 2BCT
  // types : number of types depends on the structure and types must be consistent with the species defined beforehand
  // size : vector containing a, b, c lengths of the lattice
  // repeats : number of repetition along three laboratory directions

  template< class GridT
          , class _ParticleIdField
          , class = AssertGridHasFields< GridT, _ParticleIdField > >
  static inline void compact_particle_ids( MPI_Comm comm, GridT& grid , onika::soatl::FieldId<_ParticleIdField> fid )
  {
      auto cells = grid.cells();
      IJK dims = grid.dimension();
      ssize_t gl = grid.ghost_layers();
      IJK gstart { gl, gl, gl };
      IJK gend = dims - IJK{ gl, gl, gl };
      IJK gdims = gend - gstart;

      std::vector<size_t> cell_id_ofset( grid.number_of_cells() , 0 );
      unsigned long long local_n_particles = 0;

#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN(gdims,_,loc, schedule(dynamic) reduction(+:local_n_particles) )
        {
          size_t i = grid_ijk_to_index( dims , loc + gstart );
          size_t n = cells[i].size();
          cell_id_ofset[i] = n
          local_n_particles += n;
        }
        GRID_OMP_FOR_END
      }
      
      std::exclusive_scan( cell_id_ofset.begin() , cell_id_ofset.end() , cell_id_ofset.begin() , 0 );

      unsigned long long particle_id_start = 0;
      MPI_Exscan( &local_n_particles , &particle_id_start , 1 , MPI_UNSIGNED_LONG_LONG , MPI_SUM , comm);
      
#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN(gdims,_,loc, schedule(dynamic) )
        {
          size_t i = grid_ijk_to_index( dims , loc + gstart );
          size_t n = cells[i].size();
          for(size_t j=0;i<n;j++) cells[i][fid][j] = particle_id_start + cell_id_ofset[i] + j;
        }
        GRID_OMP_FOR_END
      }
      
    }
  }

}
