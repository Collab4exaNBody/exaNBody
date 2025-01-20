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

#include <onika/log.h>
#include <onika/math/basic_types.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>

#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/make_grid_variant_operator.h>

#include <mpi.h>
#include <cstring>

#include "simulation_state.h"

namespace microStamp
{
  using namespace exanb;

  template<
    class GridT ,
    class = AssertGridHasFields< GridT, field::_vx, field::_vy, field::_vz >
    >
  struct ComputeSimulationState : public OperatorNode
  {
    ADD_SLOT( MPI_Comm           , mpi                 , INPUT , MPI_COMM_WORLD);
    ADD_SLOT( GridT              , grid                , INPUT , REQUIRED);
    ADD_SLOT( Domain             , domain              , INPUT , REQUIRED);
    ADD_SLOT( SimulationState    , simulation_state    , OUTPUT );

    inline void execute () override final
    {
      MPI_Comm comm = *mpi;
      GridT& grid = *(this->grid);

      auto cells = grid.cells();
      IJK dims = grid.dimension();
      size_t ghost_layers = grid.ghost_layers();
      IJK dims_no_ghost = dims - (2*ghost_layers);

      double kinetic_energy = 0.0;  // constructs itself with 0s
      double potential_energy = 0.;
      unsigned long long total_particles = 0;
      
#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN(dims_no_ghost,_,loc_no_ghosts, reduction(+:kinetic_energy,total_particles) )
        {
          IJK loc = loc_no_ghosts + ghost_layers;
          size_t cell_i = grid_ijk_to_index(dims,loc);

          const double* __restrict__ vx = cells[cell_i][field::vx];
          const double* __restrict__ vy = cells[cell_i][field::vy];
          const double* __restrict__ vz = cells[cell_i][field::vz];

          double local_kinetic_ernergy = 0.;
          size_t n = cells[cell_i].size();

#         pragma omp simd reduction(+:local_kinetic_ernergy)
          for(size_t j=0;j<n;j++)
          {
            const double mass = 1.0;
            Vec3d v { vx[j], vy[j], vz[j] };
            local_kinetic_ernergy += dot(v,v) * mass;
          }
          kinetic_energy += local_kinetic_ernergy;
          total_particles += n;
        }
        GRID_OMP_FOR_END
      }

      MPI_Allreduce(MPI_IN_PLACE, &kinetic_energy, 1, MPI_DOUBLE, MPI_SUM, comm);
      MPI_Allreduce(MPI_IN_PLACE, &total_particles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
      
      simulation_state->m_kinetic_energy = kinetic_energy;
      simulation_state->m_particle_count = total_particles;
    }
  };
    
  template<class GridT> using ComputeSimulationStateTmpl = ComputeSimulationState<GridT>;
    
  // === register factories ===  
  ONIKA_AUTORUN_INIT(simulation_state)
  {
   OperatorNodeFactory::instance()->register_factory( "simulation_state", make_grid_variant_operator< ComputeSimulationStateTmpl > );
  }

}

