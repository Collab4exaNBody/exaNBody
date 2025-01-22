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

#include <exanb/compute/simulation_statistics.h>

namespace exanb
{

  template<
    class GridT ,
    class = AssertGridHasFields< GridT, field::_vx, field::_vy, field::_vz >
    >
  struct ComputeSimulationStatistics : public OperatorNode
  {
    ADD_SLOT( MPI_Comm           , mpi                 , INPUT , MPI_COMM_WORLD);
    ADD_SLOT( GridT              , grid                , INPUT , REQUIRED);
    ADD_SLOT( Domain             , domain              , INPUT , REQUIRED);
    ADD_SLOT( SimulationStatistics , simulation_stats   , OUTPUT );

    inline void execute () override final
    {
      MPI_Comm comm = *mpi;
      GridT& grid = *(this->grid);

      auto cells = grid.cells();
      IJK dims = grid.dimension();
      size_t ghost_layers = grid.ghost_layers();
      IJK dims_no_ghost = dims - (2*ghost_layers);

      double kinetic_energy = 0.0;  // constructs itself with 0s
      unsigned long long total_particles = 0;
      double min_vel = std::numeric_limits<double>::max();
      double min_acc = std::numeric_limits<double>::max();
      double max_vel = 0.0;
      double max_acc = 0.0;
      
#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN(dims_no_ghost,_,loc_no_ghosts, reduction(+:kinetic_energy,total_particles)  reduction(min:min_vel,min_acc) reduction(max:max_vel,max_acc) )
        {
          IJK loc = loc_no_ghosts + ghost_layers;
          size_t cell_i = grid_ijk_to_index(dims,loc);

          const auto* __restrict__ vx = cells[cell_i][field::vx];
          const auto* __restrict__ vy = cells[cell_i][field::vy];
          const auto* __restrict__ vz = cells[cell_i][field::vz];

          const auto* __restrict__ ax = cells[cell_i][field::ax];
          const auto* __restrict__ ay = cells[cell_i][field::ay];
          const auto* __restrict__ az = cells[cell_i][field::az];

          double local_kinetic_ernergy = 0.;
          double local_min_vel = std::numeric_limits<double>::max();
          double local_min_acc = std::numeric_limits<double>::max();
          double local_max_vel = 0.0;
          double local_max_acc = 0.0;
          size_t n = cells[cell_i].size();

#         pragma omp simd reduction(+:local_kinetic_ernergy) reduction(min:local_min_vel,local_min_acc) reduction(max:local_max_vel,local_max_acc)
          for(size_t j=0;j<n;j++)
          {
            const double mass = 1.0;
            Vec3d v { vx[j], vy[j], vz[j] };
            local_kinetic_ernergy += dot(v,v) * mass;
            const double v_norm = norm(v);
            local_min_vel = std::min( local_min_vel , v_norm );
            local_max_vel = std::max( local_max_vel , v_norm );
            const double a_norm = norm( Vec3d{ax[j], ay[j], az[j]} );
            local_min_acc = std::min( local_min_acc , a_norm );
            local_max_acc = std::max( local_max_acc , a_norm );
          }
          kinetic_energy += local_kinetic_ernergy;
          total_particles += n;
          min_vel = std::min( min_vel , local_min_vel );
          max_vel = std::max( max_vel , local_max_vel );
          min_acc = std::min( min_acc , local_min_acc );
          max_acc = std::max( max_acc , local_max_acc );
        }
        GRID_OMP_FOR_END
      }

      MPI_Allreduce(MPI_IN_PLACE, &min_vel, 1, MPI_DOUBLE, MPI_MIN, comm);
      MPI_Allreduce(MPI_IN_PLACE, &max_vel, 1, MPI_DOUBLE, MPI_MAX, comm);
      MPI_Allreduce(MPI_IN_PLACE, &min_acc, 1, MPI_DOUBLE, MPI_MIN, comm);
      MPI_Allreduce(MPI_IN_PLACE, &max_acc, 1, MPI_DOUBLE, MPI_MAX, comm);
      MPI_Allreduce(MPI_IN_PLACE, &kinetic_energy, 1, MPI_DOUBLE, MPI_SUM, comm);
      MPI_Allreduce(MPI_IN_PLACE, &total_particles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
      
      simulation_stats->m_kinetic_energy = kinetic_energy;
      simulation_stats->m_particle_count = total_particles;
      simulation_stats->m_min_vel = min_vel;
      simulation_stats->m_max_vel = max_vel;
      simulation_stats->m_min_acc = min_acc;
      simulation_stats->m_max_acc = max_acc;
      
    }
  };
    
  template<class GridT> using ComputeSimulationStatisticsTmpl = ComputeSimulationStatistics<GridT>;
    
  // === register factories ===  
  ONIKA_AUTORUN_INIT(simulation_stats)
  {
   OperatorNodeFactory::instance()->register_factory( "simulation_stats", make_grid_variant_operator< ComputeSimulationStatisticsTmpl > );
  }

}

