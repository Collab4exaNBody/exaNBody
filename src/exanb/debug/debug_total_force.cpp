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
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <onika/log.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <onika/string_utils.h>
#include <onika/value_streamer.h>

#include <exanb/particle_neighbors/grid_particle_neighbors.h>

#include <memory>
#include <iomanip>
#include <mpi.h>

namespace exanb
{
  
  
  // =====================================================================
  // ========================== TestWeight ========================
  // =====================================================================

  template<
    class GridT,
    class = AssertGridHasFields< GridT, field::_fx , field::_fy , field::_fz >
    >
  struct DebugTotalForce : public OperatorNode
  {
    ADD_SLOT( MPI_Comm  , mpi      , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( GridT           , grid    , INPUT );
    ADD_SLOT( bool            , ghost   , INPUT , false );

    void execute() override final
    {
      int nprocs = 1;
      int rank = 0;
      MPI_Comm_size(*mpi,&nprocs);
      MPI_Comm_rank(*mpi,&rank);

      auto cells = grid->cells();
      size_t n_cells = grid->number_of_cells();

      Vec3d force_sum = {0,0,0};
      double force_norm_sum = 0.;

      for(size_t cell_a=0;cell_a<n_cells;cell_a++)
      {
        if( !grid->is_ghost_cell(cell_a) || *ghost )
        {
          size_t n_particles = cells[cell_a].size();
          auto * __restrict__ fx = cells[cell_a][field::fx];
          auto * __restrict__ fy = cells[cell_a][field::fy];
          auto * __restrict__ fz = cells[cell_a][field::fz];
          for(size_t p_a=0;p_a<n_particles;p_a++)
          {
            Vec3d F = { fx[p_a] , fy[p_a] , fz[p_a] };
            force_sum += F;
            force_norm_sum += norm(F);
          }
        }
      }
      
      Vec3d all_force_sum = {0.,0.,0.};
      double all_force_norm_sum = 0.;      
      {
        double tmp[4];
        onika::ValueStreamer<double>(tmp) << force_sum.x << force_sum.y << force_sum.z << force_norm_sum;
        MPI_Allreduce(MPI_IN_PLACE,tmp,4,MPI_DOUBLE,MPI_SUM,*mpi);
        onika::ValueStreamer<double>(tmp) >> all_force_sum.x >> all_force_sum.y >> all_force_sum.z >> all_force_norm_sum;
      }
#     ifndef NDEBUG
      std::cout << std::setprecision(3) << "P "<<rank<<" / "<<nprocs<<" : sum(F)="<<force_sum<<" , sum(Fnorm)="<<force_norm_sum <<std::endl<<std::flush;
#     endif
      lout << onika::format_string("sum of forces = % .3e , % .3e , % .3e (% .3e) , sum of norms = % .3e",all_force_sum.x , all_force_sum.y,all_force_sum.z,norm(all_force_sum),all_force_norm_sum) << std::endl;
    }
  };

  template<class GridT> using DebugTotalForceTmpl = DebugTotalForce<GridT>;

  // === register factories ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "debug_total_force", make_grid_variant_operator< DebugTotalForceTmpl > );
  }

}
