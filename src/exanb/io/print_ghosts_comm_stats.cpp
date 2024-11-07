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
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>

#include <mpi.h>
#include <exanb/mpi/ghosts_comm_scheme.h>
#include <exanb/mpi/data_types.h>

namespace exanb
{
  

  template<class GridT>
  struct PrintGhostsCommStats : public OperatorNode
  {

    // -----------------------------------------------
    // Operator slots
    // -----------------------------------------------
    ADD_SLOT( MPI_Comm                 , mpi               , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( GridT                 , grid              , INPUT , REQUIRED       , DocString{"Particle grid"} );
    ADD_SLOT( GhostCommunicationScheme , ghost_comm_scheme , INPUT , REQUIRED       );

    inline void execute () override final
    {
      MPI_Comm comm = *mpi;

      int nprocs = 1;
      int rank = 0;
      MPI_Comm_size(comm,&nprocs);
      MPI_Comm_rank(comm,&rank);
      
      long long data_min[ 8 ]; // cells recv, cells send, particles recv, particles send, bytes recv, bytes send, ghost cells, nb partners
      long long data_max[ 8 ];
      long long data_avg[ 8 ];
      const char* labels[8] = { "Recv cells" , "Send cells" , "Recv parts" , "Send parts" , "Recv bytes" , "Send bytes" , "Ghost cells" , "Nb Partners" };
      
      for(int i=0;i<8;i++) data_avg[i] = 0;
      for(const auto & p :  ghost_comm_scheme->m_partner )
      {
        size_t send_cells = 0;
        size_t recv_cells = 0;
        size_t send_parts = 0;
        size_t recv_parts = 0;
        for( const auto & send : p.m_sends )
        {
          if( ! send.m_particle_i.empty() ) ++ send_cells;
          send_parts += send.m_particle_i.size();
        }
        for( const auto & recv : p.m_receives )
        {
          auto [ cell_i , n_particles ] = ghost_cell_receive_info( recv );
          if ( n_particles > 0 ) ++ recv_cells;
          recv_parts += n_particles;
        }
        if( (send_parts + recv_parts) > 0 ) ++ data_avg[7]; // number of effective partners
        data_avg[0] += recv_cells;
        data_avg[1] += send_cells;
        data_avg[2] += recv_parts;
        data_avg[3] += send_parts;
        data_avg[4] += ghost_comm_scheme->m_cell_bytes * recv_cells + ghost_comm_scheme->m_particle_bytes * recv_parts;
        data_avg[5] += ghost_comm_scheme->m_cell_bytes * send_cells + ghost_comm_scheme->m_particle_bytes * send_parts;
      }
      data_avg[6] = grid->number_of_ghost_cells();
      for(int i=0;i<8;i++) data_min[i]=data_max[i]=data_avg[i];
      MPI_Allreduce(MPI_IN_PLACE,data_avg,8,MPI_LONG_LONG,MPI_SUM,comm);
      MPI_Allreduce(MPI_IN_PLACE,data_min,8,MPI_LONG_LONG,MPI_MIN,comm);
      MPI_Allreduce(MPI_IN_PLACE,data_max,8,MPI_LONG_LONG,MPI_MAX,comm);
      
      lout << "=== Ghost communication stats ===" << std::endl;
      for(int i=0;i<8;i++)
      {
        data_avg[i] /= nprocs;
        lout << labels[i] << " : " << data_min[i] << " / " << data_avg[i] << " / " << data_max[i] << std::endl; 
      }
      lout << "=================================" << std::endl << std::endl;

    }
    // ------------------------

  };

  // === register factory ===
  CONSTRUCTOR_FUNCTION
  {
//    OperatorNodeFactory::instance()->register_factory( "print_ghost_comm_scheme", make_simple_operator< PrintGhostsCommScheme > );
    OperatorNodeFactory::instance()->register_factory( "print_ghost_comm_stats", make_grid_variant_operator< PrintGhostsCommStats > );
  }

}

