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
#include <onika/scg/operator.h>
#include <exanb/core/domain.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <onika/math/basic_types_stream.h>
#include <onika/log.h>
#include <exanb/core/simple_block_rcb.h>

#include <vector>
#include <algorithm>
#include <limits>
#include <mpi.h>

namespace exanb
{

  template<typename GridT>
  struct InitRCBGrid : public OperatorNode
  { 
    using ParticleT = typename GridT::CellParticles::TupleValueType;
    using ParticleVector = std::vector<ParticleT>;

    ADD_SLOT( MPI_Comm       , mpi    , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( GridT          , grid   , INPUT_OUTPUT ); // we'll modify the grid offset in the new domain
    ADD_SLOT( Domain         , domain , INPUT_OUTPUT );

    inline void execute () override final
    {
      // MPI Initialization
      int rank=0, np=1;
      MPI_Comm_rank(*mpi, &rank);
      MPI_Comm_size(*mpi, &np);

      ldbg<<"Domain = "<< *domain << std::endl;

      if( ! check_domain( *domain ) )
      {
        fatal_error() << "Invalid domain configuration" << std::endl;
      }
      if( grid->number_of_cells() > 0 )
      {
        fatal_error() << "Grid is not empty" << std::endl;        
      }

      // compute local processor's grid size and location so that cells are evenly distributed
      GridBlock in_block = { IJK{0,0,0} , domain->grid_dimension() };
      ldbg<<"In  block = "<< in_block << std::endl;
      GridBlock out_block = simple_block_rcb( in_block, np, rank );
      ldbg<<"Out block = "<< out_block << std::endl;

      // initializes local processor's grid
      grid->reset();
      grid->set_offset( out_block.start );
      grid->set_origin( domain->bounds().bmin );
      grid->set_cell_size( domain->cell_size() );
      IJK local_grid_dim = out_block.end - out_block.start;
      grid->set_dimension( local_grid_dim );
      grid->rebuild_particle_offsets();
    }
  };
  
   // === register factories ===  
  ONIKA_AUTORUN_INIT(initialize_rcb_grid)
  {
    OperatorNodeFactory::instance()->register_factory("init_rcb_grid", make_grid_variant_operator< InitRCBGrid > );
  }

}

