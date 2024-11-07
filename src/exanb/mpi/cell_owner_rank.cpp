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
#include <exanb/core/domain.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>

#include <mpi.h>
#include <vector>

namespace exanb
{
  

  using CellOwner  = std::vector<int>;

  // simple cost model where the cost of a cell is the number of particles in it
  // 
  template<class GridT>
  struct CellOwnerRank : public OperatorNode
  {
    ADD_SLOT( MPI_Comm  , mpi         , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( Domain    , domain      , INPUT , REQUIRED );
    ADD_SLOT( GridT     , grid        , INPUT , REQUIRED );
    ADD_SLOT( GridBlock , lb_block    , INPUT , REQUIRED );
    ADD_SLOT( CellOwner , cell_owner  , INPUT_OUTPUT );

    inline void execute () override final
    {
      MPI_Comm comm = *mpi;
      int np=1, rank=0;
      MPI_Comm_size(comm,&np);
      MPI_Comm_rank(comm,&rank);

      IJK dom_dims = domain->grid_dimension();
      size_t n_cells = grid_cell_count( dom_dims );      
      IJK block_dims = dimension(*lb_block);

      std::vector<int> all_owner(n_cells,-1);

      GRID_FOR_BEGIN( block_dims , _ , loc )
      {
        size_t i = grid_ijk_to_index( dom_dims , loc + lb_block->start );
        all_owner[i] = rank;
      }
      GRID_FOR_END
      
      MPI_Allreduce( MPI_IN_PLACE, all_owner.data() , all_owner.size(), MPI_INT, MPI_MAX, comm );

      IJK grid_dims = grid->dimension();
      n_cells = grid->number_of_cells();
      cell_owner->clear();
      cell_owner->resize( n_cells , -1 );

      GRID_FOR_BEGIN( grid_dims , grid_i , loc )
      {
        IJK dom_loc = periodic_location( dom_dims , loc + grid->offset() );
        size_t dom_i = grid_ijk_to_index( dom_dims , dom_loc );
        cell_owner->at(grid_i) = all_owner[dom_i];
//#       ifndef NDEBUG        
//        ldbg << dom_loc << " : owner = "<<all_owner[dom_i] << std::endl << std::flush ;
//#       endif
      }
      GRID_FOR_END
    }

  };

  // === register factory ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "cell_owner_rank", make_grid_variant_operator< CellOwnerRank > );
  }

}

