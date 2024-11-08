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
#include <exanb/core/grid.h>
#include <onika/math/basic_types.h>
#include <exanb/core/grid_algorithm.h>
#include <exanb/core/make_grid_variant_operator.h>

namespace exanb
{
  

  template<class GridT>
  class GridClearGhosts : public OperatorNode
  {  
    // -----------------------------------------------
    // -----------------------------------------------
    ADD_SLOT( GridT              , grid      , INPUT_OUTPUT );

  public:
    // -----------------------------------------------
    // -----------------------------------------------
    inline std::string documentation() const override final
    {
      return "Empties ghost cells";
    }

    // -----------------------------------------------
    // -----------------------------------------------
    inline void execute ()  override final
    {
      auto & g = *grid;
      apply_grid_shell( grid->dimension() , 0 , grid->ghost_layers() , [&g](ssize_t i, const IJK&){ g.cell(i).clear( g.cell_allocator() ); } );
#     ifndef NDEBUG
      const size_t n_cells = grid->number_of_cells();
      for(size_t i=0;i<n_cells;i++)
      {
        if( g.is_ghost_cell(i) ) { assert( g.cell(i).empty() && g.cell(i).capacity()==0 && g.cell(i).storage_ptr()==nullptr ); }
      }
#     endif
    }

  };
    
 // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "grid_clear_ghosts", make_grid_variant_operator< GridClearGhosts > );
  }

}

