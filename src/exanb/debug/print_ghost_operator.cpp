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
#include <onika/log.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/fields.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/domain.h>
#include <exanb/core/print_particle.h>

#include <onika/soatl/field_tuple.h>

#include <vector>
#include <algorithm>
#include <sstream>

namespace exanb
{
  
  
  // =================== utility functions ==========================
  template<
    class GridT,
    class = AssertGridHasFields< GridT, field::_id>
    >
  class PrintGhosts : public OperatorNode
  {
    using ParticleIds = std::vector<uint64_t>;
    ADD_SLOT( GridT                    , grid                  , INPUT );

  public:
    inline void execute () override final
    {
      GridT& grid = *(this->grid);

      auto cells = grid.cells();
      size_t n_cells = grid.number_of_cells();

      lout << "DIMENSION : " << grid.dimension() << std::endl;
      lout << "*******************************GHOSTS****************************************" << std::endl;

#     pragma omp parallel
      {
#       pragma omp for
        for(size_t i=0;i<n_cells;i++)
        {
          if( grid.is_ghost_cell(i) )
            {
              size_t n_part = cells[i].size();
              for(size_t j=0;j<n_part;j++)
                {
#                 pragma omp critical
                  {
                    //lout << "offset : " << cell_pos.i << " " << cell_pos.j << " " << cell_pos.i << std::endl;
                    print_particle( lout , cells[i][j] );
                  }
                }
            }
        }
      }

      lout << "*******************************REAL PARTICLES****************************************" << std::endl;


#     pragma omp parallel
      {
#       pragma omp for
        for(size_t i=0;i<n_cells;i++)
          {
            //Check if cell is a ghost cell
            if(!grid.is_ghost_cell(i) )
              {
                size_t n_part = cells[i].size();
                for(size_t j=0;j<n_part;j++)
                  {
#                 pragma omp critical
                    {
                      //lout << "offset : " << cell_pos.i << " " << cell_pos.j << " " << cell_pos.i << std::endl;
                      print_particle( lout , cells[i][j] );
                    }
                  }
              }
          }
      }

    }

  };


  template<class GridT> using PrintGhostsTmpl = PrintGhosts<GridT>;
    

  // === register factories ===
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "debug_print_ghosts", make_grid_variant_operator< PrintGhostsTmpl > );
  }

}
