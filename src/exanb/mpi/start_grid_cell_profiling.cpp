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
#include <onika/scg/operator_slot.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <cmath>

namespace exanb
{

  // simple cost model where the cost of a cell is the number of particles in it
  // 
  template< class GridT >
  class StartGridCellProfiling : public OperatorNode
  {
    ADD_SLOT( GridT , grid , INPUT_OUTPUT , REQUIRED);
    
  public:

    inline void execute () override final
    {
      grid->set_cell_profiling( true );
      grid->reset_cell_profiling_data();
      //std::cout<<"cell profiling = "<<grid->cell_profiling();
    }

  };
  // === register factory ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory(
      "start_grid_cell_profiling",
      make_grid_variant_operator< StartGridCellProfiling > );
  }

}

