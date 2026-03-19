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
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <onika/log.h>
#include <onika/math/basic_types.h>
#include <onika/math/basic_types_stream.h>
#include <exanb/core/grid.h>

#include <iostream>
#include <string>

namespace exanb
{

  template<typename GridT>
  struct PrintGrid : public OperatorNode
  {
    ADD_SLOT( GridT , grid , INPUT );

    inline void execute() override final
    {

      const char* sep="";

      lout << std::defaultfloat
           << "======== Simulation Grid ========"<< std::endl
           << "origin          = " << grid->origin() << std::endl
           << "cell size       = " << grid->cell_size() << std::endl
           << "dims            = " << grid->dimension() << std::endl
           << "dims no ghost   = " << grid->dimension_no_ghost() << std::endl
           << "offset          = " << grid->offset() << std::endl
           << "ghost layers    = " << grid->ghost_layers() << std::endl
           << "bounds          = " << grid->grid_bounds() << std::endl
           << "bounds no ghost = " << grid->grid_bounds_no_ghost() << std::endl;
      lout << "================================="<< std::endl << std::endl;
    }

  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(print_grid)
  {
    OperatorNodeFactory::instance()->register_factory( "print_grid", make_grid_variant_operator< PrintGrid > );
  }

}

