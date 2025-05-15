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
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <onika/log.h>
#include <exanb/io/grid_memory_compact.h>

#include <onika/memory/allocator.h>

#include <omp.h>

#include <iostream>
#include <fstream>
//#include <filesystem>
//namespace fs = std::filesystem;

namespace exanb
{

  template<class GridT>
  struct GridMemoryCompact : public OperatorNode
  {
    ADD_SLOT(GridT , grid , INPUT_OUTPUT , DocString{"Particle grid"} );
    ADD_SLOT(bool  , enable_grid_compact , INPUT , true , DocString{"Enable array shrinking of particle arrays"} );
    ADD_SLOT(bool  , force_realloc       , INPUT , false , DocString{"Force reallocation using grid's cell allocator, even if capacity is unchanged"} );
    ADD_SLOT(bool  , ghost               , INPUT , true );

    inline void execute () override final
    {
      exanb::grid_memory_compact(*grid, *enable_grid_compact, *force_realloc, *ghost);
    }

  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(grid_memory_compact)
  {
    OperatorNodeFactory::instance()->register_factory( "grid_memory_compact", make_grid_variant_operator<GridMemoryCompact> );
  }

}

