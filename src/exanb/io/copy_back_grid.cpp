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
#include <onika/math/basic_types_yaml.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>

namespace exanb
{

  template<typename GridT>
  struct CopyBackGrid : public OperatorNode
  {
    ADD_SLOT(GridT       , grid       , INPUT_OUTPUT , REQUIRED );
    ADD_SLOT(GridT       , grid_copy  , INPUT_OUTPUT , REQUIRED );
    ADD_SLOT(bool        , clear_copy , INPUT , false );

    inline void execute () override final
    {
      *grid = *grid_copy;
      if ( *clear_copy ) { grid_copy->reset(); }
    }

  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(copy_back_grid)
  {
    OperatorNodeFactory::instance()->register_factory( "copy_back_grid", make_grid_variant_operator< CopyBackGrid > );
  }

}
