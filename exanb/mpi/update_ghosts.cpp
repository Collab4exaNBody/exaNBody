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
//#pragma xstamp_cuda_enable // DO NOT REMOVE THIS LINE

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <onika/math/basic_types_stream.h>
#include <exanb/core/grid.h>
#include <exanb/grid_cell_particles/grid_cell_values.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/particle_id_codec.h>
#include <exanb/field_sets.h>
#include <exanb/core/check_particles_inside_cell.h>

#include <onika/soatl/field_tuple.h>

#include <vector>
#include <string>
#include <list>
#include <algorithm>
#include <tuple>

#include <mpi.h>
#include <exanb/mpi/update_ghost_utils.h>
#include <exanb/mpi/ghosts_comm_scheme.h>
#include <exanb/mpi/update_ghosts.h>
#include <onika/mpi/data_types.h>

namespace exanb
{
  
  using namespace UpdateGhostsUtils;

  // === register factory ===
  namespace ghost_update
  {
    template<typename GridT> using UpdateGhostsAllFields = UpdateGhostsNode< GridT , typename GridT::Fields , true >;
    template<typename GridT> using UpdateGhostsR = UpdateGhostsNode< GridT , FieldSet<field::_rx, field::_ry, field::_rz> , false >;

    ONIKA_AUTORUN_INIT(update_ghosts)
    {
      OperatorNodeFactory::instance()->register_factory( "ghost_update_all",    make_grid_variant_operator<UpdateGhostsAllFields> );
      OperatorNodeFactory::instance()->register_factory( "ghost_update_r",      make_grid_variant_operator<UpdateGhostsR> );
    }
  }

}

