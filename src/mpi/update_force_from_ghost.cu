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
#include <onika/log.h>
#include <onika/math/basic_types_stream.h>
#include <onika/soatl/field_tuple.h>

#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/particle_id_codec.h>
#include <exanb/core/check_particles_inside_cell.h>

#include <exanb/grid_cell_particles/grid_cell_values.h>

#include <mpi.h>
#include <exanb/mpi/update_from_ghosts.h>
#include <onika/mpi/data_types.h>

namespace exanb
{
  
  using namespace UpdateGhostsUtils;

  // === register factory ===
  template<typename GridT> using UpdateForceFromGhosts = UpdateFromGhosts< GridT , FieldSet<field::_fx,field::_fy,field::_fz> , UpdateValueAdd >;
  template<typename GridT> using UpdateOptFromGhosts = UpdateFromGhosts< GridT , FieldSet<> , UpdateValueAdd >;

  ONIKA_AUTORUN_INIT(update_force_from_ghost)
  {
    OperatorNodeFactory::instance()->register_factory( "update_force_from_ghost", make_grid_variant_operator<UpdateForceFromGhosts> );
    OperatorNodeFactory::instance()->register_factory( "update_opt_from_ghost", make_grid_variant_operator<UpdateForceFromGhosts> );
  }

}

