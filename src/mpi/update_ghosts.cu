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

#include <mpi.h>

#include <onika/mpi/data_types.h>
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

#include <exanb/mpi/ghosts_comm_scheme.h>
#include <exanb/mpi/update_ghosts.h>

namespace exanb
{
  
  using namespace UpdateGhostsUtils;

  // === register factory ===
  template<typename GridT> using UpdateGhostsAllFields = UpdateGhostsNode< GridT , AddDefaultFields< typename GridT::Fields > , true >;
  template<typename GridT> using UpdateGhostsR = UpdateGhostsNode< GridT , FieldSet<field::_rx, field::_ry, field::_rz> , false >;
  template<typename GridT> using UpdateGhostsAllFieldsNoFV = UpdateGhostsNode< GridT , AddDefaultFields< RemoveFields< typename GridT::Fields , FieldSet<field::_fx,field::_fy,field::_fz,field::_vx, field::_vy, field::_vz > > > , true >;
  template<typename GridT> using UpdateGhostsOptOnly = UpdateGhostsNode< GridT , FieldSet<> , false >;

  ONIKA_AUTORUN_INIT(update_ghosts)
  {
    OperatorNodeFactory::instance()->register_factory( "ghost_update_all",       make_grid_variant_operator<UpdateGhostsAllFields> );
    OperatorNodeFactory::instance()->register_factory( "ghost_update_all_no_fv", make_grid_variant_operator<UpdateGhostsAllFieldsNoFV> );
    OperatorNodeFactory::instance()->register_factory( "ghost_update_r",         make_grid_variant_operator<UpdateGhostsR> );
    OperatorNodeFactory::instance()->register_factory( "ghost_update_opt",       make_grid_variant_operator<UpdateGhostsOptOnly> );
  }

}

