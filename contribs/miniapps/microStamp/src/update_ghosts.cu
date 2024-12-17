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

#include <onika/log.h>
#include <onika/math/basic_types.h>

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>

#include <exanb/mpi/update_ghosts.h>

namespace microStamp
{
  using namespace exanb;
  using namespace UpdateGhostsUtils;

  // === register factory ===
  template<typename GridT> using UpdateGhostsAllFieldsNoFV = UpdateGhostsNode< GridT , RemoveFields< typename GridT::Fields , FieldSet<field::_fx,field::_fy,field::_fz,field::_vx, field::_vy, field::_vz > > , true >;

  ONIKA_AUTORUN_INIT(ghost_update_all_no_fv)
  {
    OperatorNodeFactory::instance()->register_factory( "ghost_update_all_no_fv", make_grid_variant_operator<UpdateGhostsAllFieldsNoFV> );
  }

}

