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
#pragma xstamp_cuda_enable

#pragma xstamp_grid_variant

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>

#include <exanb/defbox/push_vec3_1st_order.h>

namespace exanb
{
  template<class GridT> using PushVelocityToPosition = PushVec3FirstOrder<GridT, field::_rx,field::_ry,field::_rz, field::_vx,field::_vy,field::_vz >;
  template<class GridT> using PushForceToVelocity = PushVec3FirstOrder<GridT, field::_vx,field::_vy,field::_vz, field::_fx,field::_fy,field::_fz >;
  template<class GridT> using PushForceToPosition = PushVec3FirstOrder<GridT, field::_rx,field::_ry,field::_rz, field::_fx,field::_fy,field::_fz >;
  
 // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "push_v_r", make_grid_variant_operator< PushVelocityToPosition > );
   OperatorNodeFactory::instance()->register_factory( "push_f_v", make_grid_variant_operator< PushForceToVelocity > );
   OperatorNodeFactory::instance()->register_factory( "push_f_r", make_grid_variant_operator< PushForceToPosition > );
  }

}

