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
#include <onika/memory/allocator.h>

#include <exanb/compute/compute_cell_particles.h>
#include <exanb/defbox/push_vec3_1st_order.h>
#include <exanb/defbox/push_vec3_1st_order_xform.h>
#include <exanb/core/domain.h>

#include <exanb/defbox/xform_mode.h>

namespace exanb
{

  using namespace onika;

  template<
    class GridT,
    class Field_X, class Field_Y, class Field_Z,
    class Field_dX, class Field_dY, class Field_dZ >
  struct PushVec3FirstOrder : public OperatorNode
  {  
    ADD_SLOT( GridT , grid ,INPUT_OUTPUT);
    ADD_SLOT( double , dt ,INPUT);
    ADD_SLOT( double , dt_scale ,INPUT , 1.0 );
    ADD_SLOT( Domain , domain     , INPUT , REQUIRED );
    ADD_SLOT( XFormMode, xform_mode , INPUT , XFormMode::IDENTITY );

    inline void execute () override final
    {
      const double delta_t = (*dt) * (*dt_scale);

      auto x = grid->field_accessor( onika::soatl::FieldId<Field_X> {} );
      auto y = grid->field_accessor( onika::soatl::FieldId<Field_Y> {} );
      auto z = grid->field_accessor( onika::soatl::FieldId<Field_Z> {} );

      auto dx = grid->field_accessor( onika::soatl::FieldId<Field_dX> {} );
      auto dy = grid->field_accessor( onika::soatl::FieldId<Field_dY> {} );
      auto dz = grid->field_accessor( onika::soatl::FieldId<Field_dZ> {} );

      auto compute_fields = onika::make_flat_tuple( x, y, z, dx, dy, dz );

      if( (*xform_mode) == XFormMode::IDENTITY || domain->xform_is_identity() )
      {
        ldbg<<"PushVec3FirstOrder: dt="<<(*dt)<<", dt_scale="<<(*dt_scale)<<", xform_mode="<< (*xform_mode) <<std::endl;
        PushVec3FirstOrderFunctor func { delta_t };
        compute_cell_particles( *grid , false , func , compute_fields , parallel_execution_context() );
      }
      else
      {
        const Mat3d xform = (xform_mode->m_value==XFormMode::XFORM) ? domain->xform() : domain->inv_xform();
        ldbg<<"PushVec3FirstOrderXForm: dt="<<(*dt)<<", dt_scale="<<(*dt_scale)<<", xform_mode="<< (*xform_mode)<<", xform="<<xform<<std::endl;
        PushVec3FirstOrderXFormFunctor func { xform , delta_t };
        compute_cell_particles( *grid , false , func , compute_fields , parallel_execution_context() );
      }
    }

  };

  template<class GridT> using PushVelocityToPosition = PushVec3FirstOrder<GridT, field::_rx,field::_ry,field::_rz, field::_vx,field::_vy,field::_vz >;
  template<class GridT> using PushForceToVelocity = PushVec3FirstOrder<GridT, field::_vx,field::_vy,field::_vz, field::_fx,field::_fy,field::_fz >;
  template<class GridT> using PushForceToPosition = PushVec3FirstOrder<GridT, field::_rx,field::_ry,field::_rz, field::_fx,field::_fy,field::_fz >;

  template<class GridT> using PushForceToVelocityFlat = PushVec3FirstOrder<GridT, field::_vx,field::_vy,field::_vz, field::_flat_fx,field::_flat_fy,field::_flat_fz >;
  template<class GridT> using PushForceToPositionFlat = PushVec3FirstOrder<GridT, field::_rx,field::_ry,field::_rz, field::_flat_fx,field::_flat_fy,field::_flat_fz >;
  
 // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "push_v_r", make_grid_variant_operator< PushVelocityToPosition > );
   OperatorNodeFactory::instance()->register_factory( "push_f_v", make_grid_variant_operator< PushForceToVelocity > );
   OperatorNodeFactory::instance()->register_factory( "push_f_r", make_grid_variant_operator< PushForceToPosition > );
   OperatorNodeFactory::instance()->register_factory( "push_flat_f_v", make_grid_variant_operator< PushForceToVelocityFlat > );
   OperatorNodeFactory::instance()->register_factory( "push_flat_f_r", make_grid_variant_operator< PushForceToPositionFlat > );
  }

}

