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
#pragma once

#include <exanb/compute/compute_cell_particles.h>
#include <exanb/defbox/xform_mode.h>
#include <onika/cuda/cuda.h>
#include <exanb/core/domain.h>

namespace exanb
{

  struct PushVec3SecondOrderFunctor
  {
    const double dt = 1.0;
    const double dt2 = 1.0;
    ONIKA_HOST_DEVICE_FUNC inline void operator () (double& x, double& y, double& z, double dx, double dy, double dz, double ddx, double ddy, double ddz) const
    {
      x += dx * dt + ddx * dt2;
      y += dy * dt + ddy * dt2;
      z += dz * dt + ddz * dt2;
    }
  };

  template<> struct ComputeCellParticlesTraits<PushVec3SecondOrderFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };

  struct PushVec3SecondOrderXFormFunctor
  {
    const Mat3d xform = { 1.0 , 0.0 , 0.0 ,
                          0.0 , 1.0 , 0.0 ,
                          0.0 , 0.0 , 1.0 };
    const double dt = 1.0;
    const double dt2 = 1.0;
    ONIKA_HOST_DEVICE_FUNC inline void operator () (double& x, double& y, double& z, double dx, double dy, double dz, double ddx, double ddy, double ddz) const
    {
      Vec3d d = xform * ( Vec3d{dx,dy,dz} * dt + Vec3d{ddx,ddy,ddz} * dt2 );
      x += d.x;
      y += d.y;
      z += d.z;
    }
  };

  template<> struct ComputeCellParticlesTraits<PushVec3SecondOrderXFormFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };


  template<
    class GridT,
    class Field_X, class Field_Y, class Field_Z,
    class Field_dX, class Field_dY, class Field_dZ,
    class Field_ddX, class Field_ddY, class Field_ddZ >
  struct PushVec3SecondOrderXForm : public OperatorNode
  {      
    /***************** Operator slots *********************/
    ADD_SLOT( GridT  , grid       , INPUT_OUTPUT);
    ADD_SLOT( double , dt         , INPUT , REQUIRED );
    ADD_SLOT( double , dt_scale   , INPUT , REQUIRED );
    ADD_SLOT( Domain , domain     , INPUT , REQUIRED );
    ADD_SLOT( XFormMode, xform_mode , INPUT , XFormMode::INV_XFORM );
    /******************************************************/

    inline void execute () override final
    {
      const double delta_t = (*dt) * (*dt_scale);
      const double delta_t2 = delta_t*delta_t*0.5;

      auto x = grid->field_accessor( onika::soatl::FieldId<Field_X> {} );
      auto y = grid->field_accessor( onika::soatl::FieldId<Field_Y> {} );
      auto z = grid->field_accessor( onika::soatl::FieldId<Field_Z> {} );

      auto dx = grid->field_accessor( onika::soatl::FieldId<Field_dX> {} );
      auto dy = grid->field_accessor( onika::soatl::FieldId<Field_dY> {} );
      auto dz = grid->field_accessor( onika::soatl::FieldId<Field_dZ> {} );

      auto ddx = grid->field_accessor( onika::soatl::FieldId<Field_ddX> {} );
      auto ddy = grid->field_accessor( onika::soatl::FieldId<Field_ddY> {} );
      auto ddz = grid->field_accessor( onika::soatl::FieldId<Field_ddZ> {} );

      auto compute_fields = onika::make_flat_tuple( x, y, z, dx, dy, dz, ddx, ddy, ddz );

      if( (*xform_mode) == XFormMode::IDENTITY || domain->xform_is_identity() )
      {
        ldbg<<"PushVec3SecondOrder: dt="<<(*dt)<<", dt_scale="<<(*dt_scale)<<", xform_mode="<< (*xform_mode)<<std::endl;
        PushVec3SecondOrderFunctor func { delta_t , delta_t2 };
        compute_cell_particles( *grid , false , func , compute_fields , parallel_execution_context() );
      }
      else
      {
        const Mat3d xform = (xform_mode->m_value==XFormMode::XFORM) ? domain->xform() : domain->inv_xform();
        ldbg<<"PushVec3SecondOrderXForm: dt="<<(*dt)<<", dt_scale="<<(*dt_scale)<<", xform_mode="<< (*xform_mode)<<", xform="<<xform<<std::endl;
        PushVec3SecondOrderXFormFunctor func { xform , delta_t , delta_t2 };
        compute_cell_particles( *grid , false , func , compute_fields , parallel_execution_context() );
      }
    }

  };

}

