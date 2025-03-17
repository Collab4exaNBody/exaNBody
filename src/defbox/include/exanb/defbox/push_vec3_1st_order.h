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
#include <exanb/core/domain.h>
#include <exanb/defbox/xform_mode.h>
#include <onika/cuda/cuda.h>

namespace exanb
{

  struct PushVec3FirstOrderFunctor
  {
    double dt = 1.0;
    ONIKA_HOST_DEVICE_FUNC inline void operator () (double& x, double& y, double& z, double dx, double dy, double dz) const
    {
      x += dx * dt;
      y += dy * dt;
      z += dz * dt;
    }
  };

  template<> struct ComputeCellParticlesTraits<PushVec3FirstOrderFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };

  struct PushVec3FirstOrderXFormFunctor
  {
    const Mat3d xform = { 1.0 , 0.0 , 0.0 ,
                          0.0 , 1.0 , 0.0 ,
                          0.0 , 0.0 , 1.0 };
    const double dt = 1.0;
    ONIKA_HOST_DEVICE_FUNC inline void operator () (double& x, double& y, double& z, double dx, double dy, double dz) const
    {
      const Vec3d d = xform * Vec3d{dx,dy,dz} ;
      x += d.x * dt;
      y += d.y * dt;
      z += d.z * dt;
    }
  };

  template<> struct ComputeCellParticlesTraits<PushVec3FirstOrderXFormFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };

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

}

