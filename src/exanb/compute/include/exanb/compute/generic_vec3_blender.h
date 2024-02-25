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

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/compute/compute_cell_particles.h>

namespace exanb
{

  struct GenericVec3BlendFunctor
  {
    Vec3d m_scale_src = { 0.0 , 0.0 , 0.0 };
    Vec3d m_scale_dst = { 1.0 , 1.0 , 1.0 };
    Vec3d m_add_dst = { 0.0 , 0.0 , 0.0 };
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( double& dx, double& dy, double& dz, double sx, double sy, double sz ) const
    {
      dx = dx * m_scale_dst.x + sx * m_scale_src.x + m_add_dst.x;
      dy = dy * m_scale_dst.y + sy * m_scale_src.y + m_add_dst.y;
      dz = dz * m_scale_dst.z + sz * m_scale_src.z + m_add_dst.z;
    }
  };

  template<> struct ComputeCellParticlesTraits< GenericVec3BlendFunctor >
  {
    static inline constexpr bool CudaCompatible = true;
  };

  template<
    class GridT,
    class Field_SX, class Field_SY, class Field_SZ,
    class Field_DX, class Field_DY, class Field_DZ,
    class = AssertGridHasFields< GridT, Field_DX, Field_DY, Field_DZ, Field_SX, Field_SY, Field_SZ >
    >
  class GenericVec3Blender : public OperatorNode
  {  
    ADD_SLOT( Vec3d , scale_src , INPUT        , Vec3d{1,1,1} );
    ADD_SLOT( Vec3d , scale_dst , INPUT        , Vec3d{0,0,0} );
    ADD_SLOT( Vec3d , add_dst   , INPUT        , Vec3d{0,0,0} );
    ADD_SLOT( GridT , grid      , INPUT_OUTPUT );
    ADD_SLOT( bool  , ghost     , INPUT        , false);

  public:
    inline void execute () override final
    {
      if( grid->number_of_cells() == 0 ) return;
      
      ldbg<<"GenericVec3BlendFunctor scale_src="<<(*scale_src)<<" scale_dst="<<scale_dst<<" , add_dst"<<add_dst<< std::endl;
      
      auto blend_fields = onika::make_flat_tuple(
        grid->field_accessor(Field_DX{}) , grid->field_accessor(Field_DY{}) , grid->field_accessor(Field_DZ{}) , 
        grid->field_accessor(Field_SX{}) , grid->field_accessor(Field_SY{}) , grid->field_accessor(Field_SZ{}) );
      GenericVec3BlendFunctor func = { *scale_src , *scale_dst , *add_dst };
      compute_cell_particles( *grid , *ghost , func , blend_fields , parallel_execution_context() );            
    }
  };

}

