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
    double m_scale_src = 1.0;
    double m_scale_dst = 1.0;
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( double& dx, double& dy, double& dz, double sx, double sy, double sz ) const
    {
      dx = dx * m_scale_dst + sx * m_scale_src;
      dy = dy * m_scale_dst + sy * m_scale_src;
      dz = dz * m_scale_dst + sz * m_scale_src;
    }
  };

  template<> struct ComputeCellParticlesTraits< GenericVec3BlendFunctor >
  {
    static inline constexpr bool CudaCompatible = true;
  };

  template<
    class GridT,
    class Field_SX, class Field_SY, class Field_SZ,
    class Field_DX, class Field_DY, class Field_DZ >
  class GenericVec3Blender : public OperatorNode
  {  
    ADD_SLOT( double , scale_src , INPUT        , 1.0 );
    ADD_SLOT( double , scale_dst , INPUT        , 1.0);
    ADD_SLOT( GridT  , grid      , INPUT_OUTPUT );
    ADD_SLOT( bool   , ghost     , INPUT        , true );

  public:
    inline void execute () override final
    {
      static constexpr onika::soatl::FieldId<Field_SX> field_sx = {};
      static constexpr onika::soatl::FieldId<Field_SY> field_sy = {};
      static constexpr onika::soatl::FieldId<Field_SZ> field_sz = {};

      static constexpr onika::soatl::FieldId<Field_DX> field_dx = {};
      static constexpr onika::soatl::FieldId<Field_DY> field_dy = {};
      static constexpr onika::soatl::FieldId<Field_DZ> field_dz = {};

      if( grid->number_of_cells() == 0 ) return;
      
      ldbg<<"GenericVec3BlendFunctor scale_src="<<(*scale_src)<<" scale_dst="<<(*scale_dst)<< std::endl;
      
      auto blend_fields = onika::make_flat_tuple(
        grid->field_accessor(field_dx) , grid->field_accessor(field_dy) , grid->field_accessor(field_dz) , 
        grid->field_accessor(field_sx) , grid->field_accessor(field_sy) , grid->field_accessor(field_sz) );
      GenericVec3BlendFunctor func = { *scale_src , *scale_dst };
      compute_cell_particles( *grid , *ghost , func , blend_fields , parallel_execution_context() );            
    }
  };

}

