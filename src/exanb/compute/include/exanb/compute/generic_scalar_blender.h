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

  struct GenericScalarBlendFunctor
  {
    double m_scale_src = 1.0;
    double m_scale_dst = 0.0;
    double m_add_dst = 0.0;
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( double& d, double s ) const
    {
      d = d * m_scale_dst + s * m_scale_src + m_add_dst;
    }
  };

  template<> struct ComputeCellParticlesTraits< GenericScalarBlendFunctor >
  {
    static inline constexpr bool CudaCompatible = true;
  };

  template<
    class GridT,
    class Field_SRC, class Field_DST,
    class = AssertGridHasFields< GridT, Field_DST , Field_SRC >
    >
  class GenericScalarBlender : public OperatorNode
  {  
    ADD_SLOT( double , scale_src , INPUT        , 1.0 );
    ADD_SLOT( double , scale_dst , INPUT        , 0.0 );
    ADD_SLOT( double , add_dst   , INPUT        , 0.0 );
    ADD_SLOT( GridT , grid      , INPUT_OUTPUT );
    ADD_SLOT( bool  , ghost     , INPUT        , false);

  public:
    inline void execute () override final
    {
      if( grid->number_of_cells() == 0 ) return;
      ldbg<<"GenericScalarBlendFunctor scale_src="<<(*scale_src)<<" scale_dst="<<scale_dst<<" , add_dst"<<add_dst<< std::endl;
      auto blend_fields = onika::make_flat_tuple( grid->field_accessor(Field_DST{}) , grid->field_accessor(Field_SRC{}) );
      GenericScalarBlendFunctor func = { *scale_src , *scale_dst , *add_dst };
      compute_cell_particles( *grid , *ghost , func , blend_fields , parallel_execution_context() );            
    }
  };

}

