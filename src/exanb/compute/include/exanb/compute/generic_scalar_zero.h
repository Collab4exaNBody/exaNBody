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

  struct GenericScalarZeroFunctor
  {
    template<class... T>
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( T& ... d ) const
    {
      ( ... , ( d = 0 ) );
    }
  };

  template<> struct ComputeCellParticlesTraits< GenericScalarZeroFunctor >
  {
    static inline constexpr bool CudaCompatible = true;
  };

  template<class GridT, class... Fields_To_Zero >
  class GenericScalarZero : public OperatorNode
  {  
    ADD_SLOT( GridT  , grid      , INPUT_OUTPUT );
    ADD_SLOT( bool   , ghost     , INPUT        , true );

  public:
    inline void execute () override final
    {
      if( grid->number_of_cells() == 0 ) return;
      auto zero_fields = onika::make_flat_tuple( grid->field_accessor( onika::soatl::FieldId<Fields_To_Zero>{} ) ...  );
      compute_cell_particles( *grid , *ghost , GenericScalarZeroFunctor{} , zero_fields , parallel_execution_context() );            
    }
  };

}

