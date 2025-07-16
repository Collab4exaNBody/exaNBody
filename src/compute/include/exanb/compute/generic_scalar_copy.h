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

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <exanb/compute/compute_cell_particles.h>

namespace exanb
{

  struct GenericScalarCopyFunctor
  {
    template<class T>
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( T & d, T s ) const
    {
      d = s;
    }
  };

  template<> struct ComputeCellParticlesTraits< GenericScalarCopyFunctor >
  {
    static inline constexpr bool CudaCompatible = true;
  };

  template<class GridT, class Field_SRC, class Field_DST >
  class GenericScalarCopy : public OperatorNode
  {  
    ADD_SLOT( GridT  , grid      , INPUT_OUTPUT );
    ADD_SLOT( bool   , ghost     , INPUT        , true );

  public:
    inline void execute () override final
    {
      static constexpr onika::soatl::FieldId<Field_SRC> field_src = {};
      static constexpr onika::soatl::FieldId<Field_DST> field_dst = {};

      if( grid->number_of_cells() == 0 ) return;
      auto blend_fields = onika::make_flat_tuple( grid->field_accessor(field_dst) , grid->field_accessor(field_src) );
      compute_cell_particles( *grid , *ghost , GenericScalarCopyFunctor{} , blend_fields , parallel_execution_context() );            
    }
  };

}

