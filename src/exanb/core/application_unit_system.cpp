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

//#pragma xstamp_cuda_enable // DO NOT REMOVE THIS LINE

#include <utility>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/physics/units.h>



namespace exanb
{

  class ApplicationUnitSystem : public onika::scg::OperatorNode
  {
    using UnitSystem = onika::physics::UnitSystem;
    static inline constexpr UnitSystem default_internal_unit_system()
    {
      using namespace onika::physics;
      return { { XNB_APP_INTERNAL_UNIT_SYSTEM } };
    }
  
    ADD_SLOT( UnitSystem , unit_system , INPUT , default_internal_unit_system() );

  public:

    inline void execute () override final
    {
      onika::physics::set_internal_unit_system( *unit_system );
    }
  };
  
  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   onika::scg::OperatorNodeFactory::instance()->register_factory( "application_unit_system", onika::scg::make_compatible_operator< ApplicationUnitSystem > );
  }

}

