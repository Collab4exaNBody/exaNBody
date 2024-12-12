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

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>
#include <exanb/core/string_utils.h>
#include <exanb/core/domain.h>

namespace microStamp
{
  using namespace exanb;
  using SimulationState = std::vector<double>;

  class PrintSimulationState : public OperatorNode
  {  
    // thermodynamic state & physics data
    ADD_SLOT( long               , timestep            , INPUT , REQUIRED );
    ADD_SLOT( double             , physical_time       , INPUT , REQUIRED );
    ADD_SLOT( SimulationState , simulation_state , INPUT , REQUIRED );

    // LB and particle movement statistics
    ADD_SLOT( long               , lb_counter          , INPUT_OUTPUT );
    ADD_SLOT( long               , move_counter        , INPUT_OUTPUT );
    ADD_SLOT( long               , domain_ext_counter  , INPUT_OUTPUT );
    ADD_SLOT( double             , lb_inbalance_max    , INPUT_OUTPUT );

    // NEW
    ADD_SLOT(Domain              , domain              , INPUT , OPTIONAL, DocString{"Deformation box matrix"} );

  public:
    inline bool is_sink() const override final { return true; }
  
    inline void execute () override final
    {      
      lout << "T=" << (*physical_time) << " , N="<< simulation_state->at(2) << " , Kin.E="<<simulation_state->at(0)<< " , Pot.E="<<simulation_state->at(1) << std::endl;
    }

  };
    
  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "print_simulation_state", make_simple_operator<PrintSimulationState> );
  }

}

