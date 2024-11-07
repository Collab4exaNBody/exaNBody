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
#include <exanb/core/operator_factory.h>
#include <exanb/core/operator_slot.h>
#include <onika/log.h>

#include <memory>

namespace exanb
{

  struct NextTimeStepNode : public OperatorNode
  {  
    ADD_SLOT( long   , timestep      , INPUT_OUTPUT );
    ADD_SLOT( double , dt            , INPUT        , REQUIRED );
    ADD_SLOT( double , physical_time , INPUT_OUTPUT );
    // this is equivalent to
    // OperatorSlot<int64_t> timestep { this, "timestep", INPUT_OUTPUT };

    inline void execute() override final
    {
      ++ *timestep;
      *physical_time += *dt;
      //ldbg << "timstep       -> " << timestep <<std::endl;
      //std::cout << "physical_time -> " << *physical_time <<std::endl;
    }

  };
 
  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "next_time_step", make_compatible_operator< NextTimeStepNode > );
  }

}

