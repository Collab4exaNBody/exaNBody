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
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>

#include <memory>

namespace exanb
{

  // =====================================================================
  // ========================== NthTimeStepNode ========================
  // =====================================================================

  class BooleanSetNode : public OperatorNode
  {
  public:
  
    ADD_SLOT( bool , value , INPUT_OUTPUT);
    ADD_SLOT( bool , set , INPUT);
    
    void execute() override final
    {
      *value = *set;
    }

  // === register factories ===  
  CLASS_CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "boolean_set", make_compatible_operator< BooleanSetNode > );
  }

  };

}

