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
#include <onika/string_utils.h>

#include <memory>

namespace exanb
{

  class TimeStepFileNameOperator : public OperatorNode
  {  
    ADD_SLOT( long        , timestep , INPUT , REQUIRED );
    ADD_SLOT( std::string , format   , INPUT , REQUIRED );
    ADD_SLOT( std::string , filename , OUTPUT );

  public:
    inline void execute() override final
    {
      *filename = onika::format_string( *format , *timestep );
      ldbg << "timestep file = " << *filename <<std::endl;
    }

    inline void yaml_initialize(const YAML::Node& node) override final
    {
      YAML::Node tmp;
      if( node.IsScalar() )
      {
        tmp["format"] = node;
      }
      else { tmp = node; }
      this->OperatorNode::yaml_initialize( tmp );
    }

  };
 
  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "timestep_file", make_compatible_operator< TimeStepFileNameOperator > );
  }

}

