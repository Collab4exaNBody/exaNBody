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
#include <onika/log.h>
#include <unistd.h>

namespace exanb
{

  struct PrintMessageNode : public OperatorNode
  {  
    ADD_SLOT( std::string , mesg , INPUT , ""   , DocString{"Message string to print"} );
    ADD_SLOT( bool        , endl , INPUT , true , DocString{"When set to true, a new line is inserted after messase"} );

    // -----------------------------------------------
    // ----------- Operator documentation ------------
    inline std::string documentation() const override final
    {
      return R"EOF(
        Prints a string to the standard output, and optionaly adds a new line.
        )EOF";
    }

    inline void execute () override final
    {
      lout << *mesg ;
      if( *endl ){ lout << std::endl; }
    }

   inline void yaml_initialize(const YAML::Node& node) override final
   {
      YAML::Node tmp;
      if( node.IsScalar() )
      {
        tmp["mesg"] = node;
      }
      else { tmp = node; }
      this->OperatorNode::yaml_initialize( tmp );
   }

  };
    
  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "message", make_simple_operator<PrintMessageNode> );
  }

}

