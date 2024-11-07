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

namespace exanb
{

  template<class ValueType>
  struct PrintValue : public OperatorNode
  {  
    ADD_SLOT( std::string , prefix , INPUT , "" , DocString{"text to print before"} );
    ADD_SLOT( ValueType , value , INPUT , REQUIRED , DocString{"input value to print"} );
    ADD_SLOT( std::string , suffix , INPUT , "" , DocString{"text to print after"} );

    // -----------------------------------------------
    // ----------- Operator documentation ------------
    inline std::string documentation() const override final
    {
      return R"EOF(
        Prints a value to the standard output
        )EOF";
    }

    inline void execute () override final
    {
      lout << *prefix << *value << *suffix;
    }

   inline void yaml_initialize(const YAML::Node& node) override final
   {
      YAML::Node tmp;
      if( node.IsScalar() )
      {
        tmp["value"] = node;
      }
      else { tmp = node; }
      this->OperatorNode::yaml_initialize( tmp );
   }

  };
    
  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "print_real", make_simple_operator< PrintValue<double> > );
    OperatorNodeFactory::instance()->register_factory( "print_int", make_simple_operator< PrintValue<long> > );
    OperatorNodeFactory::instance()->register_factory( "print_bool", make_simple_operator< PrintValue<bool> > );
  }

}

