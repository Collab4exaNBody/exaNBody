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

#include <iostream>
#include <string>

namespace exanb
{

  struct InputInteger : public OperatorNode
  {  
    ADD_SLOT( std::string , mesg , INPUT , "input: " );
    ADD_SLOT( bool , endl , INPUT , false );
    ADD_SLOT( long , value , OUTPUT , 0 );

    inline void execute () override final
    {
      lout << *mesg << " [" <<*value<<"] ";
      if( *endl ) lout << std::endl;
      lout << std::flush;

      std::string qstr;
      std::getline( std::cin , qstr );
      if( ! qstr.empty() )
      {
        *value = std::stol( qstr );
      }
      ldbg << "integer = '"<< *value << "'" << std::endl;
    }

  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "input_integer", make_simple_operator<InputInteger> );
  }

}

