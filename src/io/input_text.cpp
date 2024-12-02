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

#include <iostream>

namespace exanb
{

  struct InputTextNode : public OperatorNode
  {  
    ADD_SLOT( std::string , mesg , INPUT , "input: " );
    ADD_SLOT( bool , endl , INPUT , false );
    ADD_SLOT( std::string , text , OUTPUT , "" );

    inline void execute () override final
    {
      lout << *mesg ;
      if( ! text->empty() ) lout << " ["<<*text<<"] ";
      if( *endl ) lout << std::endl;
      lout << std::flush;
      
      std::string s;
      std::getline( std::cin , s );
      if( !s.empty() ) *text = s;
      ldbg << "text = '"<< *text << "'" << std::endl;
    }

  };
    
  // === register factories ===  
  ONIKA_AUTORUN_INIT(input_text)
  {
    OperatorNodeFactory::instance()->register_factory( "input_text", make_simple_operator<InputTextNode> );
  }

}

