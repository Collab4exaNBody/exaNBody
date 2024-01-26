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
#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_operators.h>
#include <exanb/core/basic_types_yaml.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/domain.h>

namespace exanb
{

  struct DomainExtractXForm : public OperatorNode
  {
    ADD_SLOT( Domain     , domain  , INPUT , REQUIRED );
    ADD_SLOT( Mat3d      , xform   , OUTPUT ); // outputs deformation matrix ( not combined with domain's matrix)
    ADD_SLOT( Mat3d      , inv_xform , OUTPUT ); // outputs inverse of deformation matrix

    inline void execute () override final
    {   
      *xform = domain->xform();
      *inv_xform = domain->inv_xform();
    }
  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "domain_extract_xform", make_compatible_operator< DomainExtractXForm > );
  }

}


