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
#include <onika/math/basic_types_stream.h>
#include <exanb/core/domain.h>

#include <iostream>
#include <string>

namespace exanb
{

  struct DomainUpdateNode : public OperatorNode
  {
    ADD_SLOT( Domain , domain , INPUT_OUTPUT );

    inline void execute() override final
    {
      compute_domain_bounds( *domain, ReadBoundsSelectionMode::DOMAIN_BOUNDS );
      ldbg<<"domain update: "<< *domain << std::endl;
      check_domain( *domain );
    }

  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "domain_update", make_simple_operator<DomainUpdateNode> );
  }

}

