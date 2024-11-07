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
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/domain.h>
#include <onika/string_utils.h>

#include <iostream>
#include <string>

namespace exanb
{

  struct DomainInitNode : public OperatorNode
  {
    ADD_SLOT( Domain , domain , OUTPUT );
    ADD_SLOT( double , default_cell_size , INPUT ); // if domain's cell_size is 0.0, we scale the ghost_dist by this factor to obtaine cell_size
    ADD_SLOT( double , auto_cell_size_factor , INPUT , 2.0 ); // if domain's cell_size is 0.0, we scale the ghost_dist by this factor to obtaine cell_size
    ADD_SLOT( double , ghost_dist , INPUT , 8.8 ); // 8.8 angstrom default

    inline void execute() override final
    {
      double gdist = *ghost_dist;
      
      if( default_cell_size.has_value() && *default_cell_size>0.0 && domain->cell_size()==0.0)
      {
        ldbg << "set domain's cell size to "<< *default_cell_size << std::endl;
        domain->set_cell_size( *default_cell_size );
      }
      
      if( gdist > 0.0 && domain->cell_size() == 0.0 )
      {
        double f = *auto_cell_size_factor;
        ldbg << "automatic cell_size = "<<gdist<<" * "<<f<<" = "<< onika::format_string("%.1f",gdist*f) <<" ang"<< std::endl;
        domain->set_cell_size( gdist * f );
      }
      ldbg<<"domain init: "<< *domain << std::endl;
      //check_domain( *domain );
    }

    inline void yaml_initialize(const YAML::Node& node) override final
    {
      YAML::Node tmp;
      if( ! node["domain"] )
      {
        tmp["domain"] = node;
      }
      else { tmp = node; }
      this->OperatorNode::yaml_initialize(tmp);
    }

  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "domain", make_simple_operator<DomainInitNode> );
  }

}

