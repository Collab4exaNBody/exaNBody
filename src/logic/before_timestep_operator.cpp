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
#include <onika/yaml/yaml_utils.h>
#include <memory>

namespace exanb
{

  // =====================================================================
  // ========================== NthTimeStepNode ========================
  // =====================================================================

  class BeforeTimeStepNode : public OperatorNode
  {
  public:
  
    ADD_SLOT( long , timestep , INPUT , REQUIRED );
    ADD_SLOT( long , end_at   , INPUT , REQUIRED );
    ADD_SLOT( bool , result   , INPUT_OUTPUT );
    
    void execute() override final
    {
      long timestep = *(this->timestep);
      long end_at = *(this->end_at);
      bool& out = *result;
      
      out = ( timestep <= end_at );
      ldbg << "BeforeTimeStepNode: timestep="<<timestep<<", result="<<std::boolalpha<<out<<std::endl;
    }    

    inline void yaml_initialize(const YAML::Node& node) override final
    {
      YAML::Node tmp;
      if( node.IsScalar() )
      {
        tmp["end_at"] = node;
      }
      else { tmp = node; }
      this->OperatorNode::yaml_initialize( tmp );
    }

  };

   // === register factories ===  
  ONIKA_AUTORUN_INIT(before_timestep_operator)
  {
    OperatorNodeFactory::instance()->register_factory( "before_timestep", make_simple_operator<BeforeTimeStepNode> );
  }

}

