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
#include <exanb/core/particle_type_id.h>

namespace exanb
{
  class ParticleTypes : public OperatorNode
  { 
    // -----------------------------------------------
    // Operator slots
    // -----------------------------------------------    
    ADD_SLOT( ParticleTypeMap , particle_type_map , INPUT_OUTPUT , ParticleTypeMap{ { "NoType" , 0 } } );
    ADD_SLOT( bool            , verbose           , INPUT        , true );
    
  public:
    inline void execute () override final
    {
      if( *verbose )
      {
        lout << "=== Particle type map ===" << std::endl;
        for(const auto & item:*particle_type_map)
        {
          lout << item.first << " -> " << item.second << std::endl;
        }
        lout << "=========================" << std::endl;
      }
    }

  };


  // === register factories ===
  ONIKA_AUTORUN_INIT(particle_types)
  {
    OperatorNodeFactory::instance()->register_factory("particle_types", make_simple_operator< ParticleTypes >);
  }

}
