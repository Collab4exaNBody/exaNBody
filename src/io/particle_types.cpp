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
#include <exanb/core/particle_type_properties.h>

namespace exanb
{
  class ParticleTypes : public OperatorNode
  { 
    // -----------------------------------------------
    // Operator slots
    // -----------------------------------------------    
    ADD_SLOT( ParticleTypeMap        , particle_type_map        , INPUT_OUTPUT , ParticleTypeMap{} , DocString{"(YAML: dict) Maps string identifiers of particle types to internal integer type IDs."});
    ADD_SLOT( ParticleTypeProperties , particle_type_properties , INPUT_OUTPUT , ParticleTypeProperties{} , DocString{"(YAML: dict) Assigns properties to each particle type. Each key is a particle identifier from particle_type_map, and each value is a dictionary of properties. If a property is not defined for all types, it is created and set to 0 for others."});
    ADD_SLOT( bool                   , verbose                  , INPUT        , true , DocString{"(YAML: bool) If true, enables detailed logging or output for debugging or informational purposes related to particle type parsing and setup.."} );
    
  public:
    inline void execute () override final
    {
      for(const auto & it : *particle_type_map) ldbg << "particle_type_map["<<it.first<<"]="<<it.second<<std::endl;
      particle_type_properties->to_stream( ldbg );
      
      bool need_mutual_update = false;
      for(const auto & it : *particle_type_map)
      {
        if( size_t(it.second) >= particle_type_properties->m_names.size() ) need_mutual_update=true;
        else if( particle_type_properties->m_names[it.second] != it.first ) need_mutual_update=true;
      }
      if( need_mutual_update )
      {
        for(const auto & it : *particle_type_map)
        {
          particle_type_properties->m_name_map[ it.first ];
        }
        for(const auto & it : particle_type_properties->m_name_map)
        {
          if( particle_type_map->find( it.first ) == particle_type_map->end() )
          {
            particle_type_map->insert( { it.first , particle_type_map->size() } );
          }
        }
        particle_type_properties->update_property_arrays( *particle_type_map );
        //*particle_type_map = particle_type_properties->build_type_map();
      }
      if( *verbose ) particle_type_properties->to_stream( lout );
    }

    inline std::string documentation() const override final
    {
      return R"EOF(

Allows to define particle species, type mapping and additional type-related properties.

Usage example:

particle_types:
  verbose: true
  particle_type_map: { A: 0 , B: 1 , C: 2 }
  particle_type_properties:
    A: { mass: 30. Da, radius: 0.5 ang, charge: 1 e- }
    B: { mass: 20. Da, radius: 1.0 ang }
    C: { mass: 10. Da, radius: 3.0 ang }

particle_types:
  verbose: true
  particle_type_map: { A: 0 , B: 1 }
  particle_type_properties: { A: { mass: 30. Da, lambda: 0.1 }, B: { mass: 20. Da } }

)EOF";
    }
  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(particle_types)
  {
    OperatorNodeFactory::instance()->register_factory("particle_types", make_simple_operator< ParticleTypes >);
  }

}
