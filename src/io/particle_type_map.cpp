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
    ADD_SLOT( ParticleTypeMap        , particle_type_map        , INPUT_OUTPUT , ParticleTypeMap{} );
    ADD_SLOT( ParticleTypeProperties , particle_type_properties , INPUT_OUTPUT , ParticleTypeProperties{} );
    ADD_SLOT( bool                   , verbose                  , INPUT        , true );
    
  public:
    inline void execute () override final
    {
      for(const auto & it : *particle_type_map) lout << "particle_type_map has "<<it.first<<std::endl;
      for(const auto & it : particle_type_properties->m_name_map) lout << "particle_type_properties has "<<it.first<<std::endl;
      
      if( particle_type_properties->empty() && particle_type_map->empty() )
      {
        particle_type_map->insert( {"NoType",0} );
      }
      for(const auto & it : *particle_type_map)
      {
        particle_type_properties->m_name_map[ it.first ];
      }
      particle_type_properties->update_property_arrays();
      *particle_type_map = particle_type_properties->build_type_map();
      
      if( *verbose )
      {
        lout << "=== Particle types ===" << std::endl;
        for(const auto & item:*particle_type_map)
        {
          lout << item.first << ":" << std::endl
               << "  id = " << item.second << std::endl;
          for(const auto & propit : particle_type_properties->m_scalars)
          {
            lout << "  "<<propit.first<<" = "<<propit.second[ item.second ]<<std::endl;
          }
          for(const auto & propit : particle_type_properties->m_vectors)
          {
            lout << "  "<<propit.first<<" = "<<propit.second[ item.second ]<<std::endl;
          }
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
