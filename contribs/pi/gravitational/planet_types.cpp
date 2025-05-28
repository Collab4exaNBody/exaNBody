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
#include <onika/memory/allocator.h>
#include <exanb/core/particle_type_id.h>

#include <microcosmos/planet_types.h>

namespace microcosmos
{
  using namespace exanb;
  
  class PlanetTypes : public OperatorNode
  {
    // -----------------------------------------------
    // Operator slots
    // -----------------------------------------------    
    ADD_SLOT( PlanetarySystem , planet_types      , INPUT_OUTPUT , PlanetarySystem{} );
    ADD_SLOT( ParticleTypeMap , particle_type_map , INPUT_OUTPUT , ParticleTypeMap{} );
    ADD_SLOT( bool            , verbose           , INPUT        , true );
    
  public:
    inline void execute () override final
    {
      particle_type_map->clear();
      for(size_t i=0;i<planet_types->m_planet_properties.size();i++)
      {
        particle_type_map->insert( { planet_types->m_planet_properties.at(i).m_name , i } );
      }

      if( *verbose )
      {
        lout << "====== Planet types ======" << std::endl;
        for(const auto & item:*particle_type_map)
        {
          lout << item.first << " : type=" << item.second << " , mass="<<planet_types->m_planet_properties.at(item.second).m_mass<<" , radius="<<planet_types->m_planet_properties.at(item.second).m_radius<<std::endl;
        }
        lout << "==========================" << std::endl;
      }
    }
    
    inline void yaml_initialize(const YAML::Node& node) override final
    {
      YAML::Node tmp;
      if( ! node["planet_types"] )
      {
        tmp["planet_types"] = node;
      }
      else { tmp = node; }
      this->OperatorNode::yaml_initialize(tmp);
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(planet_types)
  {
    OperatorNodeFactory::instance()->register_factory("planet_types", make_simple_operator< PlanetTypes >);
  }

}
