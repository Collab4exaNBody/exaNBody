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
  class ParticleTypeAddProperty : public OperatorNode
  { 
    // -----------------------------------------------
    // Operator slots
    // -----------------------------------------------    
    ADD_SLOT( ParticleTypeMap        , particle_type_map        , INPUT , REQUIRED );
    ADD_SLOT( ParticleTypeProperties , properties , INPUT , REQUIRED );
    ADD_SLOT( bool                   , verbose                  , INPUT        , true );
    ADD_SLOT( ParticleTypeProperties , particle_type_properties , INPUT_OUTPUT , ParticleTypeProperties{} );
    
  public:
    inline void execute () override final
    {
      for(const auto & it : *particle_type_map)
      {
        const auto & type_name = it.first;
        const auto type_id = it.second;
        if( type_id<0 || size_t(type_id)>=particle_type_properties->m_names.size() )
        {
          fatal_error() << "type id #"<<type_id<<" inconsistent with particle_type_properties type names"<<std::endl;
        }
        if( particle_type_properties->m_names[type_id] != type_name )
        {
          fatal_error() << "type id #"<<type_id<<" : name in particle_type_properties ("<<particle_type_properties->m_names[type_id]<<") different from name in particle_type_map ("<<type_name<<")"<< std::endl;
        }
      }
      
      for(const auto& type_it : properties->m_name_map)
      {
        const auto & type_name = type_it.first;
        if( particle_type_map->find(type_name) == particle_type_map->end() )
        {
          fatal_error() << "particle type '"<<type_name<<"' does not exist. You can only add properties to existing particle types." << std::endl;
        }
        const auto type_id = particle_type_map->at(type_name);
        for(const auto & prop_it : type_it.second.m_scalars) particle_type_properties->scalar_property(prop_it.first) [ type_id ] = prop_it.second;
        for(const auto & prop_it : type_it.second.m_vectors) particle_type_properties->vector_property(prop_it.first) [ type_id ] = prop_it.second;
      }
      
      if( *verbose ) particle_type_properties->to_stream( lout );
    }

    inline void yaml_initialize(const YAML::Node& node) override final
    {
      YAML::Node tmp;
      if( node.IsMap() && ! node["properties"] )
      {
        tmp["properties"] = node;
      }
      else { tmp = node; }
      this->OperatorNode::yaml_initialize(tmp);
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(particle_type_add_properties)
  {
    OperatorNodeFactory::instance()->register_factory("particle_type_add_properties", make_simple_operator< ParticleTypeAddProperty >);
  }

}
