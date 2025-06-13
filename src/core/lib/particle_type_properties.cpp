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

#include <exanb/core/particle_type_properties.h>

namespace exanb
{
    double * __restrict__ ParticleTypeProperties::scalar_property(const std::string& key)
    {
      m_scalars[key].resize( MAX_PARTICLE_TYPES , 0.0 );
      return m_scalars[key].data();
    }
    
    bool ParticleTypeProperties::has_scalar_property(const std::string& key) const
    {
      return m_scalars.find(key) != m_scalars.end();
    }
    
    const double * __restrict__ ParticleTypeProperties::scalar_property(const std::string& key) const
    {
      auto it = m_scalars.find(key);
      if( it == m_scalars.end() ) { onika::fatal_error()<<"Scalar property '"<<key<<"' does not exist"<<std::endl; }
      return it->second.data();
    }
    
    onika::math::Vec3d * __restrict__ ParticleTypeProperties::vector_property(const std::string& key)
    {
      m_vectors[key].resize( MAX_PARTICLE_TYPES , { 0.0 , 0.0 , 0.0 } );
      return m_vectors[key].data();
    }
    
    bool ParticleTypeProperties::has_vector_property(const std::string& key) const
    {
      return m_vectors.find(key) != m_vectors.end();
    }
    const onika::math::Vec3d * __restrict__ ParticleTypeProperties::vector_property(const std::string& key) const
    {
      auto it = m_vectors.find(key);
      if( it == m_vectors.end() ) { onika::fatal_error()<<"Vector property '"<<key<<"' does not exist"<<std::endl; }
      return it->second.data();
    }
    
    bool ParticleTypeProperties::empty() const
    {
      return m_name_map.empty();
    }
    
    ParticleTypeMap ParticleTypeProperties::build_type_map() const
    {
      ParticleTypeMap tm;
      int64_t id = 0;
      for(const auto & it : m_name_map)
      {
        tm[it.first] = id ++;
      }
      return tm;
    }
    
    void ParticleTypeProperties::update_property_arrays(const ParticleTypeMap& tm)
    {
      m_scalars.clear();
      m_vectors.clear();
      m_names.assign( MAX_PARTICLE_TYPES , "" );
      for(const auto & type_it : m_name_map)
      {
        const auto & type_name = type_it.first;
        auto it = tm.find(type_name);
        if( it == tm.end() ) { fatal_error() << "Invalid type name map"<<std::endl; }
        const auto type_id = it->second;
        if( type_id<0 || size_t(type_id)>=m_names.size() ) { fatal_error() << "Invalid type id "<<type_id<<", bigger than maximum "<<m_names.size() <<std::endl; }
        m_names[ type_id ] = type_name;
        for(const auto & prop_it : type_it.second.m_scalars) scalar_property(prop_it.first) [ type_id ] = prop_it.second;
        for(const auto & prop_it : type_it.second.m_vectors) vector_property(prop_it.first) [ type_id ] = prop_it.second;
      }
    }

}
