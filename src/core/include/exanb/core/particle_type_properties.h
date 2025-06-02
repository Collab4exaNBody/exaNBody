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

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <onika/math/basic_types.h>
#include <onika/memory/allocator.h>
#include <exanb/core/particle_type_id.h>

namespace exanb
{
  // structure suited for user input from YAML, using type names
  struct ParticleTypePropertyValues
  {
    std::map< std::string , double > m_scalars;
    std::map< std::string , onika::math::Vec3d > m_vectors;
  };

  // defines scalar and vector property values for each particle type id
  struct ParticleTypeProperties
  {
    std::map< std::string , ParticleTypePropertyValues > m_name_map;
    std::map< std::string , onika::memory::CudaMMVector<double> > m_scalars;
    std::map< std::string , onika::memory::CudaMMVector<onika::math::Vec3d> > m_vectors;

    bool has_scalar_property(const std::string& key) const;
    double * __restrict__ scalar_property(const std::string& key);
    const double * __restrict__ scalar_property(const std::string& key) const;
    bool has_vector_property(const std::string& key) const;
    onika::math::Vec3d * __restrict__ vector_property(const std::string& key);
    const onika::math::Vec3d * __restrict__ vector_property(const std::string& key) const;
    bool empty() const;
    ParticleTypeMap build_type_map() const;
    void update_property_arrays();
  };

}
   
namespace YAML
{
  template<> struct convert< exanb::ParticleTypeProperties >
  {
    static bool decode(const Node& node, exanb::ParticleTypeProperties & properties)
    {
      properties.m_name_map.clear();
      auto & v = properties.m_name_map;
      v.clear();
      if( !node.IsMap() )
      {
        onika::lerr << "ParticleTypeNameProperties expects a map" << std::endl;
        return false;
      }
      for (auto item : node)
      {
        const auto type_name = item.first.as<std::string>();
        auto & type_values = v[type_name];
        for (auto kv : item.second)
        {
          const auto prop_name = kv.first.as<std::string>();
          if( kv.second.IsSequence() ) type_values.m_vectors[prop_name] = kv.second.as<onika::math::Vec3d>();
          else type_values.m_scalars[prop_name] = kv.second.as<onika::physics::Quantity>().convert();
        }
      }
      properties.update_property_arrays();
      return true;
    }
  };
}

