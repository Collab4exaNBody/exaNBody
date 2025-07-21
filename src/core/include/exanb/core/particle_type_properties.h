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
#include <cstring>
#include <onika/math/basic_types.h>
#include <onika/memory/allocator.h>

#include <exanb/core/grid_fields.h>
#include <exanb/core/particle_type_id.h>
#include <onika/soatl/field_combiner.h>

namespace exanb
{
  template<class PropertyValueType>
  struct TypePropertyFunctor
  {
    static inline constexpr size_t MAX_NAME_LEN = 16;
    char m_name[MAX_NAME_LEN] = "TypeScalar";
    const PropertyValueType * __restrict__ m_values = nullptr;
    inline const char* short_name() const { return m_name; }
    inline const char* name() const { return m_name; }
    ONIKA_HOST_DEVICE_FUNC inline auto operator () (int t) const { return m_values[t]; }
  };
  using TypePropertyScalarFunctor = TypePropertyFunctor<double>;
  using TypePropertyVec3Functor = TypePropertyFunctor<onika::math::Vec3d>;
  using TypePropertyMat3Functor = TypePropertyFunctor<onika::math::Mat3d>;
  
  template<class T>
  inline auto make_type_property_functor(const std::string& name, const T * __restrict__ data)
  {
    TypePropertyFunctor<T> func;
    std::strncpy( func.m_name , name.c_str() , func.MAX_NAME_LEN );
    func.m_name[ func.MAX_NAME_LEN - 1 ] = '\0';
    func.m_values = data;
    return func;
  }
}

ONIKA_DECLARE_DYNAMIC_FIELD_COMBINER( exanb, TypePropertyScalarCombiner , exanb::TypePropertyScalarFunctor , exanb::field::_type )
ONIKA_DECLARE_DYNAMIC_FIELD_COMBINER( exanb, TypePropertyVec3Combiner , exanb::TypePropertyVec3Functor , exanb::field::_type )
ONIKA_DECLARE_DYNAMIC_FIELD_COMBINER( exanb, TypePropertyMat3Combiner , exanb::TypePropertyMat3Functor , exanb::field::_type )

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
    std::vector< std::string > m_names;
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

    void update_property_arrays(const ParticleTypeMap& type_name_map);
    inline void update_property_arrays() { this->update_property_arrays( build_type_map() ); }
    
    template<class StreamT>
    inline StreamT& to_stream(StreamT& out)
    {
      out << "=== Particle types ===" << std::endl;
      size_t n_types = m_names.size();
      for(size_t i=0;i<n_types;i++)
      {
        if( m_names[i] != "" )
        {
          out << m_names[i] << ":" << std::endl
               << "  id = " << i << std::endl;
          for(const auto & propit : m_scalars)
          {
            out << "  "<<propit.first<<" = "<<propit.second[ i ]<<std::endl;
          }
          for(const auto & propit : m_vectors)
          {
            out << "  "<<propit.first<<" = "<<propit.second[ i ]<<std::endl;
          }
        }
      }
      out << "=========================" << std::endl;
      return out;
    }
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

