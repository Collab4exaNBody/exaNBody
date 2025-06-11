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

#include <exanb/core/grid_fields.h>
#include <exanb/core/particle_type_id.h>
#include <onika/soatl/field_combiner.h>

// backward compatibility with Onika v1.0.1
#ifndef ONIKA_DECLARE_DYNAMIC_FIELD_COMBINER
#define ONIKA_DECLARE_DYNAMIC_FIELD_COMBINER(ns,CombT,FuncT,...) \
namespace onika { \
namespace soatl { \
template<> struct FieldCombiner<FuncT OPT_COMMA_VA_ARGS(__VA_ARGS__)> { \
  FuncT m_func; \
  using value_type = decltype( m_func( EXPAND_WITH_FUNC(_ONIKA_GET_TYPE_FROM_FIELD_ID OPT_COMMA_VA_ARGS(__VA_ARGS__)) ) ); \
  const char* short_name() const { return m_func.short_name(); } \
  const char* name() const { return m_func.name(); } \
  }; \
} } \
namespace ns { using CombT = onika::soatl::FieldCombiner<FuncT OPT_COMMA_VA_ARGS(__VA_ARGS__)>; }
#endif

namespace exanb
{
  struct TypePropertyScalarFunctor
  {
    const std::string m_name = "TypePropertyScalar";
    const double * m_scalars = nullptr;
    inline const char* short_name() const { return m_name.c_str(); }
    inline const char* name() const { return m_name.c_str(); }
    ONIKA_HOST_DEVICE_FUNC inline double operator () (int t) const { return m_scalars[t]; }
  };
}

ONIKA_DECLARE_DYNAMIC_FIELD_COMBINER( exanb, TypePropertyScalarCombiner , exanb::TypePropertyScalarFunctor , exanb::field::_type )

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
    void update_property_arrays();
    
    template<class StreamT>
    inline StreamT& to_stream(StreamT& out)
    {
      out << "=== Particle types ===" << std::endl;
      size_t n_types = m_names.size();
      for(size_t i=0;i<n_types;i++)
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

