#pragma once

#include <exanb/defbox/deformation.h>
#include <exanb/core/basic_types_yaml.h>
#include <yaml-cpp/yaml.h>

namespace YAML
{
  using exanb::Deformation;
  
  template<> struct convert<Deformation>
  {
    static inline bool decode(const Node& node, Deformation& v)
    {
      if(!node.IsMap() ) { return false; }
      v = Deformation();
      if( node["angles"] )
      {
        v.m_angles = node["angles"].as<Vec3d>();
      }
      if( node["extension"] )
      {
        v.m_extension = node["extension"].as<Vec3d>();
      }
      return true;
    }
  };
}

