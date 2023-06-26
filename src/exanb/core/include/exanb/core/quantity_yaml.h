#pragma once

#include <yaml-cpp/yaml.h>
#include <exanb/core/quantity.h>
#include <exanb/core/unityConverterHelper.h>

#include <sstream>
#include <limits>
#include <cassert>

namespace YAML
{
  using exanb::Quantity;
  using exanb::quantity_from_string;
  using exanb::UnityConverterHelper;
    
  template<> struct convert< Quantity >
  {
    static inline Node encode(const Quantity& q)
    {
      Node node;
      node["value"] = q.m_value;
      node["unity"] = UnityConverterHelper::unities_descriptor_to_string( q.m_unit );
      return node;
    }

    static inline bool decode(const Node& node, Quantity& q)
    {
      if( node.IsScalar() ) // value and unit in the same string
      {
        bool conv_ok = false;
        q = quantity_from_string( node.as<std::string>() , conv_ok );
        return conv_ok;
      }
      else if( node.IsMap() )
      {
        q.m_value = node["value"].as<double>();
        UnityConverterHelper::parse_unities_descriptor( node["unity"].as<std::string>(), q.m_unit );
        return true;
      }
      else
      {
        return false;
      }
    }

  };

}

