#pragma once

#include <onika/memory/allocator.h>
#include <yaml-cpp/yaml.h>

namespace YAML
{
  template<> struct convert< onika::memory::HostAllocationPolicy >
  {
    static inline Node encode(const onika::memory::HostAllocationPolicy& v)
    {
      Node node;
      switch( v )
      {
        case onika::memory::HostAllocationPolicy::MALLOC    : node = "MALLOC"; break;
        case onika::memory::HostAllocationPolicy::CUDA_HOST : node = "CUDA_HOST"; break;
      }
      return node;
    }
    static inline bool decode(const Node& node, onika::memory::HostAllocationPolicy& v)
    {
      if( ! node.IsScalar() ) { return false; }
      std::string s = node.as<std::string>();
      if( s == "MALLOC" ) v = onika::memory::HostAllocationPolicy::MALLOC;
      else if( s == "CUDA_HOST" ) v = onika::memory::HostAllocationPolicy::CUDA_HOST;
      else return false;
      return true;
    }    
  };  
}

