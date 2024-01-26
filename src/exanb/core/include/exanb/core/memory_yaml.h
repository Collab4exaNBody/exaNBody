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

