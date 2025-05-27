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
#include <onika/yaml/yaml_utils.h>

namespace microcosmos
{
  struct PlanetProperties
  {
    std::string m_name;
    double m_mass = 1.0;
    double m_radius = 1.0;
  };
  
  using PlanetTypeArray = onika::memory::CudaMMVector<PlanetProperties>;
}

namespace YAML
{
  template<> struct convert< microcosmos::PlanetTypeArray >
  {
    static bool decode(const Node& node, microcosmos::PlanetTypeArray & v)
    {
      v.clear();
      if( !node.IsSequence() )
      {
        onika::lerr << "PlanetTypeArray expects a list" << std::endl;
        return false;
      }
      for (auto item : node)
      {
        const auto name = item.first.as<std::string>();
        double mass = 1.0;
        double radius = 1.0;
        if( item.second["mass"] ) mass = item.second["mass"].as<onika::physics::Quantity>().convert();
        if( item.second["radius"] ) radius = item.second["radius"].as<onika::physics::Quantity>().convert();
        v.push_back( { name , mass , radius } );
      }
      return true;
    }
  };
}

