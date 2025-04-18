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

#include <yaml-cpp/yaml.h>
#include <onika/physics/units.h>

#include <string>

namespace md
{
  using namespace exanb;


  // Snap Parameters
  struct SnapParms
  {
    std::string lammps_param;
    std::string lammps_coef;
    int nt = 2;
  };

}

// Yaml conversion operators, allows to read potential parameters from config file
namespace YAML
{
  template<> struct convert<md::SnapParms>
  {
    static bool decode(const Node& node, md::SnapParms& v)
    {
      if( !node.IsMap() ) { return false; }
      if( ! node["param"] ) { return false; }
      if( ! node["coef"] ) { return false; }
      v.lammps_param = node["param"].as<std::string>();
      v.lammps_coef  = node["coef"].as<std::string>();
      if( node["nt"] ) { v.nt = node["nt"].as<int>(); }
      return true;
    }
  };
}
