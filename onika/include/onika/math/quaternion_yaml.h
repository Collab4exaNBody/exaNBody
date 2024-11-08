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

#include <onika/math/basic_types_def.h>
#include <onika/math/basic_types_accessors.h>
#include <onika/physics/units.h>

namespace YAML
{
  using onika::math::Quaternion;
  using onika::physics::Quantity;

  template <> struct convert<Quaternion>
  {
    static inline Node encode(const Quaternion &q)
    {
      Node node;
      node.push_back(q.w);
      node.push_back(q.x);
      node.push_back(q.y);
      node.push_back(q.z);
      return node;
    }
    static inline bool decode(const Node &node, Quaternion &q)
    {
      if (!node.IsSequence() || node.size() != 4)
      {
        return false;
      }
      q.w = node[0].as<Quantity>().convert();
      q.x = node[1].as<Quantity>().convert();
      q.y = node[2].as<Quantity>().convert();
      q.z = node[3].as<Quantity>().convert();
      return true;
    }
  };
} // namespace YAML
