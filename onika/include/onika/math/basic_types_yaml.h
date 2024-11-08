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
  using onika::math::IJK;
  using onika::math::Vec3d;
  using onika::math::Plane3d;
  using onika::math::Mat3d;
  using onika::math::AABB;
  using onika::physics::Quantity;
    
  template<> struct convert< IJK >
  {
    static inline Node encode(const IJK& v)
    {
      Node node;
      node.push_back(v.i);
      node.push_back(v.j);
      node.push_back(v.k);
      return node;
    }
    static inline bool decode(const Node& node, IJK& v)
    {
      if(!node.IsSequence() || node.size() != 3) { return false; }
      v.i = node[0].as<int>();
      v.j = node[1].as<int>();
      v.k = node[2].as<int>();
      return true;
    }
  };

  template<> struct convert< Vec3d >
  {
    static inline Node encode(const Vec3d& v)
    {
      Node node;
      node.push_back(v.x);
      node.push_back(v.y);
      node.push_back(v.z);
      return node;
    } 
    static inline bool decode(const Node& node, Vec3d& v)
    {
      if(!node.IsSequence() || node.size() != 3) { return false; }
      v.x = node[0].as<onika::physics::Quantity>().convert();
      v.y = node[1].as<onika::physics::Quantity>().convert();
      v.z = node[2].as<onika::physics::Quantity>().convert();
      return true;
    }
  };

  template<> struct convert< Plane3d >
  {
    static inline Node encode(const Plane3d& p)
    {
      Node node;
      node.push_back(p.N.x);
      node.push_back(p.N.y);
      node.push_back(p.N.z);
      node.push_back(p.D);
      return node;
    }
    static inline bool decode(const Node& node, Plane3d& p)
    {
      if(!node.IsSequence() || node.size() != 4) { return false; }
      p.N.x = node[0].as<onika::physics::Quantity>().convert();
      p.N.y = node[1].as<onika::physics::Quantity>().convert();
      p.N.z = node[2].as<onika::physics::Quantity>().convert();
      p.D   = node[3].as<onika::physics::Quantity>().convert();
      return true;
    }
  };

  

  template<> struct convert< Mat3d >
  {
    static inline Node encode(const Mat3d& v)
    {
      Node node;
      node.push_back( line1(v) );
      node.push_back( line2(v) );
      node.push_back( line3(v) );
      return node;
    }
    static inline bool decode(const Node& node, Mat3d& m)
    {
      if(!node.IsSequence() || node.size() != 3) { return false; }
      set_line1( m, node[0].as<Vec3d>() );
      set_line2( m, node[1].as<Vec3d>() );
      set_line3( m, node[2].as<Vec3d>() );
      return true;
    }
  };

  template<> struct convert< AABB >
  {
    static inline Node encode(const AABB& bb)
    {
      Node node;
      node.push_back( bb.bmin );
      node.push_back( bb.bmax );
      return node;
    }
    static inline bool decode(const Node& node, AABB& bb)
    {
      if(node.IsSequence())
      {
        if( node.size() != 2 ) { return false; }
        bb.bmin = node[0].as<Vec3d>();
        bb.bmax = node[1].as<Vec3d>();
        return true;
      }
      else if(node.IsMap())
      {
        if( ! static_cast<bool>(node["min"]) ) { return false; }
        if( ! static_cast<bool>(node["max"]) ) { return false; }
        bb.bmin = node["min"].as<Vec3d>();
        bb.bmax = node["max"].as<Vec3d>();
        return true;
      }
      else
      {
        return false;
      }
    }
  };

/*
 from deleted tools/microStamp/src/core/basic_types_yaml.cpp

  Node convert<std::map<std::string, std::array<int,4> > >::encode(const std::map<std::string, std::array<int,4> >& w)
  {
    Node node;

    for(auto it=w.begin(); it!=w.end();it++)
      node[it->first] = std::vector<int>{it->second[0],
                                         it->second[1],
                                         it->second[2],
                                         it->second[3]};
    return node;
  }

  bool convert<std::map<std::string, std::array<int,4> > >::decode(const Node& node, std::map<std::string, std::array<int,4> >& w)
  {
    for(const auto& n : node)
      {
        if(!n.IsMap())
          return false;

        w[n.first.as<std::string>()] = std::array<int,4>{n.second[0].as<int>(),
                                                         n.second[1].as<int>(),
                                                         n.second[2].as<int>(),
                                                         n.second[3].as<int>()};
      }

    return true;
  }

*/

}

