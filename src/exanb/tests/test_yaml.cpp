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
#include <yaml-cpp/yaml.h>

#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_yaml.h>
#include <onika/type_utils.h>

#include <exaStamp/particle_species/particle_specie.h>
#include <exaStamp/particle_species/particle_specie_yaml.h>
#include <onika/physics/units.h>
#include <exanb/core/quantity_stream.h>

#include <iostream>
#include <string>
#include <vector>
#include <list>

namespace exanb
{
  template <bool A, bool B> struct and_t : std::false_type {};
  template <> struct and_t<true,true> : std::true_type {};

  template <class T>
  struct is_yaml_convertible : IsComplete< ::YAML::convert<T> > {};

  template <class T>
  struct is_yaml_convertible< std::vector<T> > : IsComplete< ::YAML::convert<T> > {};

  template <class T>
  struct is_yaml_convertible< std::list<T> > : IsComplete< ::YAML::convert<T> > {};

  template <class K, class V>
  struct is_yaml_convertible< std::map<K,V> > : and_t< IsComplete< ::YAML::convert<K> >::value , IsComplete< ::YAML::convert<V> >::value > {};
}

int main()
{

  YAML::Node node;
  node["x"] = 3;
  node["y"] = exanb::IJK{1,2,3};

  std::cout << node["x"].as<int>() << std::endl;
  std::cout << node["x"].as<double>() << std::endl;
  std::cout << node["x"].as<std::string>() << std::endl;

  node["y"].as< std::vector<std::string> >();
  std::cout << node["y"][0].as<std::string>() << std::endl;

  std::cout << "conversion defined for IJK : " << std::boolalpha << exanb::is_yaml_convertible<exanb::IJK>() << std::endl;
  std::cout << "conversion defined for Vec3d : " << std::boolalpha << exanb::is_yaml_convertible<exanb::Vec3d>() << std::endl;

  std::cout << "conversion defined for std::vector<IJK> : " << std::boolalpha << exanb::is_yaml_convertible< std::vector<exanb::IJK> >() << std::endl;
  std::cout << "conversion defined for std::vector<Vec3d> : " << std::boolalpha << exanb::is_yaml_convertible< std::vector<exanb::Vec3d> >() << std::endl;

  std::cout << "conversion defined for std::list<IJK> : " << std::boolalpha << exanb::is_yaml_convertible< std::list<exanb::IJK> >() << std::endl;
  std::cout << "conversion defined for std::list<Vec3d> : " << std::boolalpha << exanb::is_yaml_convertible< std::list<exanb::Vec3d> >() << std::endl;

  std::cout << "conversion defined for std::map<int,IJK> : " << std::boolalpha << exanb::is_yaml_convertible< std::map<int,exanb::IJK> >() << std::endl;
  std::cout << "conversion defined for std::map<int,Vec3d> : " << std::boolalpha << exanb::is_yaml_convertible< std::map<int,exanb::Vec3d> >() << std::endl;

  std::cout << "conversion defined for std::map<IJK,int> : " << std::boolalpha << exanb::is_yaml_convertible< std::map<exanb::IJK,int> >() << std::endl;
  std::cout << "conversion defined for std::map<Vec3d,int> : " << std::boolalpha << exanb::is_yaml_convertible< std::map<exanb::Vec3d,int> >() << std::endl;

  std::cout << "conversion defined for std::map<IJK,std::string> : " << std::boolalpha << exanb::is_yaml_convertible< std::map<exanb::IJK,std::string> >() << std::endl;
  std::cout << "conversion defined for std::map<IJK,IJK> : " << std::boolalpha << exanb::is_yaml_convertible< std::map<exanb::IJK,exanb::IJK> >() << std::endl;

  std::cout << "conversion defined for std::vector<ParticleSpecie> : " << std::boolalpha << exanb::is_yaml_convertible< std::vector<exanb::ParticleSpecie> >() << std::endl;
  std::cout << "conversion defined for ParticleSpecies : " << std::boolalpha << exanb::is_yaml_convertible<exanb::ParticleSpecies>() << std::endl;

  node = YAML::Load( "testq: [ 10 kg , 11 g/s , 8 nm/s^2 ]");
  for( auto& q : node["testq"].as< std::vector<exanb::Quantity> >() )
  {
	  std::cout << q << " , ";
  }
  std::cout << std::endl;

  return 0;
}

