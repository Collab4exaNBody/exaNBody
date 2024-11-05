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

#include <onika/type_utils.h>
#include <onika/physics/units.h>

#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <cassert>
#include <cstring>

namespace onika
{

  namespace yaml
  {

    template <typename T>
    inline T read(YAML::Node n, const std::string& key, const T& defval)
    {
      if(n[key])
      {
        return n[key].as<T>();
      }
      else
      {
        return defval;
      }
    }

    template<typename StreamT>
    inline void dump_node_to_stream(StreamT& out, YAML::Node config)
    {
      YAML::Emitter yaml_out;
      yaml_out << config;
      out << yaml_out.c_str() ;
    }

    YAML::Node merge_nodes(YAML::Node a, YAML::Node b, bool append_list=false);
    YAML::Node remove_map_key(YAML::Node a, const std::string& k);

    void dump_node_to_file(const std::string& basename, YAML::Node config);

    std::vector<std::string> resolve_config_file_includes(const std::string& app_path, const std::vector<std::string>& file_names );
    YAML::Node yaml_load_file_abort_on_except(const std::string& file_name);


    // ============== YAML conversion =================================

    // --- Utility templates ---
    namespace yaml_convert_details
    {
      template <bool A, bool B> struct _and_t : std::false_type {};
      template <> struct _and_t<true,true> : std::true_type {};

      template <class T>
      struct is_yaml_convertible : IsComplete< ::YAML::convert<T> > {};

      template <class T, class A>
      struct is_yaml_convertible< std::vector<T,A> > : is_yaml_convertible< T > {};

      template <class T>
      struct is_yaml_convertible< std::list<T> > : is_yaml_convertible< T > {};

      template <class K, class V>
      struct is_yaml_convertible< std::map<K,V> > : _and_t< is_yaml_convertible<K>::value , is_yaml_convertible<V>::value > {};

      //template<typename T, bool = is_yaml_convertible<T>() > struct TypeHelper;
    }
    template<class T> static inline constexpr bool is_yaml_convertible_v = yaml_convert_details::is_yaml_convertible<T>::value ;

    template<class T, bool = is_yaml_convertible_v<T> > struct YAMLConvertWrapper
    {
      static inline bool decode(const YAML::Node& node, T& value)
      {
        //std::cout<<"call YAML conversion for "<<typeid(T).name()<<std::endl;
        return YAML::convert<T>::decode(node,value);
      }
    };
    template<class T> struct YAMLConvertWrapper<T,false>
    {
      static inline bool decode(const YAML::Node&, const T&)
      {
        //std::cout<<"don't know how to convert "<<typeid(T).name()<<std::endl;
        return false;
      }
    };

    template<> struct YAMLConvertWrapper<double,true>
    {
      static inline bool decode(const YAML::Node& node, double& value)
      {
        using onika::physics::Quantity;
        Quantity q;
        if( ! YAML::convert<Quantity>::decode(node,q) ) { return false; }
        value = q.convert();
        return true;
      }
    };

  }
  
}


