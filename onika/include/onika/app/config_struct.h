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
#include <type_traits>
#include <iostream>
#include <iomanip>
#include <string>

namespace onika
{
  namespace app
  {

    template<class T> static inline std::string xsv2_to_string_helper( const T* state );

    template<class T>
    struct AppConfigOptionState
    {
      static inline std::string to_string(const T* state)
      {
        if(state==nullptr) return "<null>";
        else return std::to_string(*state);
      }
    };

    template<>
    struct AppConfigOptionState<std::string>
    {
      static inline const std::string& to_string(const std::string* state)
      {
        static const std::string nullstr("<null>");
        if(state==nullptr) return nullstr;
        else return *state;
      }
    };

    template<>
    struct AppConfigOptionState<YAML::Node>
    {
      static inline std::string to_string(const YAML::Node* state)
      {
        static const std::string nullstr("<null>");
        if(state==nullptr) return nullstr;
        else
        {
          YAML::Emitter yaml_out;
          yaml_out << *state;
          return yaml_out.c_str() ;
        }
      }
    };

    template<class T>
    struct AppConfigOptionState< std::vector<T> >
    {
      static inline std::string to_string(const std::vector<T>* state)
      {
        if(state==nullptr) return "<null>";
        else
        {
          std::ostringstream oss;
          oss<<"["; bool s=false;
          for(const auto& x:*state) { if(s)oss<<";"; s=true; oss << xsv2_to_string_helper(&x) ; }
          oss<<"]";
          return oss.str();
        }
      }
    };

    template<class T, class U>
    struct AppConfigOptionState< std::map<T,U> >
    {
      static inline std::string to_string(const std::map<T,U>* state)
      {
        if(state==nullptr) return "<null>";
        else
        {
          std::ostringstream oss;
          oss<<"{ "; const char* sep="";
          for(const auto& x:*state) { oss << sep << xsv2_to_string_helper(& x.first) << ": " << xsv2_to_string_helper(& x.second); sep=" , "; }
          oss<<" }";
          return oss.str();
        }
      }
    };

    template<class T> static inline std::string xsv2_to_string_helper( const T* state )
    {
      return AppConfigOptionState<T>::to_string(state);
    }


    struct AppConfigOptionDocumentation
    {  
      template<class T=int>
      inline AppConfigOptionDocumentation(const std::string& t, const std::string& k, const std::string& d, const std::string& def = std::string{}, AppConfigOptionDocumentation* parent = nullptr , const T* state = nullptr)
        : m_type(t)
        , m_key(k)
        , m_doc(d)
        , m_default(def)
        , m_value( xsv2_to_string_helper(state) )
      {
        if( parent != nullptr )
        {
          m_parent = parent;
          m_parent->m_sub_items.push_back( this );
        }
      }

      template<class StreamT> inline StreamT& print_default_config( StreamT& out , int indent = 0) const
      {
        out << std::setfill(' ') << std::setw(indent*4) << "" << m_key << ": " << m_default << " #";
        if( m_sub_items.empty() )
        {
          out << " ("<<m_type<<")" ;
        }
        out << " "<<m_doc << std::endl;
        for(auto item:m_sub_items) item->print_default_config(out , indent+1 );
        return out;
      }

      template<class StreamT> inline StreamT& print_value( StreamT& out , int indent = 0)
      {
        out << std::setfill(' ') << std::setw(indent*4) << "" << m_key ;
        if( m_sub_items.empty() ) out << ": " << m_value;
        out << std::endl;
        for(auto item:m_sub_items) item->print_value(out , indent+1 );
        return out;
      }

      template<class StreamT> inline StreamT& print_command_line_options( StreamT& out , int level=0 , const std::string& path = std::string{} ) const
      {
        if( m_sub_items.empty() )
        {
          out << "--";
          if(!path.empty()) out<< path << "-";
          out << m_key << " <" << m_type << "> (default: "<<m_default<<")" << std::endl;      
        }
        else
        {
          std::string subpath;
          if( level>=1 ) subpath = path + m_key;
          for(auto item:m_sub_items) item->print_command_line_options(out,level+1,subpath);
        }
        return out;
      }

      std::string m_type;
      std::string m_key;
      std::string m_doc;
      std::string m_default;
      std::string m_value;
      AppConfigOptionDocumentation* m_parent = nullptr;
      std::list<AppConfigOptionDocumentation*> m_sub_items;
    };

    template<class T>
    struct AppConfigYamlConvertHelper
    {
      static inline T read(const YAML::Node& node, const T& dv)
      {
        if( node.IsNull() || !node.IsDefined() ) return dv;
        else return node.as<T>();
      }
    };
    template<class T>
    struct AppConfigYamlConvertHelper< std::vector<T> >
    {
      static inline std::vector<T> read(const YAML::Node& node, const std::vector<T>& dv)
      {
        if( node.IsNull() ) return dv;
        else if( node.IsScalar() ) return { node.as<T>() };
        else return node.as< std::vector<T> >();
      }
    };

    static inline YAML::Node yaml_sub_node(const YAML::Node & node, const std::string& k)
    {
      if( ! node.IsMap() ) return YAML::Node(YAML::NodeType::Null);
      else if( node[k] ) return node[k];
      else return YAML::Node(YAML::NodeType::Null);
    }

// end of namespaces
  }
}

#define ONIKA_APP_CONFIG_Begin(name,...) \
  struct ONIKA_APP_CONFIG_Struct_##name \
  { \
    inline ONIKA_APP_CONFIG_Struct_##name ( YAML::Node n , ::onika::app::AppConfigOptionDocumentation* parent_doc=nullptr ) : m_yaml_node( n ) \
    { if(parent_doc!=nullptr) { parent_doc->m_sub_items.push_back( &m_doc ); } } \
    YAML::Node m_yaml_node; \
    ::onika::app::AppConfigOptionDocumentation m_doc { std::string{} , #name , std::string{__VA_ARGS__} }

#define ONIKA_APP_CONFIG_Item(type,name,defval,...) \
  type name = ::onika::app::AppConfigYamlConvertHelper<type>::read( yaml_sub_node(m_yaml_node,#name) , defval ); \
  ::onika::app::AppConfigOptionDocumentation name##_doc { #type , #name , std::string{__VA_ARGS__} , #defval , & m_doc , & name }

#define ONIKA_APP_CONFIG_Node(name,...) \
  YAML::Node name = yaml_sub_node(m_yaml_node,#name); \
  ::onika::app::AppConfigOptionDocumentation name##_doc { std::string{"YAML::Node"} , #name , std::string{__VA_ARGS__} , std::string{} , & m_doc , & name }
  
#define ONIKA_APP_CONFIG_Struct(name) \
  ONIKA_APP_CONFIG_Struct_##name name { yaml_sub_node(m_yaml_node,#name) , & m_doc }

#define ONIKA_APP_CONFIG_End() }

