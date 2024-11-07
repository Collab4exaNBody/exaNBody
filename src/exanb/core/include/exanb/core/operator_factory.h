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

#include <exanb/core/operator.h>
#include <onika/log.h>
#include <onika/type_utils.h>
#include <onika/cpp_utils.h>

#include <yaml-cpp/yaml.h>

#include <memory>
#include <string>
#include <map>
#include <unordered_map>
#include <exception>

namespace exanb
{
  // to be specialized
  template<class T>
  struct OperatorNodeFactoryGenerator
  {
    static inline OperatorNodeCreateFunction make_factory() { return nullptr; }
  };

  // an operator flavor is a set of prefereed types corresponding to slot names
  class OperatorCreationException : public std::exception
  {
  public:
    inline OperatorCreationException(const std::string& mesg) noexcept : m_mesg(mesg) {}
    inline const char* what() const noexcept { return m_mesg.c_str(); }
  private:
    std::string m_mesg;
  };

  class OperatorNodeFactory
  {
  public:
    size_t register_factory( const std::string& name, OperatorNodeCreateFunction factory );    
    template<class T> 
    inline size_t register_factory( const std::string& name, const OperatorNodeFactoryGenerator<T>& factory_gen )
    {
      OperatorNodeCreateFunction factory = factory_gen.make_factory( name );
      if( factory != nullptr ) return this->register_factory(name,factory);
      else return m_creators.size();
    }

    std::shared_ptr<OperatorNode> make_operator( const std::string& name, YAML::Node node = YAML::Node(), const OperatorNodeFlavor& flavor = OperatorNodeFlavor() );
    std::string resolve_operator_name(const std::string& proto);
    void set_operator_defaults(YAML::Node node);
    std::set<std::string> available_operators();

    void enable_registration();
    
    static OperatorNodeFactory* instance();
    static inline int debug_verbose_level() { return s_debug_verbose_level; }
    static inline void set_debug_verbose_level(int l) { s_debug_verbose_level = l; }

  private:

    bool find_operator_defaults( std::string& target_proto , YAML::Node& node, std::map<std::string,std::string>& locals );
    
    using CreatorPair = std::pair<std::string, OperatorNodeCreateFunction>;
    std::list< CreatorPair > m_defered_creators_to_register;
    std::unordered_multimap< std::string, OperatorNodeCreateFunction > m_creators;
    YAML::Node m_operator_defaults;
    std::vector< std::map<std::string,std::string> > m_locals_stack;
    bool m_registration_enabled = false;

    static OperatorNodeFactory* s_instance;
    static int s_debug_verbose_level;
  };

  struct OperatorNodeIncompatibility
  {
    std::string slot_name;
    std::string slot_type;
    std::string flavor_type;
  };

  // return true if slots with identical names have the same type
  // remarks :
  // should it check only input to output connections ?
  // should slot which have value provided in YAML not be checked ?
  bool check_operator_compatibility( std::shared_ptr<OperatorNode> op, const OperatorNodeFlavor& f2, OperatorNodeIncompatibility & incompatibility );

  // ================== utility functions to help creation method implementation ======================

  namespace details
  {
    template<typename OperatorType>
    static inline std::shared_ptr<OperatorNode>
    make_operator_if_compatible(
      std::shared_ptr<OperatorNode> in_op,
      const YAML::Node& node,
      const OperatorNodeFlavor& flavor,
      std::vector<OperatorNodeIncompatibility>& incompatibilities )
    {  
      // if a compatible operator has already been found, simply return it
      if( in_op != nullptr ) { return in_op; }

      // really instanciate the operator
      std::shared_ptr<OperatorNode> op = std::make_shared<OperatorType>();

      // remember how to instanciate such a type later on
      // op->set_new_instance( []() -> std::shared_ptr<OperatorNode> { return std::make_shared<OperatorType>(); } );
      
      // surnumerous call to yaml_intialize here is harmless (the relevant one is in OperatorNodeFactory::make_operator)
      // its purpose is to populate slot values, so that only slots without values will be tested against flavor
      // Note: a user given value is seen as an incoming value connected to an input slot,
      // thus input slot does not need to match type of existing output slots with identical name
      op->yaml_initialize(node);
      
      OperatorNodeIncompatibility incompatibility;
      if( check_operator_compatibility( op , flavor, incompatibility ) )
      {
        if( OperatorNodeFactory::debug_verbose_level() >= 2 )
        {
          ldbg << "select operator "<< onika::pretty_short_type<OperatorType>() << std::endl;
        }
        return op;
      }
      else
      {
        if( OperatorNodeFactory::debug_verbose_level() >= 3 )
        {
          ldbg << "reject operator "<< onika::pretty_short_type<OperatorType>() << std::endl;   
        }
        incompatibilities.push_back( incompatibility );
        return nullptr;
      }
    }
  }
  
  // select an operator among several, depending on which has compatible slots with given flavor
  template<typename... OperatorType>
  static inline std::shared_ptr<OperatorNode> make_compatible_operator( const YAML::Node& node, const OperatorNodeFlavor& flavor )
  {
    std::vector<OperatorNodeIncompatibility> incompatibilities;
    std::shared_ptr<OperatorNode> op = nullptr;
    (...,( op = details::make_operator_if_compatible<OperatorType>( op, node, flavor, incompatibilities )  ));
    if( op == nullptr )
    {
      std::ostringstream oss;
      oss << "Couldn't find a compatible operator. flavor was :" << std::endl;
      for(auto& p : flavor)
      {
        oss<< '\t' << p.first << " -> " << onika::pretty_short_type(p.second) << std::endl;
      }
      oss << "Available factories failed for the following reasons :" << std::endl;      
      for(const OperatorNodeIncompatibility& i : incompatibilities)
      {
        oss<< "\tslot " << i.slot_name << " with type " << onika::pretty_short_type(i.slot_type) << " incompatible with flavor type " << onika::pretty_short_type(i.flavor_type) << std::endl;
      }
      throw OperatorCreationException( oss.str() );
    }
    return op;
  }

  template<typename OperatorType>
  static inline std::shared_ptr<OperatorNode> make_simple_operator( const YAML::Node& node, const OperatorNodeFlavor& flavor )
  {
    return std::make_shared<OperatorType>();
  }

}


