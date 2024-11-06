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
#include <exanb/core/operator_factory.h>
#include <exanb/core/operator_slot_base.h>
#include <exanb/core/operator_slot.h>
#include <onika/type_utils.h>
#include <onika/string_utils.h>
#include <exanb/core/quantity.h>
#include <onika/physics/units.h>
#include <exanb/core/quantity_stream.h>

#include <exanb/core/log.h>
#include <exanb/core/plugin.h>

#include <cassert>
#include <iostream>
#include <memory>

namespace exanb
{
  OperatorNodeFactory* OperatorNodeFactory::s_instance = nullptr;
  int OperatorNodeFactory::s_debug_verbose_level = 0;

  OperatorNodeFactory* OperatorNodeFactory::instance()
  {
    if( s_instance == nullptr )
    {
      s_instance = new OperatorNodeFactory();
    }
    return s_instance;
  }

  /*!
    Register a factory (Operator creaton function), associating it to an operator name
  */
  size_t OperatorNodeFactory::register_factory( const std::string& name, OperatorNodeCreateFunction creator )
  {
    if( ! m_registration_enabled )
    {
      m_defered_creators_to_register.push_back( {name,creator} );
      return 0;
    }
  
    if( ! exanb::quiet_plugin_register() && m_creators.find(name) == m_creators.end() )
    {
      lout<<"  operator    "<< name << std::endl;
      plugin_db_register( "operator" , name );
    }

    if( m_creators.find(name) != m_creators.end() )
    {
      ldbg<<"  overload    "<< name << std::endl;
    }
  
    m_creators.insert( CreatorPair(name,creator) );
    return m_creators.size();
  }

  void OperatorNodeFactory::enable_registration()
  {
    m_registration_enabled = true;
    auto l = std::move(m_defered_creators_to_register);
    for(auto& f:l) { register_factory( f.first , f.second ); }
  }

  void OperatorNodeFactory::set_operator_defaults(YAML::Node node)
  {
    m_operator_defaults = node;
  }

  std::set<std::string> OperatorNodeFactory::available_operators()
  {
    std::set<std::string> ops;
    for(auto it : m_creators) { ops.insert( it.first ); }
    return ops;
  }

  bool OperatorNodeFactory::find_operator_defaults( std::string& target_proto , YAML::Node& node, std::map<std::string,std::string>& locals )
  {
    using std::string;
    using std::vector;
    
    string target_name;
    vector<string> target_args;
    
    function_name_and_args( target_proto , target_name, target_args );
    size_t nargs = target_args.size();

    for (auto p : m_operator_defaults)
    {
      if( p.first.IsScalar() )
      {
        std::string proto = p.first.as<std::string>();
        string name;
        vector<string> args;
        function_name_and_args( proto , name, args );
        if( name == target_name )
        {
          // ldbg << "found matching default for macro "<<name<<std::endl;
          if( args.size() != nargs )
          {
            lerr << "macro "<<name<<"used with wrong number of arguments, expected "<<args.size()<<", got "<<nargs;
            std::abort();
          }
          locals.clear();
          for(size_t i=0;i<nargs;i++)
          {
            // ldbg << "mapping local "<<args[i]<<" to "<<target_args[i]<<std::endl;
            locals[ args[i] ] = target_args[i];
          }
          target_proto = proto;
          node = YAML::Clone(p.second);
          return true;
        }
      }
    }
    
    return false;
  }

  std::string OperatorNodeFactory::resolve_operator_name(const std::string& proto)
  {
    using std::string;
    using std::vector;

    string name;
    vector<string> args;
    function_name_and_args( proto , name, args );
    // ldbg << proto << " -> "<<name<<" ("; for(auto x:args){ldbg<<" "<<x;} ldbg << " )" << std::endl;

    ssize_t stack_size = m_locals_stack.size();
    //ldbg << "locals stack size = "<< stack_size << std::endl;
    
    for(ssize_t i=stack_size-1; i>=0; --i)
    {
      //ldbg << "scan locals :";
      //for(auto p: m_locals_stack[i]) { ldbg << " " << p.first << "->" << p.second ; }
      //ldbg << std::endl;
      
      if( m_locals_stack[i].find( name ) != m_locals_stack[i].end() ) { name = m_locals_stack[i][name] ; }
      for(auto& arg: args)
      {
        if( m_locals_stack[i].find( arg ) != m_locals_stack[i].end() ) { arg = m_locals_stack[i][arg] ; }
      }
    }
    
    string f = name;
    if( ! args.empty() )
    {
      f += "(";
      for(size_t i=0;i<args.size();i++) { if(i>0) { f+=","; } f+=args[i]; }
      f += ")";
    }
    
    /*
    if( f != proto )
    {
      ldbg << proto << " resolved to "<< f << std::endl;
    }
    */
    
    return f;
  }

  /*!
    Asks factory to build an operator given an operator name, a yaml node and an operator flavor
  */
  std::shared_ptr<OperatorNode> OperatorNodeFactory::make_operator(const std::string& in_name, YAML::Node node, const OperatorNodeFlavor& in_flavor)
  {  
    std::string name = resolve_operator_name(in_name);
    if( s_debug_verbose_level >= 1 )
    {
      ldbg << "make_operator "<<in_name;
      if( name != in_name ) { ldbg << " (renamed to "<<name<<")"; }
      ldbg << " ("<<m_creators.count(name)<<" candidates)" <<std::endl;
    }
   
    // store some special attribute to flavor map, not clean but useful
    OperatorNodeFlavor flavor = in_flavor;
    if( ! flavor["__operator_name__"].empty() ) { flavor["__operator_name__"] += "."; }
    flavor["__operator_name__"] += name;
    
    std::map<std::string,std::string> locals;

    if( node.IsNull() )
    {
      find_operator_defaults(name, node, locals);
    }
    
    auto it_range = m_creators.equal_range( name );

    std::ostringstream err_mesg;

    // first check if operator name is an alias for another one
    if( it_range.first == m_creators.end() )
    {
      bool alias_found = false;      
      do
      {
        alias_found = false;      
        for (auto p : m_operator_defaults)
        {
          if( p.first.IsScalar() && p.second.IsScalar() && p.first.as<std::string>()==name )
          {
            ldbg << "operator alias '" << name << "' replaced with '"<<p.second.as<std::string>()<<"'"<<std::endl;
            name = p.second.as<std::string>();
            it_range = m_creators.equal_range( name );
            if( m_operator_defaults[name] )
            {
              ldbg << "\t'"<<name<<"' has a default definition, using it" <<std::endl;
              node = m_operator_defaults[name];
            }
            alias_found = true;
            break;
          }
        }
      } while( alias_found );
    }

    // tells if an operator, when not found in any factory, is allowed to be considered as an implicit batch operator
    if( it_range.first == m_creators.end() )
    {
      std::string suggested_plugin = suggest_plugin_for( "operator" , name );
      if( ! suggested_plugin.empty() )
      {
        ldbg << "auto loading "<< suggested_plugin<<" to find operator "<<name<< std::endl;
        load_plugins( { suggested_plugin } );
        it_range = m_creators.equal_range( name );
        if( it_range.first == m_creators.end() )
        {
          err_mesg<<"No candidate available for operator '"<<name<<"' in plugin '"<<suggested_plugin<<"'"<< std::endl;
          throw OperatorCreationException( err_mesg.str() );
        }
      }
    }
    
    if( it_range.first == m_creators.end() )
    {
      // if operator name is not known, it indicates it's implicitly a batch operator.
      if( OperatorNodeFactory::debug_verbose_level() >= 2 )
      {
        ldbg << name << " is implicitly a batch operator" << std::endl;
      }
      if( ! node.IsNull() )
      {
        it_range = m_creators.equal_range( "batch" );
        if( it_range.first == m_creators.end() ) // we have to find plugin containing the batch operator factory
        {
          std::string suggested_plugin = suggest_plugin_for( "operator" , "batch" );
          if( ! suggested_plugin.empty() )
          {
            ldbg << "auto loading "<< suggested_plugin<<" to find operator 'batch'" << std::endl;
            load_plugins( { suggested_plugin } );
            it_range = m_creators.equal_range( "batch" );
          }
        }
        if( it_range.first == m_creators.end() )
        {
          err_mesg << "Internal error: can't find operator 'batch' in factory for implicit batch '"<<name<<"'" << std::endl;
          throw OperatorCreationException( err_mesg.str() );
        }
      }
      else
      {
        err_mesg <<"Could not find a operator factory for '"<<name<<"' (and no default declaration exists for this name)"<<std::endl;
        throw OperatorCreationException( err_mesg.str() );
      }
    }

    // add local name aliases to current local names stack
    if( ! locals.empty() )
    {
      m_locals_stack.push_back( locals );
    }

    int n_attempts = 0;
    err_mesg << "Factory was unable to create operator '"<<name<<"'"<<std::endl;

    std::shared_ptr<OperatorNode> op;
    for( auto it = it_range.first ; op == nullptr && it != it_range.second ; ++it )
    {
      try
      {
        //std::cout<<"\nattempt to build operator '"<<name<<"' from node :\n";
        //dump_node_to_stream( std::cout , node );
        //std::cout<<std::endl;
        YAML::Node ncopy = YAML::Clone( node );
        op = it->second ( ncopy , flavor );
        //std::cout<<"Ok"<<std::endl;
      }
      catch( const OperatorCreationException& e )
      {
        ++ n_attempts;
        err_mesg << "Attempt "<< n_attempts <<" failed for the following reason :" << std::endl;
        err_mesg << "===========================================" << std::endl;
        err_mesg << str_indent( e.what() , 2 , ' ' , "| " );
      }
    }
    
    if( ! locals.empty() )
    {
      //ldbg << "pop locals" << std::endl;
      m_locals_stack.pop_back();
    }

    if( op == nullptr )
    {
      throw OperatorCreationException( err_mesg.str() );
    }
        
    if( op->name().empty() )
    {
      op->set_name( name );
    }
    
    // populate slot values from YAML
    op->yaml_initialize( node );
    
    // finalize inner resources, knowing that slots won't be modified anymore
    if( ! op->compiled() )
    {
      op->compile();
    }
    
    return op;
  }

  // ===================== utility functions ==================

  bool check_operator_compatibility( std::shared_ptr<OperatorNode> op, const OperatorNodeFlavor& f2, OperatorNodeIncompatibility& incompatibility )
  {
    for( auto& s : op->in_slots() )
    {
      const std::string& k = s.first;
      const std::string& t = s.second->value_type();
      if( s.second->is_input() && s.second->is_input_connectable() )
      {
        auto it = f2.find( k );
        if( it != f2.end() )
        {
          if( t != it->second )
          {
            if( ! OperatorSlotBase::has_type_conversion( it->second , t ) )
            {
              incompatibility.slot_name = k;
              incompatibility.slot_type = t;
              incompatibility.flavor_type = it->second;
              return false;
            }
          }
        }
      }
    }
    return true;
  }

}

