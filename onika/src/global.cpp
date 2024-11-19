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
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>
#include <onika/log.h>
#include <onika/string_utils.h>

#include <memory>

namespace onika { namespace scg
{

  // ================= builtin factories implementation ======================

  template<typename T>
  static inline bool try_convert(YAML::Node node, T& val)
  {
    bool conversion_done = false;
    try
    {
      val = node.as<T>();
      conversion_done = true;
    }
    catch(const YAML::Exception&){ conversion_done = false; }
    return conversion_done;
  }

  class GlobalVariableOperatorNode : public OperatorNode
  {
  public:
    inline void execute() override final {}
  };

  // global operator node. used as a placeholder for global constant values
  std::shared_ptr<OperatorNode> make_global_operator(const YAML::Node& node, const OperatorNodeFlavor& flavor)
  {
    std::shared_ptr<OperatorNode> op = std::make_shared<GlobalVariableOperatorNode>();
    if( node.IsMap() )
    {
      for( YAML::const_iterator it=node.begin(); it!=node.end(); ++it )
      {
        std::string key = it->first.as<std::string>();
        if( ! ( it->second.IsScalar() || it->second.IsSequence() ) )
        {
          lerr << "In global, value for '"<<key<<"' is not a scalar nor a sequence in following map :\n";
          onika::yaml::dump_node_to_stream( lerr , node );
          std::abort();
        }
        
        std::string scalar_str = "";
        try { scalar_str = node.as<std::string>(); }
        catch(const YAML::Exception&){ scalar_str = ""; }
                
        if( scalar_str == "null_bool" )
        {
          // ldbg << key << "is null bool" << std::endl;
          make_operator_slot<bool>( op.get() , key , OUTPUT );
        }
        else if( scalar_str == "null_double" )
        {
          // ldbg << key << "is null double" << std::endl;
          make_operator_slot<double>( op.get() , key , OUTPUT );
        }
        else
        {
          bool conversion_found = true;
          if( it->second.IsScalar() )
          {
            bool bval;
            long lval;
            double dval;
            onika::physics::Quantity qval;
            std::string sval;
            if( try_convert(it->second,bval) )      { make_operator_slot<bool  >( op.get() , key , OUTPUT )->set_resource_default_value( bval ); }
            else if( try_convert(it->second,lval) ) { make_operator_slot<long  >( op.get() , key , OUTPUT )->set_resource_default_value( lval ); }
            else if( try_convert(it->second,dval) ) { make_operator_slot<double>( op.get() , key , OUTPUT )->set_resource_default_value( dval ); }
            else if( try_convert(it->second,qval) )
            {
              dval = qval.convert();
              make_operator_slot<double>( op.get() , key , OUTPUT )->set_resource_default_value( dval );
            }
            else if( try_convert(it->second,sval) ) { make_operator_slot<std::string>( op.get() , key , OUTPUT )->set_resource_default_value( sval ); }
            else { conversion_found=false; }
          }
          else if( it->second.IsSequence() )
          {
            std::vector<bool> bval;
            std::vector<long> lval;
            std::vector<double> dval;
            std::vector<onika::physics::Quantity> qval;
            std::vector<std::string> sval;
            if( try_convert(it->second,bval) )      { make_operator_slot< std::vector<bool>  >( op.get() , key , OUTPUT )->set_resource_default_value( bval ); }
            else if( try_convert(it->second,lval) ) { make_operator_slot< std::vector<long>  >( op.get() , key , OUTPUT )->set_resource_default_value( lval ); }
            else if( try_convert(it->second,dval) ) { make_operator_slot< std::vector<double> >( op.get() , key , OUTPUT )->set_resource_default_value( dval ); }
            else if( try_convert(it->second,qval) )
            {
              for(const auto& q:qval) dval.push_back( q.convert() );
              make_operator_slot< std::vector<double> >( op.get() , key , OUTPUT )->set_resource_default_value( dval );
            }
            else if( try_convert(it->second,sval) ) { make_operator_slot< std::vector<std::string> >( op.get() , key , OUTPUT )->set_resource_default_value( sval ); }
            else { conversion_found=false; }          
          }
          
          if( conversion_found )
          {
            ldbg << key << " -> "<< op->out_slot(key)->value_type() << std::endl;
          }
          else
          {
            fatal_error() << "no suitable type found for global key "<<key<<std::endl;
          }
        }
      }
    }
    op->set_profiling(false);
    return op;
  }

  // === register factories ===  
  ONIKA_AUTORUN_INIT(global)
  {
    OperatorNodeFactory::instance()->register_factory("global",make_global_operator);
  }

} }

