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
#include <exanb/core/operator_slot_base.h>
#include <exanb/core/operator.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/type_utils.h>
#include <exanb/core/string_utils.h>
#include <exanb/core/plugin.h>

#include <cassert>
#include <memory>

namespace exanb
{

  // ================================================================
  // ========================== OperatorSlotBase ========================
  // ================================================================

  OperatorSlotBase::OperatorSlotBase(const std::string& tp, SlotDirection dir, OperatorNode* owner, const std::string& name, const std::string& doc )
    : m_type(tp)
    , m_dir(dir)
    , m_owner(owner)
    , m_name(name)
    , m_doc(doc)
    {
      m_is_input_connectable = ! is_output_only();
    }

  void OperatorSlotBase::connect_forward_promote_type( OperatorSlotBase* from, OperatorSlotBase* to)
  {
    assert( to->is_input_connectable() );
    assert( from->is_output_connectable() );
    if( from->m_type != to->m_type )
    {
      if( ! OperatorSlotBase::has_type_conversion(from->m_type,to->m_type) )
      {
        lerr << "incompatible slot types: " << std::endl
             << "  from " << from->pathname() << std::endl
             << "       with type "<< pretty_short_type(from->m_type) << std::endl
             << "  to " << to->pathname() << std::endl
             << "       with type " << pretty_short_type(to->m_type) <<std::endl;
        std::abort();
      }
      else
      {
        // ldbg << "promote destination type " << remove_exanb_namespaces(strip_type_spaces(demangle_type_string(to->m_type)))
        // << " to "<<remove_exanb_namespaces(strip_type_spaces(demangle_type_string(from->m_type))) << std::endl;
        to->m_type = from->m_type;
      }
    }
  }

  bool OperatorSlotBase::is_required() const
  {
    return false;
  }

  void OperatorSlotBase::set_inout(bool input_connectable)
  {
    m_dir = INPUT_OUTPUT;
    m_is_input_connectable = input_connectable;
  }

  void OperatorSlotBase::add_output( OperatorSlotBase* to )
  {
    assert( is_output() );
    m_outputs.insert( to );
  }

  void OperatorSlotBase::remove_output( OperatorSlotBase* to )
  {
    auto it = m_outputs.find(to);
    if(it!=m_outputs.end())
    {
      m_outputs.erase(it);
    }
  }

  void OperatorSlotBase::set_input( OperatorSlotBase* from )
  {
    if( from == input() ) return;
    
    assert( is_input() );
    assert( input()==nullptr || from==nullptr );
    assert( from != this );
    m_input = from ;
  }

  void OperatorSlotBase::connect( OperatorSlotBase* from, OperatorSlotBase* to )
  {
    if( OperatorNodeFactory::debug_verbose_level() >= 2 )
    {
      ldbg << "CON: "<<from->pathname()<<"@"<<from<<" -> " <<to->pathname()<<"@"<<to<< std::endl;
    }

    if( to->input() == from )
    {
      lerr <<"Slots are already connected together"<<std::endl;
      std::abort();
    }

    if( ! to->is_input_connectable() )
    {
      lerr << "destination slot is not input connectable" << std::endl;
      std::abort();
    }
  
    if( from == to )
    {
      lerr << "cannot connet a slot to itsef" << std::endl;
      std::abort();
    }
    
    if( ! from->is_output() || ! to->is_input() )
    {
      lerr << "can't connect slots with incompatible directions (only output to input is valid)" << std::endl;
      std::abort();
    }
    
    connect_forward_promote_type( from, to );
    
    if( to->input() != nullptr )
    {
      lerr << "another slot is already connected to destination slot" << std::endl
           << "  src " << from->pathname() << " @"<<from<< std::endl
           << "  dst " << to->pathname() << " @"<<to<< std::endl
           << "  prv "<< to->input()->pathname() << " @"<<to->input()<< std::endl;
      std::abort();      
    }
    
    if( to->reachable_output(from) )
    {
      lerr << "cycle detected in slot graph" << std::endl;
      std::abort();      
    }
    
    from->add_output( to );
    to->set_input( from );
  }
    
  void OperatorSlotBase::set_resource( std::shared_ptr<OperatorSlotResource> resource )
  {
    m_resource = resource;
  }

  void OperatorSlotBase::free_resource()
  {
    if( m_resource != nullptr ) m_resource->free();
  }

  bool OperatorSlotBase::reachable_output( OperatorSlotBase* slot )
  {
    if( slot == this ) { return true; }
    for(auto o : m_outputs)
    {
      if( o->reachable_output(slot) ) { return true; }
    }
    return false;
  }

  void OperatorSlotBase::outflow_reachable_slots(std::set<const OperatorSlotBase*>& ors) const
  {
    if( ors.find(this) != ors.end() ) return;
    ors.insert(this);
    for(auto o : outputs())
    {
      o->outflow_reachable_slots( ors );
    }
    if(loop_output()!=nullptr)
    {
      loop_output()->outflow_reachable_slots( ors );
    }
    if( owner()->is_terminal() )
    {
      owner()->outflow_reachable_slots( ors );
    }
  }

  void OperatorSlotBase::remove_connections()
  {
    for(auto o:outputs()) { o->set_input(nullptr); }
    m_outputs.clear();

    if( input()!=nullptr ) input()->remove_output( this );
    set_input(nullptr);

    if( loop_output()!=nullptr ) loop_output()->set_loop_input(nullptr);
    set_loop_output(nullptr);

    if( loop_input()!=nullptr ) loop_input()->set_loop_output(nullptr);
    set_loop_input(nullptr);
  }

  std::string OperatorSlotBase::pathname() const
  {
    if( m_owner!= nullptr )
    {
      static const std::string slot_op_dir[4] = { "" , "I." , "O." , "IO."  };
      int d = 0;
      if( m_owner->in_slot_idx(this) != -1 ) { d += 1; }
      if( m_owner->out_slot_idx(this) != -1 ) { d += 2; }
      return m_owner->pathname() + ":" + slot_op_dir[d] + name();
    }
    else
    {
      return name();
    }
  }

  std::string OperatorSlotBase::backtrace()
  {
    std::string s;
    if( input() != nullptr )
    {
      s = input()->backtrace() + " -> ";
    }
    s += pathname();
    return s;
  }


  void OperatorSlotBase::print_input_stream(LogStreamWrapper& out)
  {
    if( input() != nullptr ) { input()->print_input_stream(out); }
    out << pathname() << std::endl;
  }

  // type conversion map
  std::map< std::string , std::map< std::string , std::function<void*(void*)> > > OperatorSlotBase::s_type_conversion;
  std::list< std::pair< std::pair<std::string,std::string> , std::function<void*(void*)> > > OperatorSlotBase::s_type_conversion_delayed;
  bool OperatorSlotBase::s_registration_enabled = false;

  void OperatorSlotBase::register_type_conversion_internal( const std::string& s, const std::string& d, std::function<void*(void*)> addr_converter)
  {
    std::function<void*(void*)> f = addr_converter;
#   ifndef NDEBUG
    f = [addr_converter,s,d](void*p)->void*
    {
      void* dp = nullptr;
      if( addr_converter != nullptr ) { dp = addr_converter(p); }
      else { dp = p; }
      ldbg<<"converting "<<pretty_short_type(s)<<" @"<<p<<" to "<<pretty_short_type(d)<<" @"<<dp<<std::endl;
      return dp;
    };
#   endif
  
    if( ! s_registration_enabled )
    {
      s_type_conversion_delayed.push_back( { {s,d} , f } );
    }
    else
    {
      // ldbg << "register conversion from "<<remove_exanb_namespaces(strip_type_spaces(demangle_type_string(s)))<<" to "<<remove_exanb_namespaces(strip_type_spaces(demangle_type_string(d)))<<std::endl;
      if( ! exanb::quiet_plugin_register() ) { lout << "  conversion  "<<pretty_short_type(s)<<" -> "<<pretty_short_type(d)<<std::endl; }
      s_type_conversion[ s ] [ d ] = f;
    }
  }

  void OperatorSlotBase::enable_registration()
  {
    auto l = std::move( s_type_conversion_delayed );
    s_registration_enabled = true;
    for(auto& conv:l) { register_type_conversion_internal( conv.first.first , conv.first.second , conv.second ); }
  }

  bool OperatorSlotBase::has_type_conversion(const std::string& s, const std::string& d)
  {
      auto it_src_type = s_type_conversion.find( s );
      if( it_src_type == s_type_conversion.end() ) { return false; }
      auto it = it_src_type->second.find( d );
      if( it == it_src_type->second.end() ) { return false; }  
      return true;
  }

/*
  void OperatorSlotBase::acquire()
  {
    if( ! m_resource->is_null() )
    {
      // ldbg << "acquire " << pathname() << " (ptr="<<m_resource->memory_ptr()<<")"<<std::endl;
      m_resource->acquire();
    }
  }
  
  void OperatorSlotBase::acquire_read_only()
  {
    if( ! m_resource->is_null() )
    {
      // ldbg << "acquire_read_only " << pathname() << " (ptr="<<m_resource->memory_ptr()<<")"<<std::endl;
      m_resource->acquire_read_only();
    }
  }
  
  void OperatorSlotBase::release()
  {
    if( ! m_resource->is_null() )
    {
      // ldbg << "release " << pathname() << " (ptr="<<m_resource->memory_ptr()<<") ";
      m_resource->release();
      //ldbg << std::endl;
    }
  }
*/

}

