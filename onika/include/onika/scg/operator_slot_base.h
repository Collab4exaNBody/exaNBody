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

#include <onika/log.h>
#include <onika/type_utils.h>
#include <onika/scg/operator_slot_direction.h>
#include <onika/scg/operator_slot_resource.h>

#include <set>
#include <string>
#include <map>
#include <memory>
#include <type_traits>
#include <cstdlib>
#include <cassert>
#include <cstring>

#include <yaml-cpp/yaml.h>

namespace onika { namespace scg
{

  // helper class to generate default address converter functor
  template<typename SourceT, typename DestT, bool = std::is_convertible_v<SourceT*,DestT*> >
  struct DefaultAddressConverter
  {
    static inline std::function<void*(void*)> func()
    { return
      [](void* p)->void*
      {
        void* np = static_cast<DestT*>( reinterpret_cast<SourceT*>(p) );
        assert(np==p);
        return np;
      };
    }
  };
  template<typename SourceT, typename DestT> struct DefaultAddressConverter<SourceT,DestT,false> { static inline std::function<void*(void*)> func(){ return nullptr; } };

  struct OperatorNode;

  class alignas(16) OperatorSlotBase
  {
  public:

    // constructor  
    OperatorSlotBase(const std::string& tp, SlotDirection dir, OperatorNode* owner, const std::string& name, const std::string& doc=std::string() );
  
    // ====== virtual API ===========
    virtual ~OperatorSlotBase() = default;
    virtual void yaml_initialize(const YAML::Node& node) =0;
    virtual size_t memory_bytes() const =0;
    virtual void initialize_resource_pointer() =0;
    virtual bool has_value() const =0; // returns true only if value stored in typed pointer cache
    virtual void set_required(bool r) =0;
    virtual bool is_required() const;
    virtual std::string value_as_string() =0;
    virtual bool value_as_bool() =0;
    virtual std::shared_ptr<OperatorSlotBase> new_instance(OperatorNode* opnode, const std::string& k, SlotDirection dir) =0;
    // ================================

    // slot state retreival        
    inline bool is_input() const { return m_dir==INPUT || m_dir==INPUT_OUTPUT; }
    inline bool is_input_only() const { return m_dir==INPUT; }
    inline bool is_output() const { return m_dir==OUTPUT || m_dir==INPUT_OUTPUT; }
    inline bool is_output_only() const { return m_dir==OUTPUT; }
    void set_inout(bool input_connectable=true);
     
    inline SlotDirection direction() const { return m_dir; }
    inline const std::string& value_type() const { return m_type; }
    inline void rename(const std::string& s) { m_name = s; }
    inline const std::string& name() const { return m_name; }
    std::string pathname() const;
    std::string backtrace();
    inline const std::string& documentation() const { return m_doc; }
    std::shared_ptr<OperatorSlotResource> resource() const { return m_resource; }

    // slot connection
    static void connect( OperatorSlotBase* from, OperatorSlotBase* to );
    void add_output( OperatorSlotBase* to );
    void remove_output( OperatorSlotBase* to );
    void set_input( OperatorSlotBase* from );
    inline OperatorSlotBase* input() const { return m_input; }
    const std::set<OperatorSlotBase*> & outputs() const { return m_outputs; }
    virtual void reset_input() =0;

    // slot loop connections ( when a batch loops, it's ouputs may be connected to its inputs )
    inline OperatorSlotBase* loop_input() const { return m_loop_input; }
    inline OperatorSlotBase* loop_output() const { return m_loop_output; }
    inline void set_loop_output(OperatorSlotBase* lo) { m_loop_output=lo; }
    inline void set_loop_input(OperatorSlotBase* li) { m_loop_input=li; }

    // outflow graph discovery
    void outflow_reachable_slots(std::set<const OperatorSlotBase*>& ors) const;

    // asynchronous resource acquisition and release    
    // void acquire();
    // void acquire_read_only();
    // void release();

    // maps source type -> ( map dest type -> conversion function )    
    template<typename SourceT, typename DestT >
    static inline void register_type_conversion( std::function<void*(void*)> addr_converter = DefaultAddressConverter<SourceT,DestT>{}.func() )
    {
      register_type_conversion_internal( typeid(SourceT).name(), typeid(DestT).name(), addr_converter );
    }

    template<typename SourceT, typename DestT >
    static inline void register_type_conversion_force_cast()
    {
      register_type_conversion_internal( typeid(SourceT).name(), typeid(DestT).name(), [](void* p)->void*{return p;} );
    }


    static bool has_type_conversion(const std::string& s, const std::string& d);
    static void enable_registration();

    // slot graph traversal
    void set_resource( std::shared_ptr<OperatorSlotResource> resource );
    void free_resource();
    bool reachable_output( OperatorSlotBase* slot );
    
    // slot allow input connection. this is false when user provided a fixed value
    inline bool is_input_connectable() const { return m_is_input_connectable; }
    inline void set_input_connectable(bool ic) { m_is_input_connectable = ic; }

    inline bool is_output_connectable() const { return m_is_output_connectable; }
    inline void set_output_connectable(bool oc) { m_is_output_connectable = oc; }
    
    inline bool is_private() { return !is_input_connectable() && !is_output_connectable(); }

    // owning OperatorNode
    inline OperatorNode* owner() const { return m_owner; }

    // pretty printing
    void print_input_stream(LogStreamWrapper& out);
    
    // not used, just informational. tells if slot drives conditional execution
    inline void set_conditional_input(bool b) { m_is_condition_slot=b; }
    bool is_conditional_input() const { return m_is_condition_slot; }
    
    void remove_connections();
    
  protected:    
    // helper function to promote type when conversion is available
    static void connect_forward_promote_type(OperatorSlotBase* from, OperatorSlotBase* to);

    // slot data type
    std::string m_type;
    
    // slot direction
    SlotDirection m_dir = INPUT_OUTPUT;

    // slot graph connections
    OperatorSlotBase* m_input = nullptr;
    OperatorSlotBase* m_loop_input = nullptr;
    std::set< OperatorSlotBase* > m_outputs;
    OperatorSlotBase* m_loop_output = nullptr;
 
    // slot identity
    OperatorNode* const m_owner;
    std::string m_name;
    const std::string m_doc;
  
    // resource (data) associated to this slot
    std::shared_ptr<OperatorSlotResource> m_resource = std::make_shared<OperatorSlotResource>(nullptr);
  
    // is this slot allowed to be connected to another slot's output
    bool m_is_input_connectable = true; // replaces is_overridable previously in OperatorSlotResource
    bool m_is_output_connectable = true;
    
    // tells if slot is used to condition execution of an operator
    bool m_is_condition_slot = false;
  
    static void register_type_conversion_internal( const std::string& s, const std::string& d, std::function<void*(void*)> addr_converter);
    static std::map< std::string , std::map< std::string , std::function<void*(void*)> > > s_type_conversion;

    static std::list< std::pair< std::pair<std::string,std::string> , std::function<void*(void*)> > > s_type_conversion_delayed;
    static bool s_registration_enabled;
  };


} }


