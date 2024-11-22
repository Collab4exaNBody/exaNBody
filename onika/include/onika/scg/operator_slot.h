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

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot_base.h>
#include <onika/type_utils.h>
#include <onika/print_utils.h>
#include <onika/yaml/yaml_utils.h>

#include <onika/memory/memory_usage.h>
#include <onika/dac/dac.h>
#include <onika/lambda_tools.h>

#include <sstream>
#include <utility>
#include <tuple>
#include <type_traits>
#include <optional>
#include <ios>

namespace onika { namespace scg
{

  // forward declaration of helper function
  template<typename T, bool IsInputOnly=false, bool HasYAMLConversion = onika::yaml::is_yaml_convertible_v<T> > class OperatorSlot;
  template<typename T>
  inline std::shared_ptr< OperatorSlot<T> > make_operator_slot( OperatorNode* opnode, const std::string& k, SlotDirection d );

} }

// stream helper operator, so that slot can be output to a std:ostream as is
template<class T>
inline std::ostream& operator << ( std::ostream& out , const onika::scg::OperatorSlot<T>& slot )
{
  return out << slot.value_as_string();
}

namespace onika { namespace scg 
{

  // set of DataSlicesSubSet (mind the final s)
  template<typename... DS> struct DataAccessPattern
  {
    static inline std::vector<uint64_t> dac_masks() { return { (DS::bit_mask_v) ... }; }
  };

  // builds up a DataAccessPatterns<N>
  template<class... SlicesSubSetT>
  static inline auto dap( SlicesSubSetT ... )
  {
    return DataAccessPattern< SlicesSubSetT ... >{};
  }

  // ================== internal gory details no one wants to see ===================================
  namespace operator_slot_details
  {
    template<class T> struct IsDAC : public std::false_type {};
    template<class... T> struct IsDAC< DataAccessPattern<T...> > : public std::true_type {};
    template<class T> static inline constexpr bool is_dac_v = IsDAC<T>::value ;
  
    // utility templates to construct T from a set of arguments
    template<class T, class Args> struct is_constructible_from_tuple : std::false_type {};
    template<class T, class... Args> struct is_constructible_from_tuple< T , std::tuple<Args...> > : std::integral_constant<bool,std::is_constructible<T,Args...>::value> {};
    template<class T,class TupleT> static inline constexpr bool is_constructible_from_tuple_v = is_constructible_from_tuple<T,TupleT>::value;

    template<class T, class Args,
             class = std::make_index_sequence< std::tuple_size<Args>::value >,
             class = std::enable_if_t< is_constructible_from_tuple_v<T,Args> , void >
             >
    struct ConstructFromTuple {};

    template<class T, class Args, size_t... S> struct ConstructFromTuple< T, Args, std::integer_sequence<size_t,S...> , void >
    {
      static inline T* New( const Args& cargs ) { return new T( std::get<S>(cargs) ... ); }
    };

    template<typename... T> struct SlotConstructArgs;

    template<> struct SlotConstructArgs<>
    {
      using DocString = typename OperatorNode::DocString;
      static constexpr bool is_required_v = false;
      static constexpr bool is_optional_v = false;
      static constexpr bool is_private_v = false;
      using default_value_t = std::nullopt_t;
      using dap_t = DataAccessPattern<>;

      static inline DocString m_empty_doc{};
      static inline const DocString & doc() { return m_empty_doc; }
      static inline SlotDirection dir() { return INPUT_OUTPUT; } // compatible with PRIVATE declaration without any given direction
      static inline const default_value_t& defval() { return std::nullopt; }
    };

    template<class T> const T& undefined_value_reference()
    {
      static char undefined_bytes[sizeof(T)];
      return * ((T*)undefined_bytes);
    }

    template<typename T0, typename... T> struct SlotConstructArgs<T0,T...>
    {
      using DocString = typename OperatorNode::DocString;

      static inline constexpr bool is_optional_v = std::is_same_v<T0,typename OperatorNode::OPTIONAL_t> || SlotConstructArgs<T...>::is_optional_v;
      static inline constexpr bool is_private_v  = std::is_same_v<T0,typename OperatorNode::PRIVATE_t > || SlotConstructArgs<T...>::is_private_v;
      static inline constexpr bool is_required_v = is_private_v || std::is_same_v<T0,typename OperatorNode::REQUIRED_t> || SlotConstructArgs<T...>::is_required_v;

      static inline constexpr bool T0_known_type = std::is_same_v<T0,typename OperatorNode::REQUIRED_t>
                                         || std::is_same_v<T0,typename OperatorNode::OPTIONAL_t>
                                         || std::is_same_v<T0,typename OperatorNode::PRIVATE_t>
                                         || std::is_same_v<T0,DocString>
                                         || std::is_same_v<T0,SlotDirection>
                                         || is_dac_v<T0>;

      using dap_t = std::conditional_t< is_dac_v<T0> , T0 , typename SlotConstructArgs<T...>::dap_t >;

      using default_value_t = std::conditional_t< !T0_known_type , T0 , typename SlotConstructArgs<T...>::default_value_t > ;
      template<typename U> static inline constexpr bool is_constructible_single_arg_v() { return std::is_constructible_v<U,default_value_t>; }
      template<typename U> static inline constexpr bool is_constructible_tuple_arg_v() { return is_constructible_from_tuple_v<U,default_value_t>; }
      template<typename U> static inline constexpr bool is_constructible_void_arg_v() { return std::is_same_v<default_value_t,std::nullopt_t> && std::is_default_constructible_v<U>; }
//      template<typename U> static inline constexpr bool is_constructible_from_default_v() { return is_constructible_single_arg_v<U>() || is_constructible_tuple_arg_v<U>() || is_constructible_void_arg_v<U>(); }

      static inline constexpr bool has_provided_default_value = ! std::is_same_v<default_value_t,std::nullopt_t>;

      static_assert( ! (is_required_v && is_optional_v) , "inconsistent OperatorSlot constructor arguments" );
      
      static inline const DocString& doc(const T0& a0, const T& ... args)
      {
        if constexpr ( std::is_same_v<T0,typename OperatorNode::DocString> ) { return a0; }
        else { return SlotConstructArgs<T...>::doc(args...); }
        return undefined_value_reference<DocString>(); // can never get there, warning (when this line is missing) is a compiler bug
      }
      
      static inline SlotDirection dir(const T0& a0, const T& ... args)
      {
        if constexpr ( std::is_same_v<T0,SlotDirection> ) { return a0; }
        else { return SlotConstructArgs<T...>::dir(args...); }
        return INPUT_OUTPUT;
      }

      static inline const default_value_t& defval(const T0& a0, const T& ... args)
      {
        if constexpr ( !T0_known_type ) { return a0; }
        if constexpr (  T0_known_type ) { return SlotConstructArgs<T...>::defval(args...); }
        return undefined_value_reference<default_value_t>(); // can never get there, warning (when this line is missing) is a compiler bug
      }
      
      inline SlotConstructArgs(const T0& a0, const T& ... args )
        : m_doc( doc(a0,args...) )
        , m_defval( defval(a0,args...) )
        , m_dir( dir(a0,args...) )
      {}

      template<typename U>
      std::function<void*()> build_default_allocator( TypePlaceHolder<U> ) const
      {
        if constexpr ( is_constructible_single_arg_v<U>() )
        {
          return [carg=m_defval]() -> void* { return new U(carg); };
        }
        if constexpr ( is_constructible_tuple_arg_v<U>() )
        {
          return [carg=m_defval]() -> void* { return ConstructFromTuple<U,default_value_t>::New(carg); };
        }
        if constexpr ( onika::lambda_is_compatible_with_v<default_value_t,U*> )
        {
          return [carg=m_defval]() -> void* { return carg(); };
        }
        if constexpr ( is_constructible_void_arg_v<U>() )
        {
          return []() -> void* { return new U(); };
        }
        return nullptr;
      }

      const DocString& m_doc;
      const default_value_t& m_defval;
      SlotDirection m_dir;
    };

    template<typename... T>
    static inline SlotConstructArgs<T...> make_operator_slot_construct_args( const T&... args )
    {
      return SlotConstructArgs<T...>( args... );
    }

  }
  // ===============================================================================================




  //! templated OperatorSlot, meant to be used in all user defined operators through ADD_SLOT macro
  template<typename T, bool _IsInputOnly, bool HasYAMLConversion>
  class alignas(16) OperatorSlot : public OperatorSlotBase
  {
  public:    
    static inline constexpr bool IsInputOnly = _IsInputOnly;
    using REQUIRED_t = typename OperatorNode::REQUIRED_t;
    using OPTIONAL_t = typename OperatorNode::OPTIONAL_t;
    using PRIVATE_t = typename OperatorNode::PRIVATE_t;
    using OperatorSlotDocString = typename OperatorNode::DocString;

    template<typename... U>
    inline OperatorSlot(OperatorNode* opnode, const std::string& k, const operator_slot_details::SlotConstructArgs<U...>& args)
      : OperatorSlotBase(typeid(T).name(),args.m_dir,opnode,k,args.m_doc.m_doc)
      //, m_access_patterns( operator_slot_details::SlotConstructArgs<U...>::dap_t::dac_masks() )
      , m_value_required( args.is_required_v )
    {
      using place_holder_t = TypePlaceHolder<T>;
      if( is_output_only() && args.is_optional_v )
      {
        lerr << "An output only slot cannot be marked as optional"<<std::endl;
        std::abort();
      }
      bool dont_build_allocator = ( args.is_optional_v || args.is_required_v || ( !is_output() && !args.has_provided_default_value ) ) && !args.is_private_v;      
      if( ! dont_build_allocator )
      {
        // try to build an allocator for T by all possible means
        auto allocator = args.build_default_allocator(place_holder_t{});
        if(allocator==nullptr) { lerr<<"WARNING: Cannot build allocator for "<<pathname()<<std::endl; }
        m_resource = std::make_shared<OperatorSlotResource>( allocator , DefaultResourceDeleter<T>::build() );
      }
      /*else
      {
        lout << "dont_build_allocator for "<<pathname()<<" : opt="<<args.is_optional_v<<", req="<<args.is_required_v<<", out="<<is_output()
             <<", hasdef="<<args.has_provided_default_value<<", priv="<<args.is_private_v <<std::endl;
      }*/

      if( args.is_private_v )
      {
        set_input_connectable( false );
        set_output_connectable( false );
      }
      opnode->register_slot( k, this );

      // std::sort( m_access_patterns.begin() , m_access_patterns.end() );
    }


    inline std::string value_as_string() override final
    {
      std::ostringstream oss;
      if( m_data_pointer_cache == nullptr ) return "<null>";
      onika::print_if_possible( oss , *m_data_pointer_cache , "<?>");
      return oss.str();
    }

    inline bool value_as_bool() override final
    {
      if( m_data_pointer_cache == nullptr ) return false;
      return onika::convert_to_bool( *m_data_pointer_cache , false );
    }

    inline void set_required(bool r) override final { m_value_required = r; }
    inline bool is_required() const override final { return m_value_required; }

    inline size_t memory_bytes() const override final
    {
      if( m_data_pointer_cache==nullptr ) { return 0; }
      else return onika::memory::memory_bytes( *m_data_pointer_cache );
    }

    // wether or not it's an output slot, inconditionnaly allocate and populate data from YAML::Node,
    // requiring that a YAML converter exists for type T
    inline void yaml_initialize(const YAML::Node& node) override final
    {
      yaml_initialize_priv( node, std::integral_constant<bool,HasYAMLConversion>() );
    }

    template<class U, class = std::enable_if_t< std::is_constructible_v<T,U> > >
    inline void set_resource_default_value(const U& init_val)
    {
      set_resource( default_value_copy_constructor_resource(init_val) );
    }
    
    inline T* get_typed_pointer()
    {
      void* p = nullptr;
      if( typeid(T).name() == OperatorSlotBase::m_type )
      {
        p = m_resource->check_allocate();
      }
      else
      {
        if( has_conversion() )
        {
          const auto & ptr_converter = s_type_conversion [ m_type ] [ typeid(T).name() ];
          p = m_resource->check_allocate();
          if( ptr_converter != nullptr ) { p = ptr_converter( p ); }
        }
      }
      T* typed_ptr = nullptr;
      assert( sizeof(p) == sizeof(typed_ptr) );
      // according to C++ reference, this does not explictly violate strict aliasing rule, in opposition to reinterpret_cast<T*>(p)
      std::memcpy( &typed_ptr , &p, sizeof(typed_ptr) );
      return typed_ptr;
    }

    
    //****************** allocates and initializes resource *******************
    inline void initialize_resource_pointer() override final
    {
      static_assert( sizeof(void*) == sizeof(m_data_pointer_cache) , "Pointer sizes don't match" );

      if( m_pointer_cache_intialized )
      {
         return;
      }

      if( input() != nullptr )
      {
        input()->initialize_resource_pointer();
        if( ! input()->resource()->is_null() )
        {
          set_resource( input()->resource() );        
        }
      }

      if( OperatorSlotBase::m_type != typeid(T).name() && ! has_conversion() )
      {
        fatal_error() << "Type violation: cannot get " << onika::pretty_short_type<T>() << " from " << onika::pretty_short_type(OperatorSlotBase::m_type) << std::endl;
      }
      
      if( m_resource->is_null() && OperatorSlotBase::m_type != typeid(T).name() )
      {
        fatal_error() << "Internal error: internal type has been promoted but slot is not connected to anything." << std::endl << std::flush;
      }

      void* p = nullptr;
      bool same_type = ( typeid(T).name() == OperatorSlotBase::m_type );
      bool has_conv = has_conversion();
      if( same_type || has_conv )
      {
        p = m_resource->check_allocate();
        if( !same_type )
        {
          auto converter = s_type_conversion [ m_type ] [ typeid(T).name() ];
          if( converter != nullptr )
          {
            p = converter(p);
          }
          else
          {
            fatal_error() << "Internal error: no conversion from "<<onika::pretty_short_type(m_type)<<" to "<<onika::pretty_short_type(typeid(T).name())<<std::endl;
          }
        }
      }

      if( m_value_required && p == nullptr )
      {
        lerr << "Fatal: Slot '"<< OperatorSlotBase::name() <<"' in operator '"<<owner()->pathname()<<"' has no value" << std::endl << std::flush;
        lerr << "resource: " << *(resource().get()) << std::endl;
        lerr << "input: ";
        if( input() != nullptr ) { lerr << input()->pathname(); }
        else { lerr << "<null>"; }
        lerr<<std::endl;
        std::abort();
      }

      m_data_pointer_cache = nullptr;
      
      // according to C++ reference, this does not explictly violate strict aliasing rule, in opposition to reinterpret_cast<T*>(p)
      std::memcpy( &m_data_pointer_cache , &p, sizeof(p) );
      if( m_value_required && m_data_pointer_cache == nullptr )
      {
        fatal_error() << "Internal error: m_data_pointer_cache is null while resource pointer is not, conversion did not work" << std::endl;
      }
      
      m_pointer_cache_intialized = true;
    }

    template<class C, class... E>
    inline auto make_access_controler( onika::dac::stencil_t<C,E...> )
    {
      static_assert( ! ( IsInputOnly && onika::dac::stencil_t<C,E...>::is_rw_v ) , "cannot use RW access on a read-only slot" );
      return onika::dac::DataAccessControler<T , onika::dac::stencil_t<C,E...> >{ m_data_pointer_cache };
    }

    template<class... S>
    inline auto make_access_controler( onika::dac::ro_t , S... )
    {
      using slices_t = std::conditional_t< sizeof...(S)==0 , typename onika::dac::DataDecompositionTraits<T>::slices_t , onika::dac::DataSlices<S...> >;
      return onika::dac::DataAccessControler<T , onika::dac::stencil_t< onika::dac::stencil_element_t<slices_t,onika::dac::DataSlices<> > > >{ m_data_pointer_cache };      
    }

    template<class... S>
    inline auto make_access_controler( onika::dac::rw_t , S... )
    {
      static_assert( ! IsInputOnly , "cannot use RW access on a read-only slot" );
      using slices_t = std::conditional_t< sizeof...(S)==0 , typename onika::dac::DataDecompositionTraits<T>::slices_t , onika::dac::DataSlices<S...> >;
      return onika::dac::DataAccessControler<T , onika::dac::stencil_t< onika::dac::stencil_element_t<onika::dac::DataSlices<>,slices_t> > >{ m_data_pointer_cache };      
    }

    inline void reset_input() override final
    {
      m_type = typeid(T).name();
      m_input = nullptr;
      m_pointer_cache_intialized = false;
      m_data_pointer_cache = nullptr;
    }

    inline std::shared_ptr<OperatorSlotBase> new_instance(OperatorNode* opnode, const std::string& k, SlotDirection dir) override final
    {
      return make_operator_slot<T>( opnode, k, dir );
    }

    // ============ value accessors ==========
    inline bool has_value() const override final { return m_data_pointer_cache != nullptr; }

    inline const T& operator *  () const { check_non_null_ptr_dereference(); return *m_data_pointer_cache; }
    inline       T& operator *  ()       { check_non_null_ptr_dereference(); return *m_data_pointer_cache; }
    inline const T* operator -> () const { check_non_null_ptr_dereference(); return  m_data_pointer_cache; }
    inline       T* operator -> ()       { check_non_null_ptr_dereference(); return  m_data_pointer_cache; }
    inline const T* get_pointer () const { return  m_data_pointer_cache; }
    inline       T* get_pointer ()       { return  m_data_pointer_cache; }

    // members to mimic behavior of a pointer
    //inline operator 

  private:
 
    inline void check_non_null_ptr_dereference()
    {
      if( ! has_value() )
      {
        lerr << "Slot "<<pathname()<<" has no value, it cannot be accessed"<<std::endl;
        std::abort();
      }
    }

    // wether or not it's an output slot, inconditionnaly allocate and populate data from YAML::Node,
    // requiring that a YAML converter exists for type T
    inline void yaml_initialize_priv(const YAML::Node& in_node, std::true_type )
    {
      set_resource( std::make_shared<OperatorSlotResource>(
        [ node=YAML::Clone(in_node), sname=this->name(), oname=this->owner()->name(), owner=this->owner()]()
        -> void*
        {
          T* allocated_value = new T();
          bool decode_success = false;
          std::ostringstream errstr;
          try
          {
            decode_success = onika::yaml::YAMLConvertWrapper<T>::decode(node,*allocated_value);
          }
          catch(const YAML::Exception& e)
          {
            errstr << "Error reading value for slot "<<sname<<std::endl
                   << "at line " << e.mark.line <<", column "<<e.mark.column <<std::endl;
            decode_success = false;
          }
          if( ! decode_success )
          {
            errstr << "could not convert key "<< sname <<" to type "<< onika::pretty_short_type<T>() << " in node "<< oname << std::endl
                   << "Owner node @ " <<owner <<std::endl
                   << "YAML data is :" <<std::endl;
            onika::yaml::dump_node_to_stream( errstr , node );
            delete allocated_value;
            fatal_error() << errstr.str() << std::endl << std::flush;
          }
          return allocated_value;
        }
        , DefaultResourceDeleter<T>::build() ) );

        // event though it is input slot, it cannot be input connected anymore because user gave it a value
        // that cannot be overriden from another output
        this->OperatorSlotBase::m_is_input_connectable = false;
    }

    inline void yaml_initialize_priv(const YAML::Node& node, std::false_type )
    {
      std::ostringstream errstr;
      //initialize_value( [this,node](T& value)
      //  {
          errstr << "no YAML conversion (known at compile time) for key "<< OperatorSlotBase::name() <<" to type "<<onika::pretty_short_type<T>() << " in node "<< this->owner()->name() <<std::endl
                 << "YAML data is :" <<std::endl;
          onika::yaml::dump_node_to_stream( errstr , node );
          fatal_error() << errstr.str() << std::endl << std::flush;
      //  } );
    }

    inline bool has_conversion()
    {
      return OperatorSlotBase::has_type_conversion( OperatorSlotBase::m_type , typeid(T).name() );
    }

    // ==== members =====
    T* m_data_pointer_cache = nullptr;
    // std::vector<uint64_t> m_access_patterns; // different parts (slices) of the data that may be accessed through data access control mechanisms
    bool m_value_required = false;
    bool m_pointer_cache_intialized = false;
  };


  // create managed slot
  template<class T >
  inline std::shared_ptr< OperatorSlot<T> > make_operator_slot( OperatorNode* opnode, const std::string& k, SlotDirection d )
  {
    std::shared_ptr< OperatorSlot<T> > slot = std::make_shared< OperatorSlot<T> >( opnode, k, operator_slot_details::make_operator_slot_construct_args(d) );
    opnode->register_managed_slot( slot );
    return slot;
  }

  // just a helper, usefull for ADD_SLOT macro
  inline constexpr bool operator == (const OperatorNode::PRIVATE_t& , const SlotDirection&) { return false; }

  // bug workaround : automatically promote [unsigned] int to [unsigned] long
  template<class T> struct OperatorSlotTypePromotion { using type = T; };
  template<> struct OperatorSlotTypePromotion<int> { using type = long; };
  template<> struct OperatorSlotTypePromotion<unsigned int> { using type = unsigned long; };
  template<class T> using automatic_slot_type_promotion = typename OperatorSlotTypePromotion<T>::type;

} }

// (!) Convinience macro, to be used _ONLY_ inside an OperatorNode derived class declaration
#define ADD_SLOT(T,N,D...) ::onika::scg::OperatorSlot< ::onika::scg::automatic_slot_type_promotion<T> , GET_FIRST_ARG(D)==::onika::scg::INPUT > N { this, #N, ::onika::scg::operator_slot_details::make_operator_slot_construct_args(D) }


