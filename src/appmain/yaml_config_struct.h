#pragma once

#include <yaml-cpp/yaml.h>
#include <type_traits>
#include <iostream>
#include <iomanip>
#include <string>


template<class T> static inline std::string xsv2_to_string_helper( const T* state );

template<class T>
struct XStampV2OptionState
{
  static inline std::string to_string(const T* state)
  {
    if(state==nullptr) return "<null>";
    else return std::to_string(*state);
  }
};

template<>
struct XStampV2OptionState<std::string>
{
  static inline const std::string& to_string(const std::string* state)
  {
    static const std::string nullstr("<null>");
    if(state==nullptr) return nullstr;
    else return *state;
  }
};

template<>
struct XStampV2OptionState<YAML::Node>
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
struct XStampV2OptionState< std::vector<T> >
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

template<class T> static inline std::string xsv2_to_string_helper( const T* state )
{
  return XStampV2OptionState<T>::to_string(state);
}


struct XStampV2OptionDocumentation
{  
  template<class T=int>
  inline XStampV2OptionDocumentation(const std::string& t, const std::string& k, const std::string& d, const std::string& def = std::string{}, XStampV2OptionDocumentation* parent = nullptr , const T* state = nullptr)
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

  template<class StreamT> inline StreamT& print_default_config( StreamT& out , int indent = 0)
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

  template<class StreamT> inline StreamT& print_command_line_options( StreamT& out , int level=0 , const std::string& path = std::string{} )
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
  XStampV2OptionDocumentation* m_parent = nullptr;
  std::list<XStampV2OptionDocumentation*> m_sub_items;
};

template<class T>
struct XStampV2YamlConvertHelper
{
  static inline T read(const YAML::Node& node, const T& dv)
  {
    if( node.IsNull() || !node.IsDefined() ) return dv;
    else return node.as<T>();
  }
};
template<class T>
struct XStampV2YamlConvertHelper< std::vector<T> >
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

#define xsv2ConfigBegin(name,...) \
  struct xsv2ConfigStruct_##name \
  { \
    inline xsv2ConfigStruct_##name ( YAML::Node n , XStampV2OptionDocumentation* parent_doc=nullptr ) : m_yaml_node( n ) \
    { if(parent_doc!=nullptr) { parent_doc->m_sub_items.push_back( &m_doc ); } } \
    YAML::Node m_yaml_node; \
    XStampV2OptionDocumentation m_doc { std::string{} , #name , std::string{__VA_ARGS__} }

#define xsv2ConfigItem(type,name,defval,...) \
  type name = XStampV2YamlConvertHelper<type>::read( yaml_sub_node(m_yaml_node,#name) , defval ); \
  XStampV2OptionDocumentation name##_doc { #type , #name , std::string{__VA_ARGS__} , #defval , & m_doc , & name }

#define xsv2ConfigNode(name,...) \
  YAML::Node name = yaml_sub_node(m_yaml_node,#name); \
  XStampV2OptionDocumentation name##_doc { std::string{"YAML::Node"} , #name , std::string{__VA_ARGS__} , std::string{} , & m_doc , & name }
  
#define xsv2ConfigStruct(name) \
  xsv2ConfigStruct_##name name { yaml_sub_node(m_yaml_node,#name) , & m_doc }

#define xsv2ConfigEnd() }

