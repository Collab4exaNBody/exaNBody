#include <exanb/core/type_utils.h>

#include <string>
#include <regex>
#include <iostream>

#ifdef __GNUG__
#include <cxxabi.h>
#endif

namespace exanb
{
  std::string demangle_type_string(std::string s)
  {
# ifdef __GNUG__
    int status = 0;
    char* nicename = abi::__cxa_demangle(s.c_str(), NULL, NULL, &status);
    if( nicename != nullptr )
    {
      s = nicename;
      std::free(nicename);
    }
# endif
    return s;
  }

  std::string remove_exanb_namespaces(std::string s)
  {
    s = std::regex_replace(s, std::regex("exanb::"), "");
    s = std::regex_replace(s, std::regex("field::_"), "");
    s = std::regex_replace(s, std::regex("field::"), "");
    s = std::regex_replace(s, std::regex("std::map"), "map");
    s = std::regex_replace(s, std::regex("std::__cxx11::basic_string"), "string");
    s = std::regex_replace(s, std::regex("string<char,std::char_traits<char>,std::allocator<char>>"), "string");
    return s;
  }

  std::string strip_type_spaces(std::string s)
  {
    using std::string;
    string::size_type p;
    while( ( p=s.find(" >") ) != string::npos ) { s.erase(p,1); }
    while( ( p=s.find(", ") ) != string::npos ) { s.erase(p+1,1); }
    return s;
  }
  
  std::string remove_shared_ptr(std::string s)
  {
    if( s.find("std::shared_ptr<") == 0 )
    {
      s = s.substr(16,s.length()-17);
    }
    return s;
  }

  std::string simplify_std_vector(std::string s)
  {
    static const std::string rs = "std::vector<(.*),std::allocator<\\1>>";
    static const std::regex re(rs);

    if( s.find("std::vector<") != 0 ) { return s; }
    std::smatch m;
    std::regex_search(s, m, re );
    if( m.size() >= 2 )
    {
      std::string subtype = m[1];
      return "vector<"+subtype+">";
    }
    else
    {
      return s;
    }
  }


  std::string pretty_short_type(std::string s)
  {
    if( s == typeid(std::string).name() ) { return "string"; }
    s = demangle_type_string(s);
    s = remove_shared_ptr(s);
    s = strip_type_spaces(s);
    s = simplify_std_vector(s);
    s = remove_exanb_namespaces(s);
    return s;
  }
  
}

