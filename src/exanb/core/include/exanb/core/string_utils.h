#pragma once

#include <string>
#include <vector>
#include <cstdlib>
#include <cassert>

#include <onika/debug.h>

namespace exanb
{

  template<typename FormatArg>
  inline const FormatArg& convert_format_arg(const FormatArg& a) { return a; }

  inline const char* convert_format_arg(const std::string& a) { return a.c_str(); }

  template<typename... Args>
  inline std::string format_string(const std::string& format, const Args & ... args)
  {
    int len = std::snprintf( nullptr, 0, format.c_str(), convert_format_arg(args)... );
    assert(len>=0);
    std::string s(len+1,' ');
    ONIKA_DEBUG_ONLY( int len2 = ) std::snprintf( & s[0], len+1, format.c_str(), convert_format_arg(args)... );
    assert(len2==len);
    s.resize(len);
    return s;
  }

  std::vector<std::string> split_string(const std::string& s, char delim=' ');

  void function_name_and_args(const std::string& proto, std::string& name, std::vector<std::string>& args );

  std::string str_tolower(std::string s);

  std::string str_indent(const std::string& s, int width=4, char indent_char=' ', const std::string& indent_suffix="" );
  
  std::string large_integer_to_string(size_t n);

  std::string plurial_suffix(double n, const std::string& suffix="s");

  std::string memory_bytes_string( ssize_t n , const char* fmt = "%.2f %s" );

}
