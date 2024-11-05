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

#include <onika/string_utils.h>

#include <string>
#include <vector>
#include <algorithm>
#include <sstream>

namespace onika
{

  std::vector<std::string> split_string(const std::string& s, char delim)
  {
    std::vector<std::string> result;
    size_t pos = 0;
    size_t nextpos = 0;
    while( (nextpos=s.find(delim,pos)) != std::string::npos )
    {
      result.push_back( s.substr(pos,nextpos-pos) );
      pos = nextpos + 1;
    }
    if( pos < s.length() )
    result.push_back( s.substr(pos) );
    return result;
  }

  void function_name_and_args(const std::string& proto, std::string& name, std::vector<std::string>& args )
  {
    using std::string;
    string::size_type parbeg = proto.find('(');
    string::size_type parend = proto.rfind(')');
    if( parbeg != string::npos && parend != string::npos && parend > parbeg )
    {
      args = split_string( proto.substr(parbeg+1,parend-parbeg-1) , ',' );
      name =  proto.substr(0,parbeg);
    }
    else
    {
      args.clear();
      name = proto;
    }
  }

  std::string str_tolower(std::string s)
  {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); } );
    return s;
  }

  std::string large_integer_to_string(size_t n)
  {
    if(n==0) return "0";
  
    std::string digits;
    {
      std::ostringstream oss;
      while( n > 0 )
      {
        int d = n%10;
        oss << static_cast<char>('0'+d);
        n /= 10;
      }
      digits = oss.str();
    }
    n=digits.length();
    std::ostringstream oss;
    for(unsigned int c=0;c<n;c++)
    {
      int p = n-c-1;
      oss << digits[n-c-1];
      if(p!=0 && (p%3)==0 ) oss << ',';
    }
    return oss.str();
  }

  std::string str_indent(const std::string& s, int width, char indent_char, const std::string& indent_suffix)
  {
    std::ostringstream oss;
    for(int i=0;i<width;i++) oss << indent_char;
    oss << indent_suffix;
    std::string istr = oss.str();
    std::vector<int> positions;
    std::string::size_type pos = 0;
    std::string::size_type nextpos = std::string::npos;
    while( (nextpos=s.find('\n',pos+1)) != std::string::npos ) { positions.push_back(pos); pos=nextpos+1; }
    int pinc=0;
    std::string r=s;
    for(auto p:positions)
    {
      r.insert( p+pinc , istr );
      pinc += istr.length();
    }
    return r;
  }

  std::string plurial_suffix(double n, const std::string& suffix)
  {
    if( std::abs(n) > 1.0 ) return suffix;
    else return "";
  }

  std::string memory_bytes_string( ssize_t n , const char* fmt )
  {
    const char* S[] = { "b" , "Kb" , "Mb" , "Gb" , "Tb" };
    static constexpr size_t nunits = sizeof(S)/sizeof(const char*);
    double N = n;
    size_t p=0;
    while( N > 1024 && p<(nunits-1) ) { N/=1024; p++; }
    return format_string( fmt , N , S[p] );
  }

}
