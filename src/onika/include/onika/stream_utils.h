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

#include <iostream>
#include <sstream>
#include <functional>

namespace onika
{

  template<class T> struct PrintableFormattedObject
  {
    template<class StreamT> inline StreamT& to_stream(StreamT& out) const { return out << "<non-printable>"; }
  };

  template<class StreamT>
  struct PrintableFormattedObject< std::function<StreamT&(StreamT&)> >
  {
    std::function<StreamT&(StreamT&)> f;
    inline StreamT& to_stream(StreamT& out) const { return f(out); }
  };

# ifndef __CUDACC__
  template<class FuncT>
  static inline auto make_streamable_functor( FuncT && f ) { std::function g = std::move(f); return PrintableFormattedObject<decltype(g)>{g}; }
# endif

  inline std::ostream& stdout_stream() { return std::cout; }
}

template<class T>
inline std::ostream& operator << (std::ostream& out , const onika::PrintableFormattedObject<T>& a )
{
  return a.to_stream(out);
}

namespace onika
{

  struct OStringStream
  {
    OStringStream(OStringStream && other) : m_oss( std::move(other.m_oss) ) , m_str(other.m_str) {}
    template<class... T> inline OStringStream(std::string& s , const T& ... args) : m_oss(s) , m_str(s) { ( ... , ( m_oss << args ) ); }
    template<class T> inline OStringStream& operator << ( const T& x ) { m_oss<<x; return *this; }
    inline ~OStringStream() { m_str = m_oss.str(); }
    std::ostringstream m_oss;
    std::string & m_str;
  };

}

#define ONIKA_STDOUT_OSTREAM ::onika::stdout_stream()
