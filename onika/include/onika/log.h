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
#include <cstdlib>

namespace onika
{
  static inline constexpr auto & lerr = std::cerr;
  static inline constexpr auto & lout = std::cout;
  static inline constexpr auto & ldbg = std::cout;

  struct FatalErrorLogStream
  {
    std::ostringstream m_oss;
    inline FatalErrorLogStream() {}    
    template<class T> inline FatalErrorLogStream& operator << (const T& x)
    {
      m_oss << x;
      return *this;
    }
    inline FatalErrorLogStream& operator << ( std::ostream& (*manip)(std::ostream&) )
    {
       m_oss << manip ;
       return *this;
    }
    inline ~FatalErrorLogStream() { std::cerr << m_oss.str(); std::abort(); }
  };
  
  inline FatalErrorLogStream fatal_error() { return FatalErrorLogStream(); }
}

