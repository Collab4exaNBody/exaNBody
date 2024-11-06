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

#include <cstring>
#include <utility>

namespace onika
{
  namespace omp
  {

    template<class StreamT , int Nin, int ... Is>
    static inline StreamT& print_depend_indices( StreamT& out , std::integer_sequence<int,Nin,Is...> )
    {
      int c=0;
      out<<"in:";
      ( ... , ( ( (c++) < Nin ) ? (out << " " << Is) : out ) );
      out<<" , inout:";
      c=0;
      ( ... , ( ( (c++) >= Nin ) ? (out << " " << Is) : out ) );
      return out;
    }

    inline const char* tag_filter_out_path(const char* tag)
    {
      static const char * no_tag = "<no-tag>";
      if( tag == nullptr ) return no_tag;
      const char* s = std::strrchr( tag , '/' );
      if( s == nullptr ) return tag;
      return s+1;
    }
     
  }
}
