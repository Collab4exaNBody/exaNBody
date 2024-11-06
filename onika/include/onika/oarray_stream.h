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

#include <onika/oarray.h>
#include <onika/stream_utils.h>
#include <iostream>

namespace onika
{

  template<class T>
  struct SimpleArrayFormat
  {
    const T* v = nullptr;
    size_t n = 0;
    const char sep='\0';
    const char beg='\0';
    const char end='\0';
  };

  template<class T>
  struct PrintableFormattedObject< SimpleArrayFormat<T> >
  {
    const SimpleArrayFormat<T> a;
    template<class StreamT> inline StreamT& to_stream(StreamT& out) const
    {
      if( a.beg != '\0' ) out << a.beg;
      if( a.n>0 )
      {
        if constexpr (std::is_integral_v<T>) { out << ssize_t(a.v[0]); }
        else { out << a.v[0]; }
      }
      
      for(size_t i=1;i<a.n;i++)
      {
        if(a.sep != '\0') out << a.sep;
        if constexpr (std::is_integral_v<T>) { out << ssize_t(a.v[i]); }
        else { out << a.v[i]; }
      }
      if( a.end != '\0' ) out << a.end;
      return out;
    }
  };

  template<class T, size_t N>
  PrintableFormattedObject< SimpleArrayFormat<T> > format_array( const oarray_t<T,N>& v , char sep=',' , char beg='(' , char end=')' )
  {
    return { { v.data() , v.size() , sep , beg , end } };
  }

  template<class T>
  PrintableFormattedObject< SimpleArrayFormat<T> > format_array( const T* v, size_t n, char sep=',' , char beg='(' , char end=')' )
  {
    return { { v , n, sep , beg , end } };
  }  
}

