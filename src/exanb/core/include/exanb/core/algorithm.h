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

#include <cstdlib>
#include <omp.h>
#include <assert.h>

namespace exanb
{

  // clamp is only available in c++17
  template<typename T>
  static inline const T&
  clamp( const T& x, const T& lower, const T& upper )
  {
    if( x < lower ) return lower;
    else if( upper < x ) return upper;
    else return x;
  }

  // exclusive prefix sum
  template<typename T>
  static inline void exclusive_prefix_sum(T* a, size_t N)
  {
    T s = 0;
    for(size_t i=0;i<N;i++)
    {
      T ns = s+a[i];
      a[i] = s;
      s = ns;
    }
  }
  
}

