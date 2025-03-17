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

#include <utility>
#include <onika/cuda/cuda.h>

#include <string>
#include <iostream>
#include <sstream>

namespace exanb
{

  // compile time version of unique_pair_count function
  template<unsigned int S>
  struct UnorderedPairCount  
  {
    static constexpr unsigned int count = ( S * (S+1) ) / 2;
  };

  // encodes an integer pair (a,b) , independently from relative order, such that unique_pair_id(a,b) == unique_pair_id(b,a)
  ONIKA_HOST_DEVICE_FUNC inline unsigned int unique_pair_count(unsigned int s)
  {
    return ( s * (s+1) ) / 2;
  }

  ONIKA_HOST_DEVICE_FUNC inline unsigned int unique_pair_id(unsigned int _a, unsigned int _b)
  {
    unsigned int a = _a;
    unsigned int b = _b;
    if( a > b )
    {
      a = _b;
      b = _a;
    }

    return unique_pair_count(b) + a;
  }

  // guarantees that a <= b, and either a=b=pair_id=0 or a<=b and a<pair_id and b<pair_id
  inline void pair_id_to_type_pair(unsigned int pair_id, unsigned int& a, unsigned int& b)
  {
    b = 0;
    unsigned int ppc = 0;
    unsigned int pc = unique_pair_count(1);
    while( pc <= pair_id )
    {
      ++b;
      ppc = pc;
      pc = unique_pair_count(b+1);
    }
    assert( ppc <= pair_id );
    a = pair_id - ppc;
  }

}


