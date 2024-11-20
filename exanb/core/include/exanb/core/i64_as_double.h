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

#include <cstdint>
#include <cmath>
#include <cassert>

namespace exanb
{
  // lossless encode as many bits as possible into a double.
  static inline double encode_i64_as_double(int64_t x)
  {
    static constexpr uint64_t bit52 = (1ull<<52);
    static constexpr uint64_t bit62 = (1ull<<62);
    static constexpr uint64_t bit63 = (1ull<<63);
    static_assert( bit52 != 0 , "constant definition error for bit52");
    static_assert( bit62 != 0 , "constant definition error for bit62");
    static_assert( bit63 != 0 , "constant definition error for bit63");
    uint64_t ax = std::abs(x);
    assert( ( ax & (bit62|bit63) ) == 0 );
    uint64_t sx = x & bit63;
    uint64_t ex = ( ax | sx ) + bit52;
    union double_as_int { uint64_t i; double d; } dai { ex };
    return dai.d;
  }

  static inline int64_t decode_i64_as_double(double d)
  {
    static constexpr uint64_t bit52 = (1ull<<52);
    static constexpr uint64_t bit63 = (1ull<<63);
    union { double d; int64_t i; } dai { d };
    int64_t x = dai.i;
    x &= ~bit63;
    x -= bit52;
    if(d<0.0) x = -x;
    return x;
  }

}


// ===================== unit tests ====================

#include <random>
#include <onika/test/unit_test.h>

ONIKA_UNIT_TEST(encode_i64_as_double)
{
  using namespace exanb; 

  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<uint64_t> ud(0);

  for(int i=0;i<1000000;i++)
  {
    int64_t a = ud(gen);
    int64_t b = ud(gen);
    if( std::abs(a)<(1ll<<62) && std::abs(b)<(1ll<<62) )
    {
      double da = encode_i64_as_double( a );
      double db = encode_i64_as_double( b );
      ONIKA_TEST_ASSERT( ( a==b && da==db )
                       || ( a<b && da<db )
                       || ( a>b && da>db ) );
      int64_t dda = decode_i64_as_double( da );
      int64_t ddb = decode_i64_as_double( db );
      ONIKA_TEST_ASSERT( dda==a && ddb==b );
    }
  }
}

