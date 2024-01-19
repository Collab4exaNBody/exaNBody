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

namespace exanb
{

  // get sub range index from index in the range [gbegin;gend[, subdivided into n_subranges parts
  static inline size_t sub_range_index( ssize_t gindex, size_t n_subranges, ssize_t gbegin, ssize_t gend )
  {
    assert( gend > gbegin );
    assert( gindex >= gbegin && gindex < gend );
    assert( n_subranges >= 1 );
    return ( (gindex-gbegin+1) * n_subranges - 1) / ( gend - gbegin );
  }

  static inline ssize_t sub_range_begin( size_t sub_range_index, size_t n_subranges, ssize_t gbegin, ssize_t gend )
  {
    assert( gend > gbegin );
    assert( n_subranges >= 1 );
    assert( sub_range_index >= 0 /*&& sub_range_index < n_subranges*/ );
    return gbegin + ( ( sub_range_index * (gend-gbegin) ) / n_subranges );
  }

  static inline ssize_t sub_range_end( size_t sub_range_index, size_t n_subranges, ssize_t gbegin, ssize_t gend )
  {
    return sub_range_begin( sub_range_index+1, n_subranges, gbegin, gend );
  }
  
}

