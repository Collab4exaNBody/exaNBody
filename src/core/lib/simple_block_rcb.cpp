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
#include <onika/math/basic_types.h>
#include <exanb/core/grid_algorithm.h>
#include <exanb/core/simple_block_rcb.h>
#include <cassert>

namespace exanb
{

  GridBlock simple_block_rcb( GridBlock block, size_t n_parts, size_t part )
  {
    if( n_parts <= 1 ) { return block; }
    assert( part>=0 && part<n_parts );
    size_t pivot = n_parts/2;
    bool side = ( part >= pivot );
    IJK dims = dimension(block);
    if( dims.i >= dims.j && dims.i >= dims.k )
    {
      if( side ) { block.start.i = block.start.i + dims.i/2 ; }
      else       { block.end.i   = block.start.i + dims.i/2 ; }
    }
    else if( dims.j >= dims.i && dims.j >= dims.k )
    {
      if( side ) { block.start.j = block.start.j + dims.j/2 ; }
      else       { block.end.j   = block.start.j + dims.j/2 ; }
    }
    else
    {
      if( side ) { block.start.k = block.start.k + dims.k/2 ; }
      else       { block.end.k   = block.start.k + dims.k/2 ; }
    }

    size_t sub_group_size = n_parts/2;
    size_t sub_group_rank = part;
    if( side )
    {
      sub_group_size = n_parts - (n_parts/2);
      sub_group_rank = part - (n_parts/2);
    }

    return simple_block_rcb( block , sub_group_size , sub_group_rank );
  }

}


