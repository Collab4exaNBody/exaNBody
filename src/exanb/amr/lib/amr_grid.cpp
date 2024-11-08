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

#include <exanb/amr/amr_grid.h>
#include <onika/math/math_utils.h>

#ifndef NDEBUG
#include <onika/math/basic_types_stream.h>
#endif

#include <algorithm>
#include <numeric>
#include <cmath>
#include <omp.h>
#include <cassert>

namespace exanb
{

  ssize_t AmrGrid::cell_resolution(size_t cell_i) const
  {
    size_t sgsize = m_sub_grid_start[cell_i+1] - m_sub_grid_start[cell_i];
    ssize_t n_sub_cells = sgsize+1;
    ssize_t sgside = std::cbrt( n_sub_cells );
    assert( (sgside*sgside*sgside) == n_sub_cells );
    assert( sgside == icbrt64(n_sub_cells) );
    return sgside;
  }

  void AmrGrid::clear_sub_grids( size_t n_cells )
  {
    m_sub_grid_start.assign( n_cells+1 , 0 );
    m_sub_grid_cells.clear();
  }

}

