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

#include <onika/math/basic_types_operators.h>
#include <onika/math/basic_types_stream.h>
#include <exanb/core/parallel_grid_algorithm.h>

#include <cstdlib>
#include <vector>
#include <unordered_set>
#include <numeric>
#include <assert.h>

namespace exanb
{

  struct CellDependencyPattern
  {
    int dep[3][3][3];
  };

  struct GridDependencyPattern
  {
    CellDependencyPattern pattern[2][2][2];
    GridDependencyPattern();
  };

  struct GridDependencyGraph
  {
    IJK m_grid_dims;
    std::vector<size_t> m_cell; // n_cells
    std::vector<size_t> m_start; // n_cells+1
    std::vector<size_t> m_deps; // total deps

    void build(IJK dims);
    void adj_matrix(std::vector<bool>& mat);
    void closure_matrix(std::vector<bool>& mat);
    bool check(IJK dims);
    
    static GridDependencyPattern s_grid_pattern;
  };

}

