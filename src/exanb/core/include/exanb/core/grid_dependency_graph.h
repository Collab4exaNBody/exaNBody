#pragma once

#include <exanb/core/basic_types_operators.h>
#include <exanb/core/basic_types_stream.h>
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

