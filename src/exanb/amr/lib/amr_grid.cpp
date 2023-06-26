
#include <exanb/amr/amr_grid.h>
#include <exanb/core/math_utils.h>

#ifndef NDEBUG
#include <exanb/core/basic_types_stream.h>
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

