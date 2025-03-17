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
#include <cstdint>
#include <vector>

#include <onika/cuda/ro_shallow_copy.h>
#include <onika/memory/allocator.h>

namespace exanb
{

  struct AmrGrid
  {
    using SubGridStartVector = onika::memory::CudaMMVector<size_t>;
    using SubGridCellsVector = onika::memory::CudaMMVector<uint32_t>;
  
    inline bool empty() const { return m_sub_grid_start.empty(); }

    // sub grid start : one value for each cell, gives offset in m_sub_grid_cells where to find sub grid indices for this cell
    inline const SubGridStartVector& sub_grid_start() const { return m_sub_grid_start; }
    inline SubGridStartVector& sub_grid_start() { return m_sub_grid_start; }

    // sub grid cells : for each cell, stores start indices of particles for each sub cell
    inline const SubGridCellsVector& sub_grid_cells() const { return m_sub_grid_cells; }
    inline SubGridCellsVector& sub_grid_cells() { return m_sub_grid_cells; }

    // get side resolution of a cell. e.g. if a cell is divided into 3x3x3 sub cells, it returns 3.
    ssize_t cell_resolution(size_t cell_i) const;

    // resets all cells to non refined state (all cells are decomposed into 1x1x1 sub grid)
    void clear_sub_grids( size_t n_cells );

    // m_sub_grid_tart[i] stores index of first sub grid offset in m_sub_grid_cells for cell i+1. cell 0 always has its first offset index equal to 0
    SubGridStartVector m_sub_grid_start;
    
    // offsets inside each cell for
    // limiting to uint32_t is fair enough because indices are relative to each cell. It allows up to 4.10^9 particles per cell.
    SubGridCellsVector m_sub_grid_cells;
    
    // tells if sub cells have been ordered using Z-curve or scan order
    bool m_z_curve = false;
  };

  struct ReadOnlyAmrGrid
  {
    size_t const * const m_sub_grid_start = nullptr;
    uint32_t const * const m_sub_grid_cells = nullptr;
    bool m_z_curve = false;    

    ReadOnlyAmrGrid() = default;
    ReadOnlyAmrGrid(const ReadOnlyAmrGrid&) = default;

    inline ReadOnlyAmrGrid( const AmrGrid& amr )
      : m_sub_grid_start( amr.m_sub_grid_start.data() )
      , m_sub_grid_cells( amr.m_sub_grid_cells.data() )
      , m_z_curve( amr.m_z_curve )
      {}
  };

} // end of namespace exanb


namespace onika
{

  namespace cuda
  {

    template<> struct ReadOnlyShallowCopyType< exanb::AmrGrid > { using type = exanb::ReadOnlyAmrGrid ; };

  }

}

