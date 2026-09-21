/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements. See the NOTICE file
distributed with this work for additional information
regarding copyright ownership. The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied. See the License for the
specific language governing permissions and limitations
under the License.
*/
#pragma once

#include <string>
#include <exanb/grid_cell_particles/grid_cell_values.h>
#include <exanb/core/grid_algorithm.h>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/parallel_for.h>

namespace exanb
{
  // Reduction of a GridCellValues field to a single value, GPU-capable and deterministic.
  //
  // Two stages: (1) one thread per (non skipped) cell folds that cell's sub-cells into a per-cell partial
  // result (onika::parallel::parallel_for, so thread-per-index, NOT block_parallel_for); (2) the partials
  // are combined on the host in fixed cell order. No atomics, so the floating point result is bit-identical
  // between runs, thread counts and host/GPU execution. The result is local to this process: MPI reduction
  // across ranks is left to the caller.
  //
  // FuncT contract (must be trivially copyable, and ONIKA_HOST_DEVICE_FUNC if CudaCompatible):
  //   void accumulate( ResultT& acc, const double* subcell_values ) const; // fold one sub-cell (ncomps values)
  //   void combine   ( ResultT& acc, const ResultT& partial ) const;       // fold one per-cell partial
  // `init` must be the neutral element (0 for a sum), it seeds every partial and the final combine.
  // GPU execution requires ReduceGridCellValuesTraits<FuncT>::CudaCompatible = true AND the calling
  // translation unit to be a .cu file (a .cpp never produces a real GPU kernel).
  //
  // ponytail: one thread per cell, cells with a huge subdiv^3*ncomps would under-use the GPU; upgrade to a
  // block-per-cell kernel with onika::cuda::block_reduce_* if that ever matters.
  template<class FuncT> struct ReduceGridCellValuesTraits
  {
    static inline constexpr bool CudaCompatible = false;
  };

  template<class FuncT, class ResultT>
  struct ReduceGridCellValuesFunctor
  {
    const double * __restrict__ m_data = nullptr; // start of the field, m_stride values per cell
    size_t m_stride = 0;
    IJK m_dims = { 0, 0, 0 };                     // full grid dims, ghost layers included
    ssize_t m_ghost_layers = 0;                   // layers skipped on each side
    size_t m_n_subcells = 0;
    size_t m_ncomps = 1;
    FuncT m_func;
    ResultT m_init;
    ResultT * __restrict__ m_partials = nullptr;

    ONIKA_HOST_DEVICE_FUNC inline void operator () ( size_t i ) const
    {
      const IJK cell_loc = grid_index_to_ijk( m_dims - 2*m_ghost_layers , ssize_t(i) ) + m_ghost_layers;
      const size_t cell_i = size_t( grid_ijk_to_index( m_dims , cell_loc ) );
      const double * cell_data = m_data + cell_i * m_stride;
      ResultT acc = m_init;
      for(size_t s=0; s<m_n_subcells; s++) m_func.accumulate( acc , cell_data + s*m_ncomps );
      m_partials[i] = acc;
    }
  };

  // partials: caller-owned scratch (resized here, kept allocated across calls)
  template<class FuncT, class ResultT>
  static inline ResultT reduce_grid_cell_values(
      const GridCellValues& gcv
    , const std::string& field_name
    , ssize_t ghost_layers // 0: reduce over every cell; gl: skip gl layers on each side (i.e. local cells only)
    , const FuncT& func
    , ResultT init
    , onika::memory::CudaMMVector<ResultT>& partials
    , onika::parallel::ParallelExecutionContext * exec_ctx )
  {
    const GridCellField& field = gcv.field( field_name );
    const IJK dims = gcv.grid_dims();
    const IJK reduced_dims = dims - 2*ghost_layers;
    if( reduced_dims.i<=0 || reduced_dims.j<=0 || reduced_dims.k<=0 ) return init;
    const size_t n_cells = size_t(reduced_dims.i) * size_t(reduced_dims.j) * size_t(reduced_dims.k);
    const size_t n_subcells = field.m_subdiv * field.m_subdiv * field.m_subdiv;

    partials.resize( n_cells );
    ReduceGridCellValuesFunctor<FuncT,ResultT> kernel = {
      gcv.field_data(field).m_data_ptr, gcv.components(), dims, ghost_layers,
      n_subcells, field.m_components / n_subcells, func, init, partials.data() };
    onika::parallel::parallel_for( n_cells , kernel , exec_ctx ); // synchronous (see ~ParallelExecutionWrapper)

    ResultT result = init;
    for(size_t i=0;i<n_cells;i++) func.combine( result , partials[i] );
    return result;
  }

  // Ready-made sum of one component of a field.
  struct GridCellValuesSum
  {
    size_t m_comp = 0;
    ONIKA_HOST_DEVICE_FUNC inline void accumulate( double& acc , const double* v ) const { acc += v[m_comp]; }
    ONIKA_HOST_DEVICE_FUNC inline void combine( double& acc , const double& partial ) const { acc += partial; }
  };
  template<> struct ReduceGridCellValuesTraits<GridCellValuesSum> { static inline constexpr bool CudaCompatible = true; };
}

namespace onika
{
  namespace parallel
  {
    template<class FuncT, class ResultT>
    struct ParallelForFunctorTraits< exanb::ReduceGridCellValuesFunctor<FuncT,ResultT> >
    {
      static inline constexpr bool CudaCompatible = exanb::ReduceGridCellValuesTraits<FuncT>::CudaCompatible;
    };
  }
}
