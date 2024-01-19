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

#include <exanb/compute/compute_cell_particle_pairs_cell.h>
#include <exanb/compute/compute_pair_traits.h>
#include <exanb/core/log.h>
#include <exanb/core/grid_cell_compute_profiler.h>

#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/block_parallel_for.h>

namespace exanb
{
  
  template<class CellsT, class FuncT, class OptionalArgsT, class ComputePairBufferFactoryT, class CSizeT, class FieldSetT, class PosFieldsT>
  struct ComputeParticlePairFunctor
  {
    CellsT m_cells;
    GridCellComputeProfiler m_cell_profiler = { nullptr };

    const IJK m_grid_dims = { 0, 0, 0 };
    const ssize_t m_ghost_layers = 0;
    
    const FuncT m_func;
    const double m_rcut2 = 0.0;
        
    const OptionalArgsT m_optional;
    const ComputePairBufferFactoryT m_cpbuf_factory;

    const CSizeT m_cs;
    const FieldSetT m_cpfields;
    const PosFieldsT m_posfields;
    
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( uint64_t i ) const
    {
      static constexpr typename decltype(m_optional.nbh)::is_symmetrical_t symmetrical;
      static constexpr bool gpu_exec = onika::cuda::gpu_device_execution_t::value ;
      static constexpr onika::BoolConst< gpu_exec ? ( ! ComputePairTraits<FuncT>::BufferLessCompatible ) : ComputePairTraits<FuncT>::ComputeBufferCompatible > prefer_compute_buffer = {}; 

      size_t cell_a = i;
      IJK cell_a_loc = grid_index_to_ijk( m_grid_dims - 2 * m_ghost_layers , i ); ;
      cell_a_loc = cell_a_loc + m_ghost_layers;
      if( m_ghost_layers != 0 )
      {
        cell_a = grid_ijk_to_index( m_grid_dims , cell_a_loc );
      }
      m_cell_profiler.start_cell_profiling(cell_a);
      compute_cell_particle_pairs_cell( m_cells, m_grid_dims, cell_a_loc, cell_a, m_rcut2, m_cpbuf_factory, m_optional, m_func, m_cs, symmetrical, m_cpfields, m_posfields , prefer_compute_buffer );
      m_cell_profiler.end_cell_profiling(cell_a);
    }
  };

}

namespace onika
{
  namespace parallel
  {
    template<class CellsT, class FuncT, class OptionalArgsT, class ComputePairBufferFactoryT, class CSizeT, class FieldSetT, class PosFieldsT>
    struct BlockParallelForFunctorTraits< exanb::ComputeParticlePairFunctor<CellsT,FuncT,OptionalArgsT,ComputePairBufferFactoryT,CSizeT,FieldSetT,PosFieldsT> >
    {
      static inline constexpr bool CudaCompatible = exanb::ComputePairTraits<FuncT>::CudaCompatible;
    };
  }
}

namespace exanb
{

  template<class CellsT, class FuncT, class OptionalArgsT, class ComputePairBufferFactoryT, class CSizeT, class FieldSetT, class PosFieldsT>
  static inline
  ComputeParticlePairFunctor<CellsT,FuncT,OptionalArgsT,ComputePairBufferFactoryT,CSizeT,FieldSetT,PosFieldsT> 
  make_compute_particle_pair_functor(
      CellsT cells
    , GridCellComputeProfiler cell_profiler
    , const IJK& grid_dims
    , ssize_t ghost_layers
    , const FuncT& func
    , double rcut2
    , const OptionalArgsT& optional
    , const ComputePairBufferFactoryT& cpbuf_factory
    , CSizeT cs
    , FieldSetT cpfields
    , PosFieldsT posfields
    )
  {
    return {cells,cell_profiler,grid_dims,ghost_layers,func,rcut2,optional,cpbuf_factory,cs,cpfields,posfields};
  }

  // ==== OpenMP parallel for style impelmentation ====
  // cells are dispatched to threads using a "#pragma omp parallel for" construct
  template<class GridT, class OptionalArgsT, class ComputePairBufferFactoryT, class FuncT, class FieldSetT , class PosFieldsT = DefaultPositionFields >
  static inline
  onika::parallel::ParallelExecutionWrapper
  compute_cell_particle_pairs(
    GridT& grid,
    double rcut,
    bool enable_ghosts,
    const OptionalArgsT& optional,
    const ComputePairBufferFactoryT& cpbuf_factory,
    const FuncT& func,
    FieldSetT cpfields,
    onika::parallel::ParallelExecutionContext * exec_ctx ,
    PosFieldsT posfields = PosFieldsT{}
    )
  {
    using onika::parallel::BlockParallelForOptions;
    using onika::parallel::block_parallel_for;

    static constexpr onika::IntConst<4> const_4{};
    static constexpr onika::IntConst<8> const_8{};

    const double rcut2 = rcut * rcut;
    const IJK dims = grid.dimension();
    int gl = grid.ghost_layers();
    if( enable_ghosts ) { gl = 0; }
    const IJK block_dims = dims - (2*gl);
    const size_t N = block_dims.i * block_dims.j * block_dims.k;

    // for debugging purposes
    ComputePairDebugTraits<FuncT>::print_func( func );

    if( ComputePairTraits<FuncT>::CudaCompatible )
    {
      if( exec_ctx->has_gpu_context() )
      {
        if( exec_ctx->m_cuda_ctx->has_devices() ) grid.check_cells_are_gpu_addressable();
      }
    }

    auto cells = grid.cells();
    auto cellprof = grid.cell_profiler();
    const unsigned int cs = optional.nbh.m_chunk_size;
    switch( cs )
    {
      case 4:
        return block_parallel_for( N, make_compute_particle_pair_functor(cells,cellprof,dims,gl,func,rcut2,optional,cpbuf_factory,const_4,cpfields,posfields) , exec_ctx );
        break;
      case 8:
        return block_parallel_for( N, make_compute_particle_pair_functor(cells,cellprof,dims,gl,func,rcut2,optional,cpbuf_factory,const_8,cpfields,posfields) , exec_ctx );
        break;
      default:
        return block_parallel_for( N, make_compute_particle_pair_functor(cells,cellprof,dims,gl,func,rcut2,optional,cpbuf_factory,     cs,cpfields,posfields) , exec_ctx );
        break;
    }
  }

}

