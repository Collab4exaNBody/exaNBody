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
#include <exanb/core/grid_particle_field_accessor.h>

#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/block_parallel_for.h>

namespace exanb
{
  template<class CellsT, class FuncT, class OptionalArgsT, class ComputePairBufferFactoryT, class FieldAccTupleT, class PosFieldsT, class CSizeT, class IndexSequence>
  struct ComputeParticlePairFunctor;
 
  template<class CellsT, class FuncT, class OptionalArgsT, class ComputePairBufferFactoryT, class FieldAccTupleT, class PosFieldsT, class CSizeT, size_t ... FieldIndex >
  struct ComputeParticlePairFunctor<CellsT,FuncT,OptionalArgsT,ComputePairBufferFactoryT,FieldAccTupleT,PosFieldsT,CSizeT, std::index_sequence<FieldIndex...> >
  {
    static_assert( FieldAccTupleT::size() == sizeof...(FieldIndex) );

    CellsT m_cells;
    GridCellComputeProfiler m_cell_profiler = { nullptr };

    const IJK m_grid_dims = { 0, 0, 0 };
    const ssize_t m_ghost_layers = 0;
    
    const FuncT m_func;
    const double m_rcut2 = 0.0;
        
    const OptionalArgsT m_optional;
    const ComputePairBufferFactoryT m_cpbuf_factory;

    const FieldAccTupleT m_cpfields;
    const PosFieldsT m_posfields;
    const CSizeT m_cs;
    
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
      compute_cell_particle_pairs_cell( m_cells, m_grid_dims, cell_a_loc, cell_a, m_rcut2, m_cpbuf_factory, m_optional, m_func, m_cpfields, m_cs, symmetrical, m_posfields , prefer_compute_buffer, std::index_sequence<FieldIndex...>{} );
      m_cell_profiler.end_cell_profiling(cell_a);
    }
  };

}

namespace onika
{
  namespace parallel
  {
    template<class CellsT, class FuncT, class OptionalArgsT, class ComputePairBufferFactoryT, class FieldAccTupleT, class PosFieldsT, class CSizeT, class ISeq>
    struct BlockParallelForFunctorTraits< exanb::ComputeParticlePairFunctor<CellsT,FuncT,OptionalArgsT,ComputePairBufferFactoryT,FieldAccTupleT,PosFieldsT,CSizeT,ISeq> >
    {
      static inline constexpr bool CudaCompatible = exanb::ComputePairTraits<FuncT>::CudaCompatible;
    };
  }
}

namespace exanb
{

  template<class CellsT, class FuncT, class OptionalArgsT, class ComputePairBufferFactoryT, class FieldAccTupleT, class PosFieldsT, class CSizeT>
  static inline
  ComputeParticlePairFunctor<CellsT,FuncT,OptionalArgsT,ComputePairBufferFactoryT,FieldAccTupleT,PosFieldsT,CSizeT,std::make_index_sequence<FieldAccTupleT::size()> > 
  make_compute_particle_pair_functor(
      CellsT cells
    , GridCellComputeProfiler cell_profiler
    , const IJK& grid_dims
    , ssize_t ghost_layers
    , const FuncT& func
    , double rcut2
    , const OptionalArgsT& optional
    , const ComputePairBufferFactoryT& cpbuf_factory
    , const FieldAccTupleT& cpfields
    , const PosFieldsT& posfields
    , CSizeT cs
    )
  {
    return {cells,cell_profiler,grid_dims,ghost_layers,func,rcut2,optional,cpbuf_factory,cpfields,posfields,cs};
  }

  // ==== OpenMP parallel for style impelmentation ====
  // cells are dispatched to threads using a "#pragma omp parallel for" construct

  template<class GridT, class OptionalArgsT, class ComputePairBufferFactoryT, class FuncT, class PosFieldsT, class... FieldAccT >
  static inline
  onika::parallel::ParallelExecutionWrapper
  compute_cell_particle_pairs2(
    GridT& grid,
    double rcut,
    bool enable_ghosts,
    const OptionalArgsT& optional,
    const ComputePairBufferFactoryT& cpbuf_factory,
    const FuncT& func,
    const onika::FlatTuple<FieldAccT...>& cpfields,
    const PosFieldsT& posfields,
    onika::parallel::ParallelExecutionContext * exec_ctx )
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

    // using CellsT = typename GridT::CellParticles;
    // GridParticleFieldAccessor< CellsT * const > cells = { grid.cells() };
    auto * const cells = grid.cells();

    auto cellprof = grid.cell_profiler();
    const unsigned int cs = optional.nbh.m_chunk_size;
    switch( cs )
    {
      case 4:
        return block_parallel_for( N, make_compute_particle_pair_functor(cells,cellprof,dims,gl,func,rcut2,optional,cpbuf_factory,cpfields,posfields,const_4) , exec_ctx );
        break;
      case 8:
        return block_parallel_for( N, make_compute_particle_pair_functor(cells,cellprof,dims,gl,func,rcut2,optional,cpbuf_factory,cpfields,posfields,const_8) , exec_ctx );
        break;
      default:
        return block_parallel_for( N, make_compute_particle_pair_functor(cells,cellprof,dims,gl,func,rcut2,optional,cpbuf_factory,cpfields,posfields,     cs) , exec_ctx );
        break;
    }
  }

  template<class GridT, class OptionalArgsT, class ComputePairBufferFactoryT, class FuncT, class... FieldAccT >
  static inline
  onika::parallel::ParallelExecutionWrapper
  compute_cell_particle_pairs(
    GridT& grid,
    double rcut,
    bool enable_ghosts,
    const OptionalArgsT& optional,
    const ComputePairBufferFactoryT& cpbuf_factory,
    const FuncT& func,
    const onika::FlatTuple<FieldAccT...>& cp_fields,
    onika::parallel::ParallelExecutionContext * exec_ctx )
  {
    return compute_cell_particle_pairs2(grid,rcut,enable_ghosts,optional,cpbuf_factory,func,cp_fields,DefaultPositionFields{},exec_ctx);
  }

  template<class GridT, class OptionalArgsT, class ComputePairBufferFactoryT, class FuncT, class... field_ids >
  static inline
  onika::parallel::ParallelExecutionWrapper
  compute_cell_particle_pairs(
    GridT& grid,
    double rcut,
    bool enable_ghosts,
    const OptionalArgsT& optional,
    const ComputePairBufferFactoryT& cpbuf_factory,
    const FuncT& func,
    FieldSet<field_ids...> ,
    onika::parallel::ParallelExecutionContext * exec_ctx )
  {
    onika::FlatTuple< onika::soatl::FieldId<field_ids> ... > cp_fields = { onika::soatl::FieldId<field_ids>{} ... };
    return compute_cell_particle_pairs2(grid,rcut,enable_ghosts,optional,cpbuf_factory,func,cp_fields,DefaultPositionFields{},exec_ctx);
  }

  template<class GridT, class OptionalArgsT, class ComputePairBufferFactoryT, class FuncT, class PosFieldsT, class... field_ids >
  static inline
  onika::parallel::ParallelExecutionWrapper
  compute_cell_particle_pairs(
    GridT& grid,
    double rcut,
    bool enable_ghosts,
    const OptionalArgsT& optional,
    const ComputePairBufferFactoryT& cpbuf_factory,
    const FuncT& func,
    FieldSet<field_ids...> ,
    PosFieldsT posfields,
    onika::parallel::ParallelExecutionContext * exec_ctx )
  {
    onika::FlatTuple< onika::soatl::FieldId<field_ids> ... > cp_fields = { onika::soatl::FieldId<field_ids>{} ... };
    return compute_cell_particle_pairs2(grid,rcut,enable_ghosts,optional,cpbuf_factory,func,cp_fields,posfields,exec_ctx);
  }

  // support for alternative argument ordering
  template<class GridT, class OptionalArgsT, class ComputePairBufferFactoryT, class FuncT, class PosFieldsT, class... field_ids >
  static inline
  onika::parallel::ParallelExecutionWrapper
  compute_cell_particle_pairs(
    GridT& grid,
    double rcut,
    bool enable_ghosts,
    const OptionalArgsT& optional,
    const ComputePairBufferFactoryT& cpbuf_factory,
    const FuncT& func,
    FieldSet<field_ids...> ,
    onika::parallel::ParallelExecutionContext * exec_ctx,
    PosFieldsT posfields )
  {
    onika::FlatTuple< onika::soatl::FieldId<field_ids> ... > cp_fields = { onika::soatl::FieldId<field_ids>{} ... };
    return compute_cell_particle_pairs2(grid,rcut,enable_ghosts,optional,cpbuf_factory,func,cp_fields,posfields,exec_ctx);
  }

}

