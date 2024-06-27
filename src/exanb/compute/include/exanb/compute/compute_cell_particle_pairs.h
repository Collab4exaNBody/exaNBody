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

#include <exanb/compute/compute_cell_particle_pairs_common.h>
#include <exanb/compute/compute_cell_particle_pairs_chunk.h>

#ifdef XNB_USE_CS1_SPECIALIZATION
#include <exanb/compute/compute_cell_particle_pairs_chunk_cs1.h>
#endif

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
      static constexpr typename decltype(m_optional.nbh)::is_symmetrical_t symmetrical = {};      
      static constexpr bool gpu_exec = onika::cuda::gpu_device_execution_t::value ;
      static constexpr onika::BoolConst< gpu_exec ? ( ! compute_pair_traits::buffer_less_compatible_v<FuncT> ) : compute_pair_traits::compute_buffer_compatible_v<FuncT> > prefer_compute_buffer = {}; 

      size_t cell_a = i;
      IJK cell_a_loc = grid_index_to_ijk( m_grid_dims - 2 * m_ghost_layers , i ); ;
      cell_a_loc = cell_a_loc + m_ghost_layers;
      if( m_ghost_layers != 0 )
      {
        cell_a = grid_ijk_to_index( m_grid_dims , cell_a_loc );
      }
      m_cell_profiler.start_cell_profiling(cell_a);
      compute_cell_particle_pairs_cell( m_cells, m_grid_dims, cell_a_loc, cell_a, m_rcut2
                                      , m_cpbuf_factory, m_optional, m_func
                                      , m_cpfields, m_cs, symmetrical, m_posfields
                                      , prefer_compute_buffer, std::index_sequence<FieldIndex...>{} );
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
      static inline constexpr bool CudaCompatible = exanb::compute_pair_traits::cuda_compatible_v<FuncT>;
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

  template<class GridT, class OptionalArgsT, class ComputePairBufferFactoryT, class FuncT, class PosFieldsT, bool ForceUseCellsAccessor=false,class... FieldAccT >
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
    onika::parallel::ParallelExecutionContext * exec_ctx,
    std::integral_constant<bool,ForceUseCellsAccessor> = {} )
  {
    static_assert( is_compute_buffer_factory_v<ComputePairBufferFactoryT> , "Only ComputePairBufferFactory<...> template instance is accepted as cpbuf_factory parameter" );

    using onika::parallel::BlockParallelForOptions;
    using onika::parallel::block_parallel_for;
    using FieldTupleT = onika::FlatTuple<FieldAccT...>;
    using CellsPointerT = decltype(grid.cells()); // typename GridT::CellParticles;
    static constexpr bool has_external_or_optional_fields = ForceUseCellsAccessor || field_tuple_has_external_fields_v<FieldTupleT> || field_tuple_has_external_fields_v<PosFieldsT>;    
    using CellsAccessorT = std::conditional_t< has_external_or_optional_fields , std::remove_cv_t<std::remove_reference_t<decltype(grid.cells_accessor())> > , CellsPointerT >;
    static constexpr bool requires_block_synchronous_call = compute_pair_traits::requires_block_synchronous_call_v<FuncT> ;

    const double rcut2 = rcut * rcut;
    const IJK dims = grid.dimension();
    int gl = grid.ghost_layers();
    if( enable_ghosts ) { gl = 0; }
    const IJK block_dims = dims - (2*gl);
    const size_t N = block_dims.i * block_dims.j * block_dims.k;

    // for debugging purposes
    ComputePairDebugTraits<FuncT>::print_func( func );

    BlockParallelForOptions bpfor_opts = {};
    if constexpr ( compute_pair_traits::cuda_compatible_v<FuncT> )
    {
      if( exec_ctx->has_gpu_context() )
      {
        if( exec_ctx->m_cuda_ctx->has_devices() )
        {
          if( !requires_block_synchronous_call && optional.nbh.m_chunk_size>1 && compute_pair_traits::buffer_less_compatible_v<FuncT> )
          {
            if( XNB_CHUNK_NBH_DELAYED_COMPUTE_MAX_BLOCK_SIZE < bpfor_opts.max_block_size )
            {
              ldbg << "INFO: GPU block size has been limited to "<< XNB_CHUNK_NBH_DELAYED_COMPUTE_MAX_BLOCK_SIZE
                   <<" to enforce synchronized thread computation in kernel "<<exec_ctx->tag()<<" "<<exec_ctx->sub_tag() << std::endl;
              bpfor_opts.max_block_size = XNB_CHUNK_NBH_DELAYED_COMPUTE_MAX_BLOCK_SIZE;
            }
          }
          grid.check_cells_are_gpu_addressable();
        }
      }
    }
        
    auto cellprof = grid.cell_profiler();
    CellsAccessorT cells = {};
    if constexpr ( has_external_or_optional_fields ) cells = grid.cells_accessor();
    else cells = grid.cells();
        
#   define _XNB_CHUNK_NEIGHBORS_CCPP(CS) \
    { XNB_CHUNK_NEIGHBORS_CS_VAR( CS , cs , optional.nbh.m_chunk_size ); \
      if ( static_cast<unsigned int>(cs) == optional.nbh.m_chunk_size ) \
        return block_parallel_for( N, make_compute_particle_pair_functor(cells,cellprof,dims,gl,func,rcut2,optional,cpbuf_factory,cpfields,posfields,cs) , exec_ctx , bpfor_opts ); \
    }
    XNB_CHUNK_NEIGHBORS_CS_SPECIALIZE( _XNB_CHUNK_NEIGHBORS_CCPP )
#   undef _XNB_CHUNK_NEIGHBORS_CCPP
    fatal_error()<< "compute_cell_particle_pairs: neighbors configuration not supported : chunk_size=" << optional.nbh.m_chunk_size <<std::endl;
    return {};
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

