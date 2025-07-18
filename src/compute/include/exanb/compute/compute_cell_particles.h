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

#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_error.h>
#include <onika/cuda/device_storage.h>
#include <exanb/core/grid_particle_field_accessor.h>
//#include <exanb/core/grid.h>

#include <exanb/core/parallel_grid_algorithm.h>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/block_parallel_for.h>
#include <onika/lambda_tools.h>
#include <onika/flat_tuple.h>
#include <utility>

#ifdef ONIKA_OMP_NUM_THREADS_WORKAROUND
#include <omp.h>
#endif


namespace exanb
{

  // this template is here to know if compute buffer must be built or computation must be ran on the fly
  template<class FuncT> struct ComputeCellParticlesTraits
  {
    //static inline constexpr bool RequiresBlockSynchronousCall = false; // nonsense for particles, we cannot call functor with 'fake particle fields' while we don't pass an array which size coudle be 0
    static inline constexpr bool CudaCompatible = false;
  };

  template<class FuncT> struct ComputeCellParticlesTraitsUseCellIdx
  {
    static inline constexpr bool UseCellIdx = false;
  };
  
  // this template is here to know if ComputeCellParticles iterates only on filled particles
  template<class FuncT> struct ComputeCellParticlesEmptyCells
  {
    static inline constexpr bool EmptyCells = false;
  };


  template<class CellsT, class FuncT, class FieldAccTupleT , class IndexSequence> struct ComputeCellParticlesFunctor;

  template<class CellsT, class FuncT, class FieldAccTupleT, size_t ... FieldIndex >
  struct ComputeCellParticlesFunctor< CellsT, FuncT, FieldAccTupleT , std::index_sequence<FieldIndex...> >
  {
    static_assert( FieldAccTupleT::size() == sizeof...(FieldIndex) );
    CellsT m_cells;
    const IJK m_grid_dims = { 0, 0, 0 };
    const FuncT m_func;
    FieldAccTupleT m_cpfields;
    
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( onikaInt3_t coord ) const
    {
      using onika::lambda_is_compatible_with_v;
      static constexpr bool call_func_without_idx = lambda_is_compatible_with_v<FuncT,void, decltype( m_cells[0][m_cpfields.get(onika::tuple_index_t<FieldIndex>{})][0] ) ... >;
      static constexpr bool call_func_with_cell_idx = lambda_is_compatible_with_v<FuncT,void, size_t, decltype( m_cells[0][m_cpfields.get(onika::tuple_index_t<FieldIndex>{})][0] ) ... >;
      static constexpr bool call_func_with_cell_particle_idx = lambda_is_compatible_with_v<FuncT,void, size_t, unsigned int, decltype( m_cells[0][m_cpfields.get(onika::tuple_index_t<FieldIndex>{})][0] ) ... >;
      
    	IJK cell_a_loc = { coord.x , coord.y , coord.z };
    	size_t cell_a = grid_ijk_to_index( m_grid_dims , cell_a_loc );
      
      const unsigned int n = m_cells[cell_a].size();
      //if( ONIKA_CU_THREAD_IDX == 0 ) printf("GPU: cell particles functor: cell #%d @%d,%d,%d : %d particles\n",int(cell_a),int(cell_a_loc.i),int(cell_a_loc.j),int(cell_a_loc.k),int(n));
      ONIKA_CU_BLOCK_SIMD_FOR(unsigned int , p , 0 , n )
      {
			  if constexpr ( call_func_without_idx )
				{
          static_assert( ! ComputeCellParticlesTraitsUseCellIdx<FuncT>::UseCellIdx );
	        m_func( m_cells[cell_a][m_cpfields.get(onika::tuple_index_t<FieldIndex>{})][p] ... );
				}
				else if constexpr ( call_func_with_cell_idx )
				{
	        m_func(cell_a, m_cells[cell_a][m_cpfields.get(onika::tuple_index_t<FieldIndex>{})][p] ... );
				}
				else if constexpr ( call_func_with_cell_particle_idx )
				{
	        m_func(cell_a, p, m_cells[cell_a][m_cpfields.get(onika::tuple_index_t<FieldIndex>{})][p] ... );
				}
      }
    }
 
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( uint64_t i ) const
    {
    	IJK cell_a_loc = grid_index_to_ijk( m_grid_dims , i ); ;
      this->operator() ( { cell_a_loc.i , cell_a_loc.j , cell_a_loc.k } );
    }
    
  };

}

namespace onika
{
  namespace parallel
  {
    template<class CellsT, class FuncT, class FieldTupleT, class IndexSequenceT>
    struct BlockParallelForFunctorTraits< exanb::ComputeCellParticlesFunctor<CellsT,FuncT,FieldTupleT,IndexSequenceT> >
    {
      static inline constexpr bool CudaCompatible = exanb::ComputeCellParticlesTraits<FuncT>::CudaCompatible;
    };
  }
}

namespace exanb
{

  struct ComputeCellParticlesOptions
  {
    static inline constexpr size_t NO_CELL_INDICES = std::numeric_limits<size_t>::max();
    // -1 means all cells are processed and m_cell_indices is ignored.
    // 0 or greater value indicates you want a 1D kernel using indices stored in m_num_cell_indices.
    size_t m_num_cell_indices = NO_CELL_INDICES;
    const size_t * m_cell_indices = nullptr;

    // 0 means no override of BlockParallelForOptions' max_block_size parameter
    size_t m_max_block_size = 0;
  };

  template<class GridT, class FuncT, class FieldSetT>
  static inline
  onika::parallel::ParallelExecutionWrapper
  compute_cell_particles(
    GridT& grid,
    bool enable_ghosts,
    const FuncT& func,
    FieldSetT cpfields ,
    onika::parallel::ParallelExecutionContext * exec_ctx );

  template<class GridT, class FuncT, class... FieldAccT>
  static inline
  onika::parallel::ParallelExecutionWrapper
  compute_cell_particles(
    GridT& grid,
    bool enable_ghosts,
    const FuncT& func,
    const onika::FlatTuple<FieldAccT...>& cpfields ,
    onika::parallel::ParallelExecutionContext * exec_ctx )
  {
    using onika::parallel::BlockParallelForOptions;
    using onika::parallel::block_parallel_for;   
    using CellsPointerT = decltype(grid.cells()); // typename GridT::CellParticles;
    using FieldTupleT = onika::FlatTuple<FieldAccT...>;
    static constexpr bool has_external_or_optional_fields = field_tuple_has_external_fields_v<FieldTupleT>;    
    using CellsAccessorT = std::conditional_t< has_external_or_optional_fields , std::remove_cv_t<std::remove_reference_t<decltype(grid.cells_accessor())> > , CellsPointerT >;
    using PForFuncT = ComputeCellParticlesFunctor<CellsAccessorT,FuncT,FieldTupleT,std::make_index_sequence<sizeof...(FieldAccT)> >;
    
  	const IJK dims = grid.dimension();
  	const int gl = enable_ghosts ? 0 : grid.ghost_layers();	
    	    
    CellsAccessorT cells = {};
    if constexpr ( has_external_or_optional_fields ) cells = grid.cells_accessor();
    else cells = grid.cells();

    onika::parallel::ParallelExecutionSpace<3> parallel_range = { { gl , gl , gl } , { dims.i-gl , dims.j-gl , dims.k-gl } }; 
    PForFuncT pfor_func = { cells , dims , func , cpfields };
    return block_parallel_for( parallel_range, pfor_func, exec_ctx );
  }

  template<class GridT, class FuncT, class... FieldAccT>
  static inline
  onika::parallel::ParallelExecutionWrapper
  compute_cell_particles(
    GridT& grid,
    bool enable_ghosts,
    const FuncT& func,
    const onika::FlatTuple<FieldAccT...>& cpfields ,
    onika::parallel::ParallelExecutionContext * exec_ctx,
    const size_t * cell_indices,
    size_t number_cell_indices )
  {
    using onika::parallel::BlockParallelForOptions;
    using onika::parallel::block_parallel_for;   
    using CellsPointerT = decltype(grid.cells()); // typename GridT::CellParticles;
    using FieldTupleT = onika::FlatTuple<FieldAccT...>;
    static constexpr bool has_external_or_optional_fields = field_tuple_has_external_fields_v<FieldTupleT>;    
    using CellsAccessorT = std::conditional_t< has_external_or_optional_fields , std::remove_cv_t<std::remove_reference_t<decltype(grid.cells_accessor())> > , CellsPointerT >;
    using PForFuncT = ComputeCellParticlesFunctor<CellsAccessorT,FuncT,FieldTupleT,std::make_index_sequence<sizeof...(FieldAccT)> >;
    
    if( number_cell_indices>0 && cell_indices==nullptr )
    {
      fatal_error() << "compute_cell_particles: cell_indices cannot be NULL if number_cell_indices > 0" << std::endl;
    }

  	const IJK dims = grid.dimension();
  	const int gl = enable_ghosts ? 0 : grid.ghost_layers();	
    CellsAccessorT cells = {};
    if constexpr ( has_external_or_optional_fields ) cells = grid.cells_accessor();
    else cells = grid.cells();

    using CellListT = std::span<const size_t>; // default span definition would use ssize_t and prevent constructor match with size_t* pointer
    using PES = onika::parallel::ParallelExecutionSpace<1,1,CellListT>;
    PES parallel_range = { {0} , {number_cell_indices} , CellListT{cell_indices,number_cell_indices} }; 
    PForFuncT pfor_func = { cells , dims , func , cpfields };
    return block_parallel_for( parallel_range, pfor_func, exec_ctx );
  }

  template<class GridT, class FuncT, class... FieldAccT>
  static inline
  onika::parallel::ParallelExecutionWrapper
  compute_cell_particles(
    GridT& grid,
    bool enable_ghosts,
    const FuncT& func,
    const onika::FlatTuple<FieldAccT...>& cpfields ,
    onika::parallel::ParallelExecutionContext * exec_ctx,
    const ComputeCellParticlesOptions& ccpo  )
  {
    using onika::parallel::BlockParallelForOptions;
    using onika::parallel::block_parallel_for;   
    using CellsPointerT = decltype(grid.cells()); // typename GridT::CellParticles;
    using FieldTupleT = onika::FlatTuple<FieldAccT...>;
    static constexpr bool has_external_or_optional_fields = field_tuple_has_external_fields_v<FieldTupleT>;    
    using CellsAccessorT = std::conditional_t< has_external_or_optional_fields , std::remove_cv_t<std::remove_reference_t<decltype(grid.cells_accessor())> > , CellsPointerT >;
    using PForFuncT = ComputeCellParticlesFunctor<CellsAccessorT,FuncT,FieldTupleT,std::make_index_sequence<sizeof...(FieldAccT)> >;
    
    if( ccpo.m_num_cell_indices>0 && ccpo.m_cell_indices==nullptr )
    {
      fatal_error() << "compute_cell_particles: cell_indices cannot be NULL if number_cell_indices > 0" << std::endl;
    }
    if( ccpo.m_num_cell_indices == ComputeCellParticlesOptions::NO_CELL_INDICES && ccpo.m_cell_indices != nullptr )
    {
      fatal_error() << "compute_cell_particles: cell_indices ignored but cell_indices!=nullptr" << std::endl;
    }

  	const IJK dims = grid.dimension();
  	const int gl = enable_ghosts ? 0 : grid.ghost_layers();	
    CellsAccessorT cells = {};
    if constexpr ( has_external_or_optional_fields ) cells = grid.cells_accessor();
    else cells = grid.cells();

    BlockParallelForOptions pfor_opts = {};
    if( ccpo.m_max_block_size > 0 ) pfor_opts.max_block_size = ccpo.m_max_block_size;

    PForFuncT pfor_func = { cells , dims , func , cpfields };

    if( ccpo.m_num_cell_indices == ComputeCellParticlesOptions::NO_CELL_INDICES )
    {
      onika::parallel::ParallelExecutionSpace<3> parallel_range = { { gl , gl , gl } , { dims.i-gl , dims.j-gl , dims.k-gl } }; 
      return block_parallel_for( parallel_range, pfor_func, exec_ctx, pfor_opts );
    }
    else
    {
      using CellListT = std::span<const size_t>; // default span definition would use ssize_t and prevent constructor match with size_t* pointer
      onika::parallel::ParallelExecutionSpace<1,1,CellListT> parallel_range = { {0} , {ccpo.m_num_cell_indices} , CellListT{ccpo.m_cell_indices,ccpo.m_num_cell_indices} }; 
      return block_parallel_for( parallel_range, pfor_func, exec_ctx, pfor_opts );
    }
  }


  // ========== old API ==================

  template<class GridT, class FuncT, class... field_ids>
  static inline
  onika::parallel::ParallelExecutionWrapper
  compute_cell_particles(
    GridT& grid,
    bool enable_ghosts,
    const FuncT& func,
    FieldSet<field_ids...> cpfields ,
    onika::parallel::ParallelExecutionContext * exec_ctx,
    const size_t * cell_indices,
    size_t number_cell_indices)
  {
    using FieldTupleT = onika::FlatTuple< onika::soatl::FieldId<field_ids> ... >;
    FieldTupleT cp_fields = { onika::soatl::FieldId<field_ids>{} ... };
    return compute_cell_particles(grid,enable_ghosts,func,cp_fields,exec_ctx, cell_indices, number_cell_indices);
  }


  template<class GridT, class FuncT, class... field_ids>
  static inline
  onika::parallel::ParallelExecutionWrapper
  compute_cell_particles(
    GridT& grid,
    bool enable_ghosts,
    const FuncT& func,
    FieldSet<field_ids...> cpfields ,
    onika::parallel::ParallelExecutionContext * exec_ctx )
  {
    using FieldTupleT = onika::FlatTuple< onika::soatl::FieldId<field_ids> ... >;
    FieldTupleT cp_fields = { onika::soatl::FieldId<field_ids>{} ... };
    return compute_cell_particles(grid,enable_ghosts,func,cp_fields,exec_ctx);
  }

}

