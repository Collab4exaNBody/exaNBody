#pragma once

#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_error.h>
#include <onika/cuda/device_storage.h>
#include <onika/soatl/field_id.h>

#include <exanb/core/parallel_grid_algorithm.h>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/block_parallel_for.h>

#ifdef XSTAMP_OMP_NUM_THREADS_WORKAROUND
#include <omp.h>
#endif

namespace exanb
{

  struct reduce_thread_local_t {};
  struct reduce_thread_block_t {};
  struct reduce_global_t {};

  // this template is here to know if compute buffer must be built or computation must be ran on the fly
  template<class FuncT> struct ReduceCellParticlesTraits
  {
    //static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool RequiresCellParticleIndex = false;
    static inline constexpr bool CudaCompatible = false;
  };

  template<class CellsT, class FuncT, class ResultT, class FieldSetT> struct ReduceCellParticlesFunctor;

  template<class CellsT, class FuncT, class ResultT, class... field_ids>
  struct ReduceCellParticlesFunctor< CellsT, FuncT, ResultT, FieldSet<field_ids...> >
  {
    CellsT m_cells;
    const IJK m_grid_dims = { 0, 0, 0 };
    const ssize_t m_ghost_layers = 0;
    const FuncT m_func;
    ResultT* reduced_val = nullptr;

    ONIKA_HOST_DEVICE_FUNC inline void operator () ( uint64_t i ) const
    {
      ResultT local_val = ResultT();

      size_t cell_a = i;
      IJK cell_a_loc = grid_index_to_ijk( m_grid_dims - 2 * m_ghost_layers , i ); ;
      cell_a_loc = cell_a_loc + m_ghost_layers;
      if( m_ghost_layers != 0 )
      {
        cell_a = grid_ijk_to_index( m_grid_dims , cell_a_loc );
      }
      const unsigned int n = m_cells[cell_a].size();

      ONIKA_CU_BLOCK_SIMD_FOR(unsigned int , p , 0 , n )
      {
        if constexpr ( ReduceCellParticlesTraits<FuncT>::RequiresCellParticleIndex )
        {
          m_func( local_val, cell_a_loc, cell_a , p , m_cells[cell_a][onika::soatl::FieldId<field_ids>{}][p] ... , reduce_thread_local_t{} );
        }
        if constexpr ( ! ReduceCellParticlesTraits<FuncT>::RequiresCellParticleIndex )
        {
          m_func( local_val, m_cells[cell_a][onika::soatl::FieldId<field_ids>{}][p] ... , reduce_thread_local_t{} );
        }
      }
      
      ONIKA_CU_BLOCK_SHARED ResultT team_val;
      if( ONIKA_CU_THREAD_IDX == 0 ) { team_val = local_val; }
      ONIKA_CU_BLOCK_SYNC();

      if( ONIKA_CU_THREAD_IDX != 0 ) 
      {
        m_func( team_val, local_val , reduce_thread_block_t{} );
      }
      ONIKA_CU_BLOCK_SYNC();

      if( ONIKA_CU_THREAD_IDX == 0 )
      {
        m_func( *reduced_val, team_val , reduce_global_t{} );      
      }
      ONIKA_CU_BLOCK_SYNC();
    }
  };
  
}

namespace onika
{
  namespace parallel
  {
    template<class CellsT, class FuncT, class ResultT, class FieldSetT>
    struct BlockParallelForFunctorTraits< exanb::ReduceCellParticlesFunctor<CellsT,FuncT,ResultT,FieldSetT> >
    {
      static inline constexpr bool CudaCompatible = exanb::ReduceCellParticlesTraits<FuncT>::CudaCompatible;
    };
  }
}

namespace exanb
{
  // ==== OpenMP parallel for style implementation ====
  // cells are dispatched to threads using a "#pragma omp parallel for" construct
  template<class GridT, class FuncT, class ResultT, class FieldSetT>
  static inline
  onika::parallel::ParallelExecutionWrapper
  reduce_cell_particles(
    GridT& grid,
    bool enable_ghosts ,
    const FuncT& func  ,
    ResultT& reduced_val , // initial value is used as a start value for reduction
    FieldSetT cpfields ,
    onika::parallel::ParallelExecutionContext * exec_ctx ,
    onika::parallel::ParallelExecutionCallback user_cb = {} )
  {
    using onika::parallel::block_parallel_for;
    using ParForOpts = onika::parallel::BlockParallelForOptions;
    using CellsT = typename GridT::CellParticles;
    const IJK dims = grid.dimension();
    const int gl = enable_ghosts ? 0 : grid.ghost_layers();
    const IJK block_dims = dims - (2*gl);
    const size_t N = block_dims.i * block_dims.j * block_dims.k;

    ResultT* target_reduced_value_ptr = &reduced_val;
    CellsT * cells = grid.cells();
    if constexpr ( ReduceCellParticlesTraits<FuncT>::CudaCompatible )
    {
      if( exec_ctx->has_gpu_context() && exec_ctx->m_cuda_ctx->has_devices() )
      {
        grid.check_cells_are_gpu_addressable();
        exec_ctx->init_device_scratch();
        target_reduced_value_ptr = (ResultT*) exec_ctx->get_device_return_data_ptr();
      }
    }
    
    ReduceCellParticlesFunctor<CellsT*,FuncT,ResultT,FieldSetT> pfor_op = { cells , dims , gl , func , target_reduced_value_ptr };
    return block_parallel_for( N, pfor_op , exec_ctx , ParForOpts{ .user_cb = user_cb , .return_data = &reduced_val, .return_data_size = sizeof(ResultT) } );
  }

}

