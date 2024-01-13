#pragma once

#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_error.h>
#include <onika/cuda/device_storage.h>
#include <onika/soatl/field_id.h>
#include <exanb/field_sets.h>

#include <exanb/core/parallel_grid_algorithm.h>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/block_parallel_for.h>

#ifdef XSTAMP_OMP_NUM_THREADS_WORKAROUND
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

  template<class CellsT, class FuncT, class FieldSetT> struct ComputeCellParticlesFunctor;

  template<class CellsT, class FuncT, class... field_ids>
  struct ComputeCellParticlesFunctor< CellsT, FuncT, FieldSet<field_ids...> >
  {
    CellsT m_cells;
    const IJK m_grid_dims = { 0, 0, 0 };
    const ssize_t m_ghost_layers = 0;
    const FuncT m_func;
    //const FieldSet<field_ids...> m_cpfields;
    
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( uint64_t i ) const
    {
      static constexpr bool call_func_without_idx = lambda_is_compatible_with_v<FuncT,void, decltype(m_cells[0][onika::soatl::FieldId<field_ids>{}][0]) ... >;
      static constexpr bool call_func_with_cell_idx = lambda_is_compatible_with_v<FuncT,void, size_t, decltype(m_cells[0][onika::soatl::FieldId<field_ids>{}][0]) ... >;
      static constexpr bool call_func_with_cell_particle_idx = lambda_is_compatible_with_v<FuncT,void, size_t, unsigned int, decltype(m_cells[0][onika::soatl::FieldId<field_ids>{}][0]) ... >;

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
			  if constexpr ( call_func_without_idx )
				{
          static_assert( ! ComputeCellParticlesTraitsUseCellIdx<FuncT>::UseCellIdx );
	        m_func( m_cells[cell_a][onika::soatl::FieldId<field_ids>{}][p] ... );
				}
				else if constexpr ( call_func_with_cell_idx )
				{
	        m_func(cell_a, m_cells[cell_a][onika::soatl::FieldId<field_ids>{}][p] ... );
				}
				else if constexpr ( call_func_with_cell_particle_idx )
				{
	        m_func(cell_a, p, m_cells[cell_a][onika::soatl::FieldId<field_ids>{}][p] ... );
				}
      }
    }
  };

}

namespace onika
{
  namespace parallel
  {
    template<class CellsT, class FuncT, class FieldSetT>
    struct BlockParallelForFunctorTraits< exanb::ComputeCellParticlesFunctor<CellsT,FuncT,FieldSetT> >
    {
      static inline constexpr bool CudaCompatible = exanb::ComputeCellParticlesTraits<FuncT>::CudaCompatible;
    };
  }
}

namespace exanb
{
  template<class GridT, class FuncT, class FieldSetT>
  static inline void compute_cell_particles(
    GridT& grid,
    bool enable_ghosts,
    const FuncT& func,
    FieldSetT cpfields ,
    onika::parallel::ParallelExecutionContext * exec_ctx = nullptr,
    bool async = false )
  {
    using CellsT = typename GridT::CellParticles;
    const IJK dims = grid.dimension();
    const int gl = enable_ghosts ? 0 : grid.ghost_layers();
    const IJK block_dims = dims - (2*gl);
    const size_t N = block_dims.i * block_dims.j * block_dims.k;

    CellsT * cells = grid.cells();
    bool enable_gpu = false;
    if constexpr ( ComputeCellParticlesTraits<FuncT>::CudaCompatible )
    {
      if(exec_ctx!=nullptr && exec_ctx->has_gpu_context() && exec_ctx->m_cuda_ctx->has_devices() )
      {
        grid.check_cells_are_gpu_addressable();
      }
      enable_gpu = true;
    }
    
    onika::parallel::block_parallel_for( N, ComputeCellParticlesFunctor<CellsT*,FuncT,FieldSetT>{cells,dims,gl,func} , exec_ctx , enable_gpu , async );
  }
}

