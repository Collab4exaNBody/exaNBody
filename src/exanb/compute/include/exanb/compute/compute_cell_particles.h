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
      size_t cell_a = i;
      IJK cell_a_loc = grid_index_to_ijk( m_grid_dims - 2 * m_ghost_layers , i ); ;
      cell_a_loc = cell_a_loc + m_ghost_layers;
      if( m_ghost_layers != 0 )
      {
        cell_a = grid_ijk_to_index( m_grid_dims , cell_a_loc );
      }
      const unsigned int n = m_cells[i].size();
      ONIKA_CU_BLOCK_SIMD_FOR(unsigned int , p , 0 , n )
      {
        m_func( m_cells[cell_a][onika::soatl::FieldId<field_ids>{}][p] ... );
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

/*
  template<class CellsT, class FuncT, class... field_ids>
  ONIKA_DEVICE_KERNEL_FUNC
  ONIKA_DEVICE_KERNEL_BOUNDS(ONIKA_CU_MAX_THREADS_PER_BLOCK,ONIKA_CU_MIN_BLOCKS_PER_SM)
  void compute_cell_particles_gpu_kernel(
    CellsT* cells,
    IJK dims,
    unsigned int gl,
    const FuncT func,
    onika::parallel::GPUKernelExecutionScratch* scratch,
    FieldSet<field_ids...> )
  {
    // avoid use of compute buffer when possible
    const IJK dimsNoGL = { dims.i-2*gl , dims.j-2*gl , dims.k-2*gl };
    const uint64_t ncells_no_gl = dimsNoGL.i * dimsNoGL.j * dimsNoGL.k;
    ONIKA_CU_BLOCK_SHARED unsigned int cell_a_no_gl;
    do
    {
      if( ONIKA_CU_THREAD_IDX == 0 )
      {
        cell_a_no_gl = ONIKA_CU_ATOMIC_ADD( scratch->counters[0] , 1u );
        //printf("processing cell #%d\n",int(cell_a_no_gl));
      }
      ONIKA_CU_BLOCK_SYNC();
      if( cell_a_no_gl < ncells_no_gl )
      {
        const IJK loc_a_no_gl = grid_index_to_ijk( dimsNoGL, cell_a_no_gl );
        const IJK loc_a = { loc_a_no_gl.i+gl , loc_a_no_gl.j+gl , loc_a_no_gl.k+gl };
        const size_t cell_a = grid_ijk_to_index( dims, loc_a );
        const unsigned int n = cells[cell_a].size();
        ONIKA_CU_BLOCK_SIMD_FOR(unsigned int , i , 0 , n )
        {
          func( cells[cell_a][onika::soatl::FieldId<field_ids>{}][i] ... );
        }
      }
    }
    while( cell_a_no_gl < ncells_no_gl );
  }

  template<class CellsT, class FuncT, class... field_ids>
  inline void compute_cell_particles_omp_kernel(
    CellsT* cells,
    IJK dims,
    unsigned int gl,
    const FuncT& func,
    FieldSet<field_ids...> )
  {
    const IJK block_dims = { dims.i-2*gl , dims.j-2*gl , dims.k-2*gl };

#   ifdef XSTAMP_OMP_NUM_THREADS_WORKAROUND
    omp_set_num_threads( omp_get_max_threads() );
#   endif

#   pragma omp parallel
    {
      GRID_OMP_FOR_BEGIN(block_dims,_,block_cell_a_loc, schedule(dynamic) )
      {
        const IJK cell_a_loc = block_cell_a_loc + gl;
        const size_t cell_a = grid_ijk_to_index( dims , cell_a_loc );
        const size_t n = cells[cell_a].size();
#       pragma omp simd
        for(size_t i=0;i<n;i++)
        {
          func( cells[cell_a][onika::soatl::FieldId<field_ids>{}][i] ... );
        }
      }
      GRID_OMP_FOR_END
    }

  }
*/

  template<class GridT, class FuncT, class FieldSetT>
  static inline void compute_cell_particles(
    GridT& grid,
    bool enable_ghosts,
    const FuncT& func,
    FieldSetT cpfields ,
    onika::parallel::GPUKernelExecutionContext * exec_ctx = nullptr,
    bool async = false
    )
  {
    const IJK dims = grid.dimension();
    int gl = grid.ghost_layers();
    if( enable_ghosts ) { gl = 0; }
    const IJK block_dims = dims - (2*gl);
    const size_t N = block_dims.i * block_dims.j * block_dims.k;
    auto cells = grid.cells();

    bool enable_gpu = false;
    if constexpr ( ComputeCellParticlesTraits<FuncT>::CudaCompatible )
    {
      if(exec_ctx != nullptr) 
      {
        if( exec_ctx->has_gpu_context() )
        {
          if( exec_ctx->m_cuda_ctx->has_devices() ) grid.check_cells_are_gpu_addressable();
        }
      }
      enable_gpu = true;
    }

    onika::parallel::block_parallel_for( N, ComputeCellParticlesFunctor<decltype(cells),FuncT,FieldSetT>{cells,dims,gl,func} , exec_ctx , enable_gpu , async );
  }

}

