#pragma once

#include <onika/task/parallel_task_config.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_error.h>
#include <onika/cuda/device_storage.h>
#include <onika/soatl/field_id.h>
#include <exanb/field_sets.h>

#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/profiling_tools.h>
#include <exanb/core/gpu_execution_context.h>

#ifdef XSTAMP_OMP_NUM_THREADS_WORKAROUND
#include <omp.h>
#endif

namespace exanb
{

  // this template is here to know if compute buffer must be built or computation must be ran on the fly
  template<class FuncT> struct ComputeCellParticlesTraits
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = false;
  };

  template<class CellsT, class FuncT, class... field_ids>
  ONIKA_DEVICE_KERNEL_FUNC
  ONIKA_DEVICE_KERNEL_BOUNDS(ONIKA_CU_MAX_THREADS_PER_BLOCK,ONIKA_CU_MIN_BLOCKS_PER_SM)
  void compute_cell_particles_gpu_kernel(
    CellsT* cells,
    IJK dims,
    unsigned int gl,
    const FuncT func,
    GPUKernelExecutionScratch* scratch,
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

  template<class GridT, class FuncT, class FieldSetT, class GPUAccountFuncT = ProfilingAccountTimeNullFunc>
  static inline void compute_cell_particles(
    GridT& grid,
    bool enable_ghosts,
    const FuncT& func,
    FieldSetT cpfields ,
    GPUKernelExecutionContext * exec_ctx = nullptr,
    GPUAccountFuncT gpu_account_func = {}
    )
  {
    const IJK dims = grid.dimension();
    const int gl = enable_ghosts ? 0 : grid.ghost_layers() ;
    auto cells = grid.cells();

    if constexpr ( ComputeCellParticlesTraits<FuncT>::CudaCompatible )
    {
      bool allow_cuda_exec = ( exec_ctx != nullptr );
      if( allow_cuda_exec ) allow_cuda_exec = ( exec_ctx->m_cuda_ctx != nullptr );
      if( allow_cuda_exec ) allow_cuda_exec = exec_ctx->m_cuda_ctx->has_devices();
      if( allow_cuda_exec )
      {
        exec_ctx->check_initialize();
        const unsigned int BlockSize = std::min( static_cast<size_t>(ONIKA_CU_MAX_THREADS_PER_BLOCK) , static_cast<size_t>(onika::task::ParallelTaskConfig::gpu_block_size()) );
        const unsigned int GridSize = exec_ctx->m_cuda_ctx->m_devices[0].m_deviceProp.multiProcessorCount * onika::task::ParallelTaskConfig::gpu_sm_mult()
                                    + onika::task::ParallelTaskConfig::gpu_sm_add();
        const int streamIndex = 0;
        auto custream = exec_ctx->m_cuda_ctx->m_threadStream[streamIndex];

        grid.check_cells_are_gpu_addressable();

        ProfilingTimer timer;
        if constexpr ( ! std::is_same_v<GPUAccountFuncT,ProfilingAccountTimeNullFunc> ) profiling_timer_start(timer);

        exec_ctx->reset_counters( streamIndex );
        auto * scratch = exec_ctx->m_cuda_scratch.get();

        ONIKA_CU_LAUNCH_KERNEL(GridSize,BlockSize,0,custream, compute_cell_particles_gpu_kernel, cells, dims, gl, func, scratch, cpfields );
        checkCudaErrors( ONIKA_CU_STREAM_SYNCHRONIZE(custream) );
        if constexpr ( ! std::is_same_v<GPUAccountFuncT,ProfilingAccountTimeNullFunc> ) gpu_account_func( profiling_timer_elapsed_restart(timer) );
        return;
      }
    }

    compute_cell_particles_omp_kernel(cells, dims, gl, func, cpfields);
  }

}

