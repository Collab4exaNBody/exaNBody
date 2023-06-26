#pragma once

#include <exanb/compute/compute_pair_singlemat_cell.h>
#include <exanb/compute/compute_pair_singlemat_gpu.h>
#include <exanb/compute/compute_pair_traits.h>
#include <exanb/core/profiling_tools.h>
#include <exanb/core/log.h>
#include <exanb/core/gpu_execution_context.h>

#include <onika/task/parallel_task_config.h>
#include <onika/declare_if.h>

namespace exanb
{

  // ==== OpenMP parallel for style impelmentation ====
  // cells are dispatched to threads using a "#pragma omp parallel for" construct
  template<class GridT, class OptionalArgsT, class ComputePairBufferFactoryT, class FuncT, class FieldSetT , class PosFieldsT = DefaultPositionFields ,  class GPUAccountFuncT = ProfilingAccountTimeNullFunc>
  static inline void compute_pair_singlemat(
    GridT& grid,
    double rcut,
    bool enable_ghosts,
    const OptionalArgsT& optional,
    const ComputePairBufferFactoryT& cpbuf_factory,
    const FuncT& func,
    FieldSetT cpfields,
    PosFieldsT posfields = PosFieldsT{},
    GPUKernelExecutionContext * exec_ctx = nullptr,
    GPUAccountFuncT gpu_account_func = {}
    )
  {
    const double rcut2 = rcut * rcut;
    const IJK dims = grid.dimension();
    int gl = grid.ghost_layers();
    if( enable_ghosts ) { gl = 0; }
    const IJK block_dims = dims - (2*gl);

    ComputePairDebugTraits<FuncT>::print_func( func ); // for debugging purposes

    auto cells = grid.cells();
    auto cellprof = grid.cell_profiler();

    if constexpr ( ComputePairTraits<FuncT>::CudaCompatible )
    {
      //static constexpr onika::IntConst<1> const_1{};
      //static constexpr onika::IntConst<2> const_2{};
      static constexpr onika::IntConst<4> const_4{};
      static constexpr onika::IntConst<8> const_8{};
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
        const unsigned int cs = optional.nbh.m_chunk_size;

        grid.check_cells_are_gpu_addressable();
        
//        std::cout << "going GPU ...\n";
        static constexpr bool has_gpu_timer = ! std::is_same_v<GPUAccountFuncT,ProfilingAccountTimeNullFunc> ;
        DECLARE_IF_CONSTEXPR(has_gpu_timer,ProfilingTimer,timer);
        if constexpr ( has_gpu_timer ) profiling_timer_start(timer);
        
        exec_ctx->reset_counters( streamIndex );
        auto * scratch = exec_ctx->m_cuda_scratch.get();

        switch( cs )
        {
          case 4:
            ONIKA_CU_LAUNCH_KERNEL(GridSize,BlockSize,0,custream, compute_pair_gpu_kernel, cells,dims,gl,optional,cpbuf_factory,func,rcut2,scratch,cellprof,const_4,cpfields,posfields);
            break;
          case 8:
            ONIKA_CU_LAUNCH_KERNEL(GridSize,BlockSize,0,custream, compute_pair_gpu_kernel, cells,dims,gl,optional,cpbuf_factory,func,rcut2,scratch,cellprof,const_8,cpfields,posfields);
            break;
          default:
            ONIKA_CU_LAUNCH_KERNEL(GridSize,BlockSize,0,custream, compute_pair_gpu_kernel, cells,dims,gl,optional,cpbuf_factory,func,rcut2,scratch,cellprof,cs,cpfields,posfields);
            //lerr << "compute_pair_singlemat: chunk size "<<cs<<" not supported. Accepted values are 1, 4, 8." << std::endl;
            //std::abort();
            break;
        }
        checkCudaErrors( ONIKA_CU_STREAM_SYNCHRONIZE(custream) );
        
        if constexpr ( has_gpu_timer ) gpu_account_func( profiling_timer_elapsed_restart(timer) );
        
#       ifdef XNB_GPU_BLOCK_OCCUPANCY_PROFILE
        unsigned int busy_blocks = exec_ctx->get_occupancy_stats( streamIndex );
        lout << "GPU Occupancy : "<< busy_blocks << " / " << GridSize <<" busy blocks"<<std::endl;
#       endif
        
        return ;
      }
    }

#   ifdef XSTAMP_OMP_NUM_THREADS_WORKAROUND
    omp_set_num_threads( omp_get_max_threads() );
#   endif

#   pragma omp parallel
    {
      GRID_OMP_FOR_BEGIN(block_dims,_,block_cell_a_loc, schedule(dynamic) )
      {
        IJK cell_a_loc = block_cell_a_loc + gl;
        size_t cell_a = grid_ijk_to_index( dims , cell_a_loc );
        cellprof.start_cell_profiling(cell_a);
        compute_pair_singlemat_cell(cells,dims,cell_a_loc,cell_a,rcut2,cpbuf_factory,optional,func,cpfields,posfields);
        cellprof.end_cell_profiling(cell_a);
      }
      GRID_OMP_FOR_END
    }
  }

}

