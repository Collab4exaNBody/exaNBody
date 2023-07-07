#pragma once

#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_context.h>
#include <onika/cuda/cuda_error.h>
#include <onika/cuda/device_storage.h>

#include <cstdlib>

#include <exanb/compute/compute_pair_singlemat_cell.h>
#include <onika/parallel/parallel_execution_context.h>

namespace exanb
{

  template<class CellsT, class OptionalT, class CPBufFactoryT, class ForceOpT, class ChunkSizeT, class CellProfT, class ComputeFieldsT, class PosFieldsT = DefaultPositionFields>
  ONIKA_DEVICE_KERNEL_FUNC
  ONIKA_DEVICE_KERNEL_BOUNDS(ONIKA_CU_MAX_THREADS_PER_BLOCK,ONIKA_CU_MIN_BLOCKS_PER_SM)
  void compute_pair_gpu_kernel(
    CellsT cells,
    IJK dims,
    unsigned int gl,
    const OptionalT optional,
    const CPBufFactoryT cpbuf_factory,
    const ForceOpT force_op,
    double rcut2,
    onika::parallel::GPUKernelExecutionScratch* scratch,
    CellProfT cellprof,
    ChunkSizeT CS,
    ComputeFieldsT cpfields,
    PosFieldsT posfields)
  {
    // avoid use of compute buffer when possible
    static constexpr onika::BoolConst< ! ComputePairTraits<ForceOpT>::BufferLessCompatible > UseComputeBuffer{}; 
    const IJK dimsNoGL = { dims.i-2*gl , dims.j-2*gl , dims.k-2*gl };
    const uint64_t ncells_no_gl = dimsNoGL.i * dimsNoGL.j * dimsNoGL.k;
    ONIKA_CU_BLOCK_SHARED unsigned int cell_a_no_gl;

#   ifdef XNB_GPU_BLOCK_OCCUPANCY_PROFILE
    ONIKA_CU_BLOCK_SHARED unsigned int n_cells_processed;
    if( ONIKA_CU_THREAD_IDX == 0 ) { n_cells_processed = 0; }
#   endif

    do
    {
      if( ONIKA_CU_THREAD_IDX == 0 )
      {
        cell_a_no_gl = ONIKA_CU_ATOMIC_ADD( scratch->counters[0] , 1u );
#       ifdef XNB_GPU_BLOCK_OCCUPANCY_PROFILE
        if( cell_a_no_gl < ncells_no_gl ) { ++ n_cells_processed; }
#       endif
        //printf("processing cell #%d\n",int(cell_a_no_gl));
      }
      ONIKA_CU_BLOCK_SYNC();
      if( cell_a_no_gl < ncells_no_gl )
      {
        const IJK loc_a_no_gl = grid_index_to_ijk( dimsNoGL, cell_a_no_gl );
        const IJK loc_a = { loc_a_no_gl.i+gl , loc_a_no_gl.j+gl , loc_a_no_gl.k+gl };
        const size_t cell_a = grid_ijk_to_index( dims, loc_a );
        cellprof.start_cell_profiling(cell_a);
        compute_pair_singlemat_cell(cells,dims,loc_a,cell_a,rcut2 ,cpbuf_factory, optional,force_op,
                                    CS, std::integral_constant<bool,false>{}, cpfields , posfields, UseComputeBuffer );
        cellprof.end_cell_profiling(cell_a);
      }
    }
    while( cell_a_no_gl < ncells_no_gl );
    
#   ifdef XNB_GPU_BLOCK_OCCUPANCY_PROFILE
    if( ONIKA_CU_THREAD_IDX == 0 && ONIKA_CU_BLOCK_IDX < onika::parallel::GPUKernelExecutionScratch::MAX_GPU_BLOCKS ) { scratch->block_occupancy[ONIKA_CU_BLOCK_IDX] = n_cells_processed; }
#   endif
  }

}

