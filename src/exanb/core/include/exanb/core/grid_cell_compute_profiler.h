#pragma once

#include <onika/memory/allocator.h>
#include <onika/cuda/cuda.h>

#define XNB_GRID_CELL_COMPUTE_PROFILING 1

namespace exanb
{

  // per cell compute time profiling
  struct GridCellComputeProfiling
  {
    onika::cuda::onika_cu_clock_t m_timer;
    double m_time;
  };
  using GridComputeProfiling = onika::memory::CudaMMVector<GridCellComputeProfiling>;

  struct GridCellComputeProfiler
  {
#   ifdef XNB_GRID_CELL_COMPUTE_PROFILING
    GridCellComputeProfiling * __restrict__ m_grid_compute_profiling_ptr = nullptr;
    ONIKA_HOST_DEVICE_FUNC inline void start_cell_profiling( size_t cell_idx )
    {
      if( m_grid_compute_profiling_ptr != nullptr )
      {
        ONIKA_CU_BLOCK_SYNC();
        if( ONIKA_CU_THREAD_IDX == 0 )
        {
          m_grid_compute_profiling_ptr[cell_idx].m_timer = ONIKA_CU_CLOCK();
        }
      }
    }
    ONIKA_HOST_DEVICE_FUNC inline void end_cell_profiling( size_t cell_idx )
    {
      if( m_grid_compute_profiling_ptr != nullptr )
      {
        ONIKA_CU_BLOCK_SYNC();
        if( ONIKA_CU_THREAD_IDX == 0 )
        {
          auto Tend = ONIKA_CU_CLOCK();
          m_grid_compute_profiling_ptr[cell_idx].m_time += ONIKA_CU_CLOCK_ELAPSED( m_grid_compute_profiling_ptr[cell_idx].m_timer , Tend );
        }
      }
    }
    inline double get_cell_time(size_t cell_idx)
    {
      if( m_grid_compute_profiling_ptr != nullptr ) return m_grid_compute_profiling_ptr[cell_idx].m_time;
      else return 1.0;
    }
#   else
    ONIKA_HOST_DEVICE_FUNC inline GridComputeProfiler(GridCellComputeProfiling *) {}
    ONIKA_HOST_DEVICE_FUNC static inline void start_cell_profiling( size_t ) {}
    ONIKA_HOST_DEVICE_FUNC static inline void end_cell_profiling( size_t ) {}
    ONIKA_HOST_DEVICE_FUNC static inline double get_cell_time( size_t ) { return 1.0; }
#   endif
  };

} // end of namespace exanb

