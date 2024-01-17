#pragma once

#include <cuda_runtime.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_error.h>

namespace onika
{

  namespace cuda
  {

    static inline constexpr unsigned int CUDA_MAX_SM_COUNT = 1024;

    struct SMOverlapCounters
    {
      int sm_overlap_counter[CUDA_MAX_SM_COUNT];
      int sm_max_overlapped_blocks[CUDA_MAX_SM_COUNT];
      volatile int execute_or_profile;
    };

    template<class FuncT, class ... Args >
    ONIKA_DEVICE_KERNEL_FUNC void max_overlapped_blocks_per_sm_kernel( SMOverlapCounters* d_counters, FuncT func , Args ... args )
    {
      const unsigned int smid = onika::cuda::get_smid();

      if( d_counters->execute_or_profile == 0 && ONIKA_CU_THREAD_IDX == 0 )
      {
        const int overlap_count = 1 + ONIKA_CU_ATOMIC_ADD( d_counters->sm_overlap_counter[smid] , 1 );
        ONIKA_CU_ATOMIC_MAX( d_counters->sm_max_overlapped_blocks[smid] , overlap_count );
      }
      
      if( d_counters->execute_or_profile == 1 ) // never executed, but the compiler cannot know that
      {
        func( args ... );
      }
      
      if( d_counters->execute_or_profile == 0 && ONIKA_CU_THREAD_IDX == 0 )
      {
       ONIKA_CU_ATOMIC_ADD( d_counters->sm_overlap_counter[smid] , -1 );
      } 
    }

    template<class FuncT, class ... Args >
    inline int max_overlapped_blocks_per_sm(unsigned int BlockSize, FuncT func , Args ... args)
    {
      SMOverlapCounters * d_counters = nullptr;
      ONIKA_CU_CHECK_ERRORS( cudaMalloc(&d_counters , sizeof(SMOverlapCounters) ) );
      ONIKA_CU_CHECK_ERRORS( cudaMemset(d_counters, 0, sizeof(SMOverlapCounters) ) );

      max_overlapped_blocks_per_sm_kernel<<< CUDA_MAX_SM_COUNT*8 , BlockSize >>>( d_counters, func, args... );
      ONIKA_CU_CHECK_ERRORS( cudaDeviceSynchronize() );
      
      SMOverlapCounters h_counters;
      ONIKA_CU_CHECK_ERRORS( cudaMemcpy(&h_counters, d_counters, sizeof(SMOverlapCounters), cudaMemcpyDeviceToHost) );
      ONIKA_CU_CHECK_ERRORS( cudaFree(d_counters) );

      int max_overlap = 0;
      for(unsigned int i=0;i<CUDA_MAX_SM_COUNT;i++)
      {
        max_overlap = std::max( max_overlap , h_counters.sm_max_overlapped_blocks[i] );
      }
      return max_overlap;
    }

  }
}

