#pragma once

#include <onika/cuda/device_storage.h>
#include <onika/cuda/cuda_context.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_error.h>

//#define XNB_GPU_BLOCK_OCCUPANCY_PROFILE 1

namespace exanb
{

  struct GPUKernelExecutionScratch
  {
    static constexpr size_t MAX_COUNTERS = 4; // only one is used so far, for dynamic attribution of cell indices
    static constexpr size_t RESERVED_COUNTERS = 1;
    static constexpr size_t CELL_COUNTER_IDX = 0;
    //static constexpr size_t MAX_RETURN_SIZE = 64;
    static constexpr size_t MAX_RETURN_SIZE = 64*9;
    static constexpr size_t FREE_COUNTERS = MAX_COUNTERS - RESERVED_COUNTERS;
    //static constexpr size_t FREE_BYTES = FREE_COUNTERS * sizeof( unsigned long long int );
    static constexpr size_t FREE_BYTES = FREE_COUNTERS * sizeof( double );
//    static constexpr size_t FREE_BYTES = FREE_COUNTERS * (64*9);

    unsigned char return_data[MAX_RETURN_SIZE];
    unsigned long long int counters[MAX_COUNTERS];

#   ifdef XNB_GPU_BLOCK_OCCUPANCY_PROFILE
    static constexpr size_t MAX_GPU_BLOCKS = 2048;
    unsigned int block_occupancy[MAX_GPU_BLOCKS];
#   endif

    ONIKA_HOST_DEVICE_FUNC inline unsigned long long int * cell_idx_counter() { return counters+CELL_COUNTER_IDX; }
    ONIKA_HOST_DEVICE_FUNC inline unsigned long long int * free_counters() { return counters+RESERVED_COUNTERS; }
  };

  struct GPUKernelExecutionContext
  {
    onika::cuda::CudaContext* m_cuda_ctx = nullptr;
    onika::cuda::CudaDeviceStorage<GPUKernelExecutionScratch> m_cuda_scratch;
    
    inline void check_initialize()
    {
      if( m_cuda_scratch == nullptr && m_cuda_ctx != nullptr )
      {
        m_cuda_scratch = onika::cuda::CudaDeviceStorage<GPUKernelExecutionScratch>::New( m_cuda_ctx->m_devices[0]);
      }
    }
    
    inline void reset_counters(int streamIndex)
    {
      if( m_cuda_ctx != nullptr )
      {
        checkCudaErrors( ONIKA_CU_MEMSET( m_cuda_scratch.get(), 0, sizeof(GPUKernelExecutionScratch), m_cuda_ctx->m_threadStream[streamIndex] ) );
      }
    }
    
    template<class T>
    inline void set_return_data( const T* init_value, int streamIndex )
    {
      static_assert( sizeof(T) <= GPUKernelExecutionScratch::MAX_RETURN_SIZE , "return type size too large" );
      checkCudaErrors( ONIKA_CU_MEMCPY( m_cuda_scratch->return_data, init_value , sizeof(T) , m_cuda_ctx->m_threadStream[streamIndex] ) );
    }

    template<class T>
    inline void retrieve_return_data( T* result, int streamIndex )
    {
      static_assert( sizeof(T) <= GPUKernelExecutionScratch::MAX_RETURN_SIZE , "return type size too large" );
      checkCudaErrors( ONIKA_CU_MEMCPY( result , m_cuda_scratch->return_data , sizeof(T) , m_cuda_ctx->m_threadStream[streamIndex] ) );
    }

    inline unsigned int get_occupancy_stats(int streamIndex)
    {
      unsigned int n_busy = 0;
#     ifdef XNB_GPU_BLOCK_OCCUPANCY_PROFILE
      unsigned int tmp[GPUKernelExecutionScratch::MAX_GPU_BLOCKS];
      checkCudaErrors( ONIKA_CU_MEMCPY( tmp , m_cuda_scratch->block_occupancy , sizeof(unsigned int)*GPUKernelExecutionScratch::MAX_GPU_BLOCKS , m_cuda_ctx->m_threadStream[streamIndex] ) );
      checkCudaErrors( ONIKA_CU_STREAM_SYNCHRONIZE(m_cuda_ctx->m_threadStream[streamIndex]) );
      for(unsigned int i=0;i<GPUKernelExecutionScratch::MAX_GPU_BLOCKS;i++)
      {
        if( tmp[i] > 0 ) ++ n_busy;
      }
#     endif
      return n_busy;
    }

  };

}

