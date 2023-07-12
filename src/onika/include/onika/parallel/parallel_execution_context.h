#pragma once

#include <onika/cuda/device_storage.h>
#include <onika/cuda/cuda_context.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_error.h>

namespace onika
{

  namespace parallel
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

      ONIKA_HOST_DEVICE_FUNC inline unsigned long long int * cell_idx_counter() { return counters+CELL_COUNTER_IDX; }
      ONIKA_HOST_DEVICE_FUNC inline unsigned long long int * free_counters() { return counters+RESERVED_COUNTERS; }
    };

    struct ParallelExecutionContext;

    struct ParallelExecutionStreamCallback
    {
      void(*m_user_callback)(ParallelExecutionContext*,void*) = nullptr;
      void *m_user_data = nullptr;
      ParallelExecutionContext * m_exec_ctx = nullptr;
      cudaStream_t m_cu_stream; // stream that triggered the callback
    };

    struct ParallelExecutionContext
    {      
      ~ParallelExecutionContext();
      bool has_gpu_context() const;
      void init_device_scratch();
      void check_initialize();
      void reset_counters();
      void set_return_data( const void* ptr, size_t sz );
      void* get_device_return_data_ptr();
      void retrieve_return_data( void* ptr, size_t sz );
      void record_start_event();
      void gpuSynchronizeStream();
      void wait();
      double collect_gpu_execution_time();
      double collect_async_cpu_execution_time();
      void register_stream_callback( ParallelExecutionStreamCallback* user_cb );
      static void execution_end_callback( cudaStream_t stream,  cudaError_t status, void*  userData);

      // convivnience templates
      template<class T> inline void set_return_data( const T* init_value )
      {
        static_assert( sizeof(T) <= GPUKernelExecutionScratch::MAX_RETURN_SIZE , "return type size too large" );
        set_return_data( init_value , sizeof(T) );
      }
      template<class T> inline void retrieve_return_data( T* result )
      {
        static_assert( sizeof(T) <= GPUKernelExecutionScratch::MAX_RETURN_SIZE , "return type size too large" );
        retrieve_return_data( result , sizeof(T) );
      }

    // members
      onika::cuda::CudaContext* m_cuda_ctx = nullptr;
      onika::cuda::CudaDeviceStorage<GPUKernelExecutionScratch> m_cuda_scratch;

      unsigned int m_streamIndex = 0;
      
      // desired number of tasks.
      // m_omp_num_tasks == 0 means no task (opens and then close its own parallel region).
      // if m_omp_num_tasks > 0, assume we're in a parallel region running on a single thread (parallel->single->taskgroup), thus uses taskloop construcut underneath
      unsigned int m_omp_num_tasks = 0;

      cudaStream_t m_cuda_stream;
      cudaEvent_t m_start_evt = nullptr;
      cudaEvent_t m_stop_evt = nullptr;
      
      double m_total_async_cpu_execution_time = 0.0;

      double m_total_gpu_execution_time = 0.0;
      unsigned int m_gpu_kernel_exec_count = 0; // number of currently executing GPU kernels

    };

  }

}

