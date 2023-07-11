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

    struct GPUKernelExecutionContext;

    struct GPUStreamCallback
    {
      void(*m_user_callback)(GPUKernelExecutionContext*,void*) = nullptr;
      void *m_user_data = nullptr;
      GPUKernelExecutionContext * m_exec_ctx = nullptr;
      cudaStream_t m_cu_stream; // stream that triggered the callback
    };

    struct GPUKernelExecutionContext
    {
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
      
      inline bool has_gpu_context() const { return m_cuda_ctx != nullptr; }
      
      inline void check_initialize()
      {
        if( m_cuda_scratch == nullptr && m_cuda_ctx != nullptr )
        {
          m_cuda_scratch = onika::cuda::CudaDeviceStorage<GPUKernelExecutionScratch>::New( m_cuda_ctx->m_devices[0]);
        }
        if( m_cuda_ctx != nullptr )
        {
          m_cuda_stream = m_cuda_ctx->getThreadStream(m_streamIndex);
          if( m_start_evt == nullptr )
          {
            checkCudaErrors( ONIKA_CU_CREATE_EVENT( m_start_evt ) );
          }
          if( m_stop_evt == nullptr )
          {
            checkCudaErrors( ONIKA_CU_CREATE_EVENT( m_stop_evt ) );
          }
        }
      }

      inline ~GPUKernelExecutionContext()
      {
        if( m_start_evt != nullptr )
        {
          checkCudaErrors( ONIKA_CU_DESTROY_EVENT( m_start_evt ) );
          m_start_evt = nullptr;
        }
        if( m_stop_evt != nullptr )
        {
          checkCudaErrors( ONIKA_CU_DESTROY_EVENT( m_stop_evt ) );
          m_stop_evt = nullptr;
        }
      }

      inline void reset_counters()
      {
        if( m_cuda_ctx != nullptr )
        {
          checkCudaErrors( ONIKA_CU_MEMSET( m_cuda_scratch.get(), 0, sizeof(GPUKernelExecutionScratch), m_cuda_stream ) );
        }
      }
      
      template<class T>
      inline void set_return_data( const T* init_value )
      {
        static_assert( sizeof(T) <= GPUKernelExecutionScratch::MAX_RETURN_SIZE , "return type size too large" );
        checkCudaErrors( ONIKA_CU_MEMCPY( m_cuda_scratch->return_data, init_value , sizeof(T) , m_cuda_stream ) );
      }

      template<class T>
      inline void retrieve_return_data( T* result )
      {
        static_assert( sizeof(T) <= GPUKernelExecutionScratch::MAX_RETURN_SIZE , "return type size too large" );
        checkCudaErrors( ONIKA_CU_MEMCPY( result , m_cuda_scratch->return_data , sizeof(T) , m_cuda_stream ) );
      }

      inline void record_start_event()
      {
        if( m_gpu_kernel_exec_count == 0 )
        {
          checkCudaErrors( ONIKA_CU_STREAM_EVENT( m_start_evt, m_cuda_stream ) );
        }
        ++ m_gpu_kernel_exec_count;
      }
      
      inline void wait()
      {
        // wait for any in flight async taskloop previously launched and associated with this execution context
        [[maybe_unused]] auto & depvar = *this;
#       pragma omp task depend(in:depvar) if(0) // blocks until dependency is satisfied
        {}

        if( m_gpu_kernel_exec_count > 0 )
        {
          checkCudaErrors( ONIKA_CU_STREAM_EVENT( m_stop_evt, m_cuda_stream ) );
          checkCudaErrors( ONIKA_CU_STREAM_SYNCHRONIZE( m_cuda_stream ) );
          float time_ms = 0.0f;      
          checkCudaErrors( ONIKA_CU_EVENT_ELAPSED(time_ms,m_start_evt,m_stop_evt) );
          m_total_gpu_execution_time += time_ms;      
          -- m_gpu_kernel_exec_count;
          if( m_gpu_kernel_exec_count > 0 )
          {
            // re-insert a timer for next executing kernel
            checkCudaErrors( ONIKA_CU_STREAM_EVENT( m_start_evt, m_cuda_stream ) );
          }
        }
                
      }

      inline double collect_gpu_execution_time()
      {
        double exec_time = 0.0;
        if( m_total_gpu_execution_time > 0.0 )
        {
          exec_time = m_total_gpu_execution_time;
          m_total_gpu_execution_time = 0.0;
        }
        return exec_time;
      }

      double collect_async_cpu_execution_time()
      {
        double exec_time = 0.0;
        if( m_total_async_cpu_execution_time > 0.0 )
        {
          exec_time = m_total_async_cpu_execution_time;
          m_total_async_cpu_execution_time = 0.0;
        }
        return exec_time;
      }

      static inline void execution_end_callback( cudaStream_t stream,  cudaError_t status, void*  userData)
      {
        //std::cout << "execution_end_callback , userData="<<userData << std::endl;
        checkCudaErrors( status );
        GPUStreamCallback* cb = (GPUStreamCallback*) userData;
        assert( cb != nullptr );
        assert( cb->m_exec_ctx != nullptr );
        cb->m_cu_stream = stream;
        ( * cb->m_user_callback ) ( cb->m_exec_ctx , cb->m_user_data );
      }

    };

  }

}

