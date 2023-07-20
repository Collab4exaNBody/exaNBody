#include <onika/cuda/device_storage.h>
#include <onika/cuda/cuda_context.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_error.h>
#include <onika/parallel/parallel_execution_context.h>

namespace onika
{

  namespace parallel
  {

    ParallelExecutionContext::~ParallelExecutionContext()
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
     
    bool ParallelExecutionContext::has_gpu_context() const
    {
      return m_cuda_ctx != nullptr;
    }
    
    void ParallelExecutionContext::init_device_scratch()
    {
      if( m_cuda_scratch == nullptr )
      {
        if( m_cuda_ctx == nullptr )
        {
          std::cerr << "Fatal error: no Cuda context, cannot initialize device scratch mem" << std::endl;
          std::abort();
        }
        m_cuda_scratch = onika::cuda::CudaDeviceStorage<GPUKernelExecutionScratch>::New( m_cuda_ctx->m_devices[0]);
      }
    }
    
    void ParallelExecutionContext::check_initialize()
    {
      if( m_cuda_ctx != nullptr )
      {
        init_device_scratch();
        
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

    void* ParallelExecutionContext::get_device_return_data_ptr()
    {
      init_device_scratch();
      return m_cuda_scratch->return_data;
    }

    void ParallelExecutionContext::reset_counters()
    {
      if( m_cuda_ctx != nullptr )
      {
        checkCudaErrors( ONIKA_CU_MEMSET( m_cuda_scratch.get(), 0, sizeof(GPUKernelExecutionScratch), m_cuda_stream ) );
      }
    }

    void ParallelExecutionContext::set_return_data( const void* ptr, size_t sz )
    {
      if( sz > GPUKernelExecutionScratch::MAX_RETURN_SIZE )
      {
        std::cerr << "Fatal error: return type size too large" << std::endl;
        std::abort();
      }
      checkCudaErrors( ONIKA_CU_MEMCPY( m_cuda_scratch->return_data, ptr , sz , m_cuda_stream ) );
    }
    
    void ParallelExecutionContext::retrieve_return_data( void* ptr, size_t sz )
    {
      if( sz > GPUKernelExecutionScratch::MAX_RETURN_SIZE )
      {
        std::cerr << "Fatal error: return type size too large" << std::endl;
        std::abort();
      }
      checkCudaErrors( ONIKA_CU_MEMCPY( ptr , m_cuda_scratch->return_data , sz , m_cuda_stream ) );
    }

    // enqueue start event to associated GPU execution stream and increments number of executing GPU kernels
    // maximum number of executing GPU kernels is 1
    void ParallelExecutionContext::gpu_kernel_start()
    {
      std::unique_lock<std::mutex> lk(m_kernel_count_mutex);
      m_kernel_count_condition.wait( lk, [this](){ return m_gpu_kernel_exec_count==0; } );
      assert( m_gpu_kernel_exec_count == 0 );
      checkCudaErrors( ONIKA_CU_STREAM_EVENT( m_start_evt, m_cuda_stream ) );
      ++ m_gpu_kernel_exec_count;
      lk.unlock();
      m_kernel_count_condition.notify_all();
    }

    // enqueue stop event in associated GPU stream for timing purposes
    void ParallelExecutionContext::gpu_kernel_end()
    {
      assert( m_gpu_kernel_exec_count == 1 );
      checkCudaErrors( ONIKA_CU_STREAM_EVENT( m_stop_evt, m_cuda_stream ) );
    }

    void ParallelExecutionContext::omp_kernel_start()
    {
      std::unique_lock<std::mutex> lk(m_kernel_count_mutex);
      ++ m_omp_kernel_exec_count;
      lk.unlock();
      m_kernel_count_condition.notify_all();
    }
    
    void ParallelExecutionContext::omp_kernel_end()
    {
      std::unique_lock<std::mutex> lk(m_kernel_count_mutex);
      -- m_omp_kernel_exec_count;
      lk.unlock();
      m_kernel_count_condition.notify_all();
    }

    void ParallelExecutionContext::gpuSynchronizeStream()
    {
      checkCudaErrors( ONIKA_CU_STREAM_SYNCHRONIZE( m_cuda_stream ) );
    }

    bool ParallelExecutionContext::queryStatus()
    {
      std::unique_lock<std::mutex> lk(m_kernel_count_mutex);
      return ( m_omp_kernel_exec_count + m_gpu_kernel_exec_count ) == 0;
    }
    
    void ParallelExecutionContext::wait()
    {    
      std::unique_lock<std::mutex> lk(m_kernel_count_mutex);
      m_kernel_count_condition.wait( lk , 
        [this]()
        {
          // wait for GPU kernels completion
          if( m_gpu_kernel_exec_count > 0 )
          {
            assert( m_gpu_kernel_exec_count == 1 ); // multiple flying GPU kernels not supported yet
            gpuSynchronizeStream();
            float time_ms = 0.0f;      
            checkCudaErrors( ONIKA_CU_EVENT_ELAPSED(time_ms,m_start_evt,m_stop_evt) );
            m_total_gpu_execution_time += time_ms;
            m_gpu_kernel_exec_count = 0;
          }

          return m_omp_kernel_exec_count+m_gpu_kernel_exec_count == 0;
        });
      
      lk.unlock();
      m_kernel_count_condition.notify_all();
    }
    
    double ParallelExecutionContext::collect_gpu_execution_time()
    {
      double exec_time = 0.0;
      if( m_total_gpu_execution_time > 0.0 )
      {
        exec_time = m_total_gpu_execution_time;
        m_total_gpu_execution_time = 0.0;
      }
      return exec_time;
    }

    double ParallelExecutionContext::collect_async_cpu_execution_time()
    {
      double exec_time = 0.0;
      if( m_total_async_cpu_execution_time > 0.0 )
      {
        exec_time = m_total_async_cpu_execution_time;
        m_total_async_cpu_execution_time = 0.0;
      }
      return exec_time;
    }

    void ParallelExecutionContext::register_stream_callback( ParallelExecutionStreamCallback* user_cb )
    {
      if( user_cb != nullptr )
      {
        user_cb->m_exec_ctx = this;
        checkCudaErrors( cudaStreamAddCallback(m_cuda_stream,onika::parallel::ParallelExecutionContext::execution_end_callback,user_cb,0) );
      }
    }

    void ParallelExecutionContext::execution_end_callback( cudaStream_t stream,  cudaError_t status, void*  userData)
    {
      //std::cout << "execution_end_callback , userData="<<userData << std::endl;
      checkCudaErrors( status );
      ParallelExecutionStreamCallback* cb = (ParallelExecutionStreamCallback*) userData;
      assert( cb != nullptr );
      assert( cb->m_exec_ctx != nullptr );
      cb->m_cu_stream = stream;
      ( * cb->m_user_callback ) ( cb->m_exec_ctx , cb->m_user_data );
    }

  }

}

