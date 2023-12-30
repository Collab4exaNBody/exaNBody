#include <onika/cuda/device_storage.h>
#include <onika/cuda/cuda_context.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_error.h>
#include <onika/parallel/parallel_execution_context.h>

namespace onika
{

  namespace parallel
  {

    // global configuration variables
    int ParallelExecutionContext::s_parallel_task_core_mult = 4;
    int ParallelExecutionContext::s_parallel_task_core_add = 0;
    int ParallelExecutionContext::s_gpu_sm_mult = -1; // if -1, s_parallel_task_core_mult is used
    int ParallelExecutionContext::s_gpu_sm_add = -1;  // if -1, s_parallel_task_core_add is used instead
    int ParallelExecutionContext::s_gpu_block_size = 128;

    ParallelExecutionContext::~ParallelExecutionContext()
    {
/*
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
*/
    }
     
    bool ParallelExecutionContext::has_gpu_context() const
    {
      return m_cuda_ctx != nullptr;
    }
    
    onika::cuda::CudaContext* ParallelExecutionContext::gpu_context() const
    {
      return m_cuda_ctx;
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
        m_cuda_scratch = onika::cuda::CudaDeviceStorage<GPUKernelExecutionScratch>::New( m_cuda_ctx->m_devices[0] );
      }
    }
    
    void ParallelExecutionContext::check_initialize()
    {
      if( m_cuda_ctx != nullptr )
      {
        init_device_scratch();
        /*
        m_cuda_stream = m_cuda_ctx->getThreadStream(m_streamIndex);
        if( m_start_evt == nullptr )
        {
          checkCudaErrors( ONIKA_CU_CREATE_EVENT( m_start_evt ) );
        }
        if( m_stop_evt == nullptr )
        {
          checkCudaErrors( ONIKA_CU_CREATE_EVENT( m_stop_evt ) );
        }
        */
      }
    }

    void* ParallelExecutionContext::get_device_return_data_ptr()
    {
      init_device_scratch();
      return m_cuda_scratch->return_data;
    }

    void ParallelExecutionContext::set_reset_counters(bool rst)
    {
      m_reset_counters = rst;
      /*
      if( m_cuda_ctx != nullptr )
      {
        checkCudaErrors( ONIKA_CU_MEMSET( m_cuda_scratch.get(), 0, sizeof(GPUKernelExecutionScratch), m_cuda_stream ) );
      }
      */
    }

    void ParallelExecutionContext::set_return_data_input( const void* ptr, size_t sz )
    {
      if( sz > GPUKernelExecutionScratch::MAX_RETURN_SIZE )
      {
        std::cerr << "Fatal error: return data size too large" << std::endl;
        std::abort();
      }
      m_return_data_input = ptr;
      m_return_data_size = sz;
      // checkCudaErrors( ONIKA_CU_MEMCPY( m_cuda_scratch->return_data, ptr , sz , m_cuda_stream ) );
    }
    
    void ParallelExecutionContext::set_return_data_output( void* ptr, size_t sz )
    {      
      if( sz > GPUKernelExecutionScratch::MAX_RETURN_SIZE )
      {
        std::cerr << "Fatal error: return data size too large" << std::endl;
        std::abort();
      }
      if( m_return_data_input != nullptr && m_return_data_size != 0 && m_return_data_size != sz )
      {
        std::cerr << "Fatal error: return data size mismatch" << std::endl;
        std::abort();
      }
      // checkCudaErrors( ONIKA_CU_MEMCPY( ptr , m_cuda_scratch->return_data , sz , m_cuda_stream ) );
    }

    void ParallelExecutionContext::execution_end_callback( cudaStream_t stream,  cudaError_t status, void*  userData )
    {
      const ParallelExecutionContext * pec = reinterpret_cast<const ParallelExecutionContext *>( userData );
      if( pec != nullptr && pec->m_execution_end_callback.m_func != nullptr )
      {
        ( * pec->m_execution_end_callback.m_func ) ( pec->m_execution_end_callback.m_data );
      }
    }

  }

}

