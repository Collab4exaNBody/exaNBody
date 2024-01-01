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

    ParallelExecutionContext::reset()
    {
      // reset to default state
      m_cuda_ctx = nullptr;
      m_default_stream = nullptr;
      m_omp_num_tasks = 0;
      m_next = nullptr;
      m_execution_end_callback = ParallelExecutionCallback{};
      m_finalize = ParallelExecutionFinalize{};
      m_return_data_input = nullptr;
      m_return_data_output = nullptr;
      m_return_data_size = 0;
      m_execution_target = EXECUTION_TARGET_OPENMP;
      m_block_size = ONIKA_CU_MAX_THREADS_PER_BLOCK;
      m_grid_size = 0; // =0 means that grid size will adapt to number of tasks and workstealing is deactivated. >0 means fixed grid size with workstealing based load balancing
      m_parallel_space = ParallelExecutionSpace{};
      m_reset_counters = false;
      m_total_cpu_execution_time = 0.0;
      m_total_gpu_execution_time = 0.0;
    }

    ParallelExecutionContext::~ParallelExecutionContext()
    {
      reset();
      if( m_start_evt != nullptr ) { ONIKA_CU_DESTROY_EVENT( m_start_evt ); }
      if( m_stop_evt != nullptr ) { ONIKA_CU_DESTROY_EVENT( m_stop_evt ); }      
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

