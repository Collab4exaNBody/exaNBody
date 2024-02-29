/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/
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

    void ParallelExecutionContext::reset()
    {
      // reset to default state
      m_tag = nullptr;
      m_sub_tag = nullptr;
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
      if( m_start_evt != nullptr ) { ONIKA_CU_CHECK_ERRORS( ONIKA_CU_DESTROY_EVENT( m_start_evt ) ); }
      if( m_stop_evt  != nullptr ) { ONIKA_CU_CHECK_ERRORS( ONIKA_CU_DESTROY_EVENT( m_stop_evt  ) ); }
    }
    
    void ParallelExecutionContext::initialize_stream_events()
    {
      if( m_cuda_ctx != nullptr )
      {
        if( m_start_evt == nullptr ) { ONIKA_CU_CHECK_ERRORS( ONIKA_CU_CREATE_EVENT( m_start_evt ) ); }
        if( m_stop_evt  == nullptr ) { ONIKA_CU_CHECK_ERRORS( ONIKA_CU_CREATE_EVENT( m_stop_evt  ) ); }
      }
    }
    
    bool ParallelExecutionContext::has_gpu_context() const
    {
      return m_cuda_ctx != nullptr;
    }
    
    onika::cuda::CudaContext* ParallelExecutionContext::gpu_context() const
    {
      return m_cuda_ctx;
    }

    const char* ParallelExecutionContext::tag() const
    {
      return (m_tag!=nullptr) ? m_tag : "<unknown>" ;
    }
    
    const char* ParallelExecutionContext::sub_tag() const
    {
      return (m_sub_tag!=nullptr) ? m_sub_tag : "" ;
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

    void* ParallelExecutionContext::get_device_return_data_ptr()
    {
      return m_cuda_scratch->return_data;
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
      // ONIKA_CU_CHECK_ERRORS( ONIKA_CU_MEMCPY( m_cuda_scratch->return_data, ptr , sz , m_cuda_stream ) );
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
      m_return_data_output = ptr;
      m_return_data_size = sz;
    }

    void ParallelExecutionContext::execution_end_callback( onikaStream_t stream,  onikaError_t status, void*  userData )
    {
      const ParallelExecutionContext * pec = reinterpret_cast<const ParallelExecutionContext *>( userData );
      if( pec != nullptr && pec->m_execution_end_callback.m_func != nullptr )
      {
        ( * pec->m_execution_end_callback.m_func ) ( pec->m_execution_end_callback.m_data );
      }
    }

  }

}

