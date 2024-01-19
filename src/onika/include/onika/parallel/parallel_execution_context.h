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
#pragma once

#include <onika/cuda/device_storage.h>
#include <onika/cuda/cuda_context.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_error.h>

#include <mutex>
#include <condition_variable>

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
      void gpu_kernel_start();
      void gpu_kernel_end();
      void omp_kernel_start();
      void omp_kernel_end();

      cudaStream_t gpu_stream() const;
      onika::cuda::CudaContext* gpu_context() const;
      
      void gpuSynchronizeStream();
      bool queryStatus();
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
      // if m_omp_num_tasks > 0, assume we're in a parallel region running on a single thread (parallel->single/master->taskgroup), thus uses taskloop construcut underneath
      unsigned int m_omp_num_tasks = 0;
      
      std::mutex m_kernel_count_mutex;
      std::condition_variable m_kernel_count_condition;
      uint64_t m_omp_kernel_exec_count = 0;
      uint64_t m_gpu_kernel_exec_count = 0; // number of currently executing GPU kernels

      cudaStream_t m_cuda_stream;
      cudaEvent_t m_start_evt = nullptr;
      cudaEvent_t m_stop_evt = nullptr;
      
      double m_total_async_cpu_execution_time = 0.0;

      double m_total_gpu_execution_time = 0.0;
      
      
      // ============ global configuration variables ===============
      static int s_parallel_task_core_mult;
      static int s_parallel_task_core_add;
      static int s_gpu_sm_mult; // if -1, s_parallel_task_core_mult is used
      static int s_gpu_sm_add;  // if -1, s_parallel_task_core_add is used instead
      static int s_gpu_block_size;
      
      static inline int parallel_task_core_mult() { return s_parallel_task_core_mult; }
      static inline int parallel_task_core_add() { return s_parallel_task_core_add; }
      static inline int gpu_sm_mult() { return ( s_gpu_sm_mult >= 0 ) ? s_gpu_sm_mult : parallel_task_core_mult() ; }
      static inline int gpu_sm_add() { return ( s_gpu_sm_add >= 0 ) ? s_gpu_sm_add : parallel_task_core_add() ; }
      static inline int gpu_block_size() { return  s_gpu_block_size; }
    };

  }

}

