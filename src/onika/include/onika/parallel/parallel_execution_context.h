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
#include <onika/memory/allocator.h>

#include <mutex>
#include <condition_variable>

namespace onika
{

  namespace parallel
  {

    enum OMPScheduling
    {
      OMP_SCHED_DYNAMIC ,
      OMP_SCHED_GUIDED ,
      OMP_SCHED_STATIC
    };

    struct HostKernelExecutionScratch
    {
      static constexpr size_t SCRATCH_BUFFER_SIZE = 1024; // total device side temporary buffer
      static constexpr size_t MAX_FUNCTOR_SIZE = SCRATCH_BUFFER_SIZE;
      static constexpr size_t MAX_FUNCTOR_ALIGNMENT = onika::memory::DEFAULT_ALIGNMENT;
      alignas(MAX_FUNCTOR_ALIGNMENT) char functor_data[MAX_FUNCTOR_SIZE];
    };

    struct GPUKernelExecutionScratch
    {
      static constexpr size_t SCRATCH_BUFFER_SIZE = 1024; // total device side temporary buffer
      static constexpr size_t MAX_COUNTERS = 8; // only one is used so far, for dynamic attribution of cell indices
      static constexpr size_t MAX_RETURN_SIZE = SCRATCH_BUFFER_SIZE - MAX_COUNTERS * sizeof(unsigned long long);

      unsigned long long int counters[MAX_COUNTERS];
      char return_data[MAX_RETURN_SIZE];
    };
    
    static_assert( sizeof(GPUKernelExecutionScratch) == GPUKernelExecutionScratch::SCRATCH_BUFFER_SIZE );

    struct ParallelExecutionContext;

    struct ParallelExecutionCallback
    {
      void(*m_func)(void*) = nullptr;
      void *m_data = nullptr;
    };

    struct ParallelExecutionFinalize
    {
      void(*m_func)(ParallelExecutionContext*,void*) = nullptr;
      void *m_data = nullptr;
    };

    // abstract parallel space indices
    struct ParallelExecutionSpace
    {
      uint64_t m_start = 0;
      uint64_t m_end = 0;
      uint64_t * __restrict__ m_idx = nullptr;
    };

    struct ParallelExecutionStream;

    struct ParallelExecutionContext
    {
      enum ExecutionTarget
      {
        EXECUTION_TARGET_OPENMP ,
        EXECUTION_TARGET_CUDA
      };
    
      // GPU device context, null if non device available for parallel execution
      onika::cuda::CudaContext* m_cuda_ctx = nullptr;

      // default stream to use for immediate execution if parallel operation is not
      // queued in any stream or graph queue.
      ParallelExecutionStream* m_default_stream = nullptr;

      // desired number of OpenMP tasks.
      // m_omp_num_tasks == 0 means no task (opens and then close its own parallel region).
      // if m_omp_num_tasks > 0, assume we're in a parallel region running on a single thread (parallel->single/master->taskgroup),
      // thus uses taskloop construcut underneath
      unsigned int m_omp_num_tasks = 0;
      
      // allows chaining, for stream queues
      ParallelExecutionContext* m_next = nullptr;
      
      // keep track of creation site
      const char* m_tag = nullptr;
      
      // device side scratch memory for counters, return_data and functor_data
      onika::cuda::CudaDeviceStorage<GPUKernelExecutionScratch> m_cuda_scratch;
      HostKernelExecutionScratch m_host_scratch;

      // additional information about what to do before/after kernel execution
      ParallelExecutionCallback m_execution_end_callback = {};
      ParallelExecutionFinalize m_finalize = {};
      const void * m_return_data_input = nullptr;
      void * m_return_data_output = nullptr;
      unsigned int m_return_data_size = 0;
      ExecutionTarget m_execution_target = EXECUTION_TARGET_OPENMP;
      unsigned int m_block_size = ONIKA_CU_MAX_THREADS_PER_BLOCK;
      unsigned int m_grid_size = 0; // =0 means that grid size will adapt to number of tasks and workstealing is deactivated. >0 means fixed grid size with workstealing based load balancing
      OMPScheduling m_omp_sched = OMP_SCHED_DYNAMIC;
      ParallelExecutionSpace m_parallel_space = {};
      bool m_reset_counters = false;

      // executuion profiling 
      onikaEvent_t m_start_evt = nullptr;
      onikaEvent_t m_stop_evt = nullptr;
      double m_total_cpu_execution_time = 0.0;
      double m_total_gpu_execution_time = 0.0;

      void initialize_stream_events();
      void reset();
      ~ParallelExecutionContext();
      bool has_gpu_context() const;
      void init_device_scratch();
      
      // device side return_data ptr
      void* get_device_return_data_ptr();

      // sets the return_data initialization input. pointer must be valid until execution has ended
      void set_return_data_input( const void* ptr, size_t sz );

      // sets the host pointer receiving return_data after execution has completed
      void set_return_data_output( void* ptr, size_t sz );

      // GPU device context, or nullptr if node device available
      onika::cuda::CudaContext* gpu_context() const;
      
      // convivnience templates
      template<class T> inline void set_return_data_input( const T* init_value )
      {
        static_assert( sizeof(T) <= GPUKernelExecutionScratch::MAX_RETURN_SIZE , "return type size too large" );
        set_return_data_input( init_value , sizeof(T) );
      }
      template<class T> inline void set_return_data_output( T* result )
      {
        static_assert( sizeof(T) <= GPUKernelExecutionScratch::MAX_RETURN_SIZE , "return type size too large" );
        set_return_data_output( result , sizeof(T) );
      }
      
      // callback trampoline function
      static void execution_end_callback( onikaStream_t stream,  onikaError_t status, void*  userData );
      
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

    struct ParallelExecutionGraphQueue
    {
      // to be defined ...
    };

  }

}

