#pragma once

#include <onika/cuda/cuda_context.h>
#include <onika/parallel/parallel_execution_context.h>
#include <mutex>
#include <atomic>

namespace onika
{

  namespace parallel
  {

    // temporarily holds ParallelExecutionContext instance until it is either queued in a stream or graph execution flow,
    // or destroyed, in which case it inserts instance onto the default stream queue
    struct ParallelExecutionWrapper
    {
      ParallelExecutionContext* m_pec = nullptr;
      ~ParallelExecutionWrapper();
    };

    // allows asynchronous sequential execution of parallel executions queued in the same stream
    // multiple kernel execution concurrency can be handled manually using several streams (same as Cuda stream)
    struct ParallelExecutionStream
    {
      // GPU device context, null if non device available for parallel execution
      // any parallel executiion enqueued to this stream must have either a null CudaContext or the same context as the stream
      onika::cuda::CudaContext* m_cuda_ctx = nullptr; 
      cudaStream_t m_cu_stream = 0;
      uint32_t m_stream_id = 0;
      std::atomic<uint32_t> m_omp_execution_count = 0;
      std::mutex m_mutex;
    };

    struct ParallelExecutionStreamQueue
    {
      ParallelExecutionStream* m_stream = nullptr;
      ParallelExecutionContext* m_queue = nullptr;

      ParallelExecutionStreamQueue() = default;
      ParallelExecutionStreamQueue(ParallelExecutionStream* st);
      ParallelExecutionStreamQueue(ParallelExecutionStreamQueue && o);
      ParallelExecutionStreamQueue& operator = (ParallelExecutionStreamQueue && o);
      ~ParallelExecutionStreamQueue();
      void wait();
      bool query_status();
      bool empty() const;
      bool has_stream() const;
    };
        
    // real implementation of how a parallel operation is pushed onto a stream queue
    ParallelExecutionStreamQueue operator << ( ParallelExecutionStreamQueue && pes , ParallelExecutionWrapper && pew );
  }

}

