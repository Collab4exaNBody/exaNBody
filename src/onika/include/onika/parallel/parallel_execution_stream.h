#pragma once

#include <onika/parallel/parallel_execution_context.h>

namespace onika
{

  namespace parallel
  {

    // allows asynchronous sequential execution of parallel executions queued in the same stream
    // multiple kernel execution concurrency can be handled manually using several streams (same as Cuda stream)
    struct ParallelExecutionStream
    {
//      static constexpr unsigned int MAX_PARALLEL_EXECUTION_STREAMS = 1024;
      unsigned int m_stream_id = 0;
      cudaStream_t m_cu_stream = 0;
    };

    struct ParallelExecutionStreamQueue
    {
      ParallelExecutionStream* m_stream = nullptr;
      ParallelExecutionContext* m_queue = nullptr;

      ParallelExecutionStreamQueue() = default;
      inline ParallelExecutionStreamQueue(ParallelExecutionStream* st, ParallelExecutionCallback cb) : m_stream(st) , m_callback(cb) {}
      inline ParallelExecutionStreamQueue(ParallelExecutionStreamQueue && o)
        : m_stream( std::move(o.m_stream) )
        , m_queue( std::move(o.m_queue) )
      {
        o.m_stream = nullptr;
        o.m_queue = nullptr;
      }
      inline ParallelExecutionStreamQueue& operator = (ParallelExecutionStreamQueue && o)
      {
        m_stream = std::move(o.m_stream);
        m_queue = std::move(o.m_queue);
        o.m_stream = nullptr;
        o.m_queue = nullptr;
      }
      
      // triggers stream synchronization and callbacks
      ~ParallelExecutionStreamQueue();
    };
    
    // just a shorcut to start building a stream queue when a parallel operation is pushed onto a stream
    inline
    ParallelExecutionStreamQueue operator << ( ParallelExecutionStream& pes , ParallelExecutionContext& pec ) { return ParallelExecutionStreamQueue{&pes} << pec ; }
    
    // real implementation of how a parallel operation is pushed onto a stream queue
    ParallelExecutionStreamQueue operator << ( ParallelExecutionStreamQueue && pes , ParallelExecutionContext& pec );
  }

}

