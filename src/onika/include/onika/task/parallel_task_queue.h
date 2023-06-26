#pragma once

#include <onika/task/parallel_task.h>
#include <onika/memory/streaming_storage_pool.h>

#include <memory>
#include <utility>
#include <unordered_map>
#include <deque>
#include <atomic>
#include <unordered_set>

#include <omp.h>

#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_context.h>

namespace onika
{
  namespace task
  {

    class ParallelTaskQueue
    {    

      enum TokenType { NOOP , SCHEDULE , FLUSH , STOP };

      struct ParallelTaskToken
      {
        TokenType type = NOOP;
        ParallelTask* ptask = nullptr;
      };

      void enqueue_token( ParallelTaskToken && t );
      void scheduler_loop();

    public:
      using PTAllocator = memory::StreamingStoragePool;
      
      ParallelTaskQueue();
      
      inline void set_immediate_execution( bool yn ) { m_immediate_execute = yn; }
      
      void enqueue_ptask( ParallelTask* pt );
      inline PTAllocator& allocator() { return m_pt_alloc; }

      void flush();
      void wait_all();
      
      void start();
      void stop();
            
      template<class T> inline T operator << ( T && t )
      {
        assert(t.ptask_queue()==nullptr);
        auto t2 = std::move(t);
        t2.attach_to_queue(this);
        return t2;
      }

      static ParallelTaskQueue& global_ptask_queue();

      inline cudaStream_t* cuda_streams() { return m_cuda_ctx!=nullptr ? m_cuda_ctx->m_threadStream.data() : nullptr ; }
      inline size_t num_cuda_streams() const { return m_cuda_ctx!=nullptr ? m_cuda_ctx->m_threadStream.size() : 0; }
      inline void set_cuda_ctx( cuda::CudaContext* ctx ) { m_cuda_ctx = ctx; }
      inline cuda::CudaContext* cuda_ctx() const { return m_cuda_ctx; }

    private:
      std::deque<ParallelTaskToken> m_pending_ptasks;
      std::unordered_set<ParallelTask*> m_scheduled_ptasks;
      std::unordered_multimap< void* , ParallelTask* > m_reading_ptasks;
      std::unordered_multimap< void* , ParallelTask* > m_writing_ptasks;
      PTAllocator m_pt_alloc;

      omp_lock_t m_queue_lock;

      cuda::CudaContext* m_cuda_ctx = nullptr;
      
      bool m_immediate_execute = false;
      
      static std::atomic<ParallelTaskQueue*> s_global_ptask_queue;
    };

    // short-cut, can be overloaded in some other namespace to select another default task queue
    inline ParallelTaskQueue& default_ptask_queue() { return ParallelTaskQueue::global_ptask_queue(); }

  }
}
