#pragma once

#include <onika/cuda/cuda_context.h>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/block_parallel_for_functor.h>
#include <onika/parallel/stream_utils.h>
#include <mutex>
#include <atomic>

namespace onika
{

  namespace parallel
  {

    // allows asynchronous sequential execution of parallel executions queued in the same stream
    // multiple kernel execution concurrency can be handled manually using several streams (same as Cuda stream)
    struct ParallelExecutionStream
    {
      // GPU device context, null if non device available for parallel execution
      // any parallel executiion enqueued to this stream must have either a null CudaContext or the same context as the stream
      onika::cuda::CudaContext* m_cuda_ctx = nullptr; 
      onikaStream_t m_cu_stream = 0;
      uint32_t m_stream_id = 0;
      std::atomic<uint32_t> m_omp_execution_count = 0;
      std::mutex m_mutex;
      
      inline void wait()
      {
        std::lock_guard lk( m_mutex );
        wait_nolock();
      }

      inline void wait_nolock()
      {
        // OpenMP wait
        if( m_omp_execution_count.load() > 0 )
        {
          auto * st = this;
#         pragma omp task default(none) firstprivate(st) depend(in:st[0]) if(0)
          {
            int n = st->m_omp_execution_count.load();
            if( n > 0 )
            {
              log_err()<<"Internal error : unterminated OpenMP tasks ("<<n<<") in queue remain"<<std::endl;
              std::abort();
            }
          }
        }
        
        // Cuda wait
        if( m_cuda_ctx != nullptr )
        {
          ONIKA_CU_CHECK_ERRORS( ONIKA_CU_STREAM_SYNCHRONIZE( m_cu_stream ) );
        }
      }
    };

    struct ParallelExecutionStreamQueue
    {
      ParallelExecutionStream* m_stream = nullptr;
      ParallelExecutionContext* m_queue = nullptr;
      ParallelExecutionStreamQueue() = default;
      
      inline ParallelExecutionStreamQueue(ParallelExecutionStream* st) : m_stream(st) , m_queue(nullptr) {}
      
      inline ParallelExecutionStreamQueue(ParallelExecutionStreamQueue && o)
        : m_stream( std::move(o.m_stream) )
        , m_queue( std::move(o.m_queue) )
      {
        o.m_stream = nullptr;
        o.m_queue = nullptr;
      }
      
      inline ParallelExecutionStreamQueue& operator = (ParallelExecutionStreamQueue && o)
      {
        wait();
        m_stream = std::move(o.m_stream);
        m_queue = std::move(o.m_queue);
        o.m_stream = nullptr;
        o.m_queue = nullptr;
        return *this;
      }
         
      inline ~ParallelExecutionStreamQueue()
      {
        wait();
        m_stream = nullptr;
      }
      
      inline void wait()
      {
        if( m_stream != nullptr && m_queue != nullptr )
        {
          std::lock_guard lk( m_stream->m_mutex );

          // synchronize stream
          m_stream->wait_nolock();
          
          // collect execution times
          auto* pec = m_queue;
          while(pec!=nullptr)
          {
            float Tgpu = 0.0;
            if( pec->m_execution_target == ParallelExecutionContext::EXECUTION_TARGET_CUDA )
            {
              ONIKA_CU_CHECK_ERRORS( ONIKA_CU_EVENT_ELAPSED(Tgpu,pec->m_start_evt,pec->m_stop_evt) );
              pec->m_total_gpu_execution_time = Tgpu;
            }
            auto* next = pec->m_next;
            if( pec->m_finalize.m_func != nullptr )
            {
              // may account for elapsed time, and free pec allocated memory
              ( * pec->m_finalize.m_func ) ( pec , pec->m_finalize.m_data );
            }
            reinterpret_cast<BlockParallelForHostFunctor*>(pec->m_host_scratch.functor_data)-> ~BlockParallelForHostFunctor();
            pec = next;
          }
          
          m_queue = nullptr;
        }
      }
      
      inline bool query_status()    
      {
        std::lock_guard lk( m_stream->m_mutex );
        if( m_stream == nullptr || m_queue == nullptr )
        {
          return true;
        }
        if( m_stream->m_omp_execution_count.load() > 0 )
        {
          return false;
        }
        if( m_stream->m_cuda_ctx != nullptr && m_queue != nullptr )
        {
          assert( m_queue->m_stop_evt != nullptr );
          if( ONIKA_CU_EVENT_QUERY( m_queue->m_stop_evt ) != onikaSuccess )
          {
            return false;
          }
        }
        wait();
        return true;
      }
      
      inline bool empty() const     
      {
        if( m_stream == nullptr ) return true;
        std::lock_guard lk( m_stream->m_mutex );
        return m_queue == nullptr;
      }
      
      inline bool has_stream() const   
      {
        return m_stream != nullptr;
      }

    };



    // temporarily holds ParallelExecutionContext instance until it is either queued in a stream or graph execution flow,
    // or destroyed, in which case it inserts instance onto the default stream queue
    struct ParallelExecutionWrapper
    {
      ParallelExecutionContext* m_pec = nullptr;
      inline ~ParallelExecutionWrapper();      
    };



    // real implementation of how a parallel operation is pushed onto a stream queue
    inline ParallelExecutionStreamQueue operator << ( ParallelExecutionStreamQueue && pes , ParallelExecutionWrapper && pew )
    {
      assert( pes.m_stream != nullptr );
      std::lock_guard lk( pes.m_stream->m_mutex );

      assert( pew.m_pec != nullptr );
      auto & pec = * pew.m_pec;
      pew.m_pec = nullptr;

      const auto & func = * reinterpret_cast<BlockParallelForHostFunctor*>( pec.m_host_scratch.functor_data );
      
      switch( pec.m_execution_target )
      {
        case ParallelExecutionContext::EXECUTION_TARGET_OPENMP :
        {
          if( pec.m_omp_num_tasks == 0 )
          {
            func.execute_omp_parallel_region( &pec , pes.m_stream );
          }
          else
          {
            // preferred number of tasks : trade off between overhead (less is better) and load balancing (more is better)
            const unsigned int num_tasks = pec.m_omp_num_tasks * onika::parallel::ParallelExecutionContext::parallel_task_core_mult() + onika::parallel::ParallelExecutionContext::parallel_task_core_add() ;
            func.execute_omp_tasks( &pec , pes.m_stream , num_tasks );
          }
        }
        break;
        
        case ParallelExecutionContext::EXECUTION_TARGET_CUDA :
        {
          if( pes.m_stream->m_cuda_ctx == nullptr || pes.m_stream->m_cuda_ctx != pec.m_cuda_ctx )
          {
            std::cerr << "Mismatch Cuda context, cannot queue parallel execution to this stream" << std::endl;
            std::abort();
          }
        
          // if device side scratch space hasn't be allocated yet, do it now
          pec.init_device_scratch();
          
          // insert start event for profiling
          assert( pec.m_start_evt != nullptr );
          ONIKA_CU_CHECK_ERRORS( ONIKA_CU_STREAM_EVENT( pec.m_start_evt, pes.m_stream->m_cu_stream ) );

          // copy in return data intial value. mainly useful for reduction where you might want to start reduction with a given initial value
          if( pec.m_return_data_input != nullptr && pec.m_return_data_size > 0 )
          {
            ONIKA_CU_CHECK_ERRORS( ONIKA_CU_MEMCPY( pec.m_cuda_scratch->return_data, pec.m_return_data_input , pec.m_return_data_size , pes.m_stream->m_cu_stream ) );
          }

          // sets all scratch counters to 0
          if( pec.m_reset_counters || pec.m_grid_size > 0 )
          {
            ONIKA_CU_CHECK_ERRORS( ONIKA_CU_MEMSET( pec.m_cuda_scratch->counters, 0, GPUKernelExecutionScratch::MAX_COUNTERS * sizeof(unsigned long long int), pes.m_stream->m_cu_stream ) );
          }

          // Instantiaite device side functor : calls constructor with a placement new using scratch "functor_data" space
          // then call functor prolog if available
          func.stream_gpu_initialize( &pec , pes.m_stream );
          func.stream_gpu_kernel( &pec , pes.m_stream );
          func.stream_gpu_finalize( &pec , pes.m_stream );
          
          // copy out return data to host space at given pointer
          if( pec.m_return_data_output != nullptr && pec.m_return_data_size > 0 )
          {
            ONIKA_CU_CHECK_ERRORS( ONIKA_CU_MEMCPY( pec.m_return_data_output , pec.m_cuda_scratch->return_data , pec.m_return_data_size , pes.m_stream->m_cu_stream ) );
          }
          
          // inserts a callback to stream if user passed one in
          if( pec.m_execution_end_callback.m_func != nullptr )
          {
            ONIKA_CU_CHECK_ERRORS( ONIKA_CU_STREAM_ADD_CALLBACK(pes.m_stream->m_cu_stream, ParallelExecutionContext::execution_end_callback , &pec ) );
          }
          
          // inserts stop event to account for total execution time
          assert( pec.m_stop_evt != nullptr );
          ONIKA_CU_CHECK_ERRORS( ONIKA_CU_STREAM_EVENT( pec.m_stop_evt, pes.m_stream->m_cu_stream ) );
        }
        break;          
        
        default:
        {
          std::cerr << "Invalid execution target" << std::endl;
          std::abort();
        }
        break;
      }
      
      // add parallel execution to queue
      pec.m_next = pes.m_queue;
      pes.m_queue = &pec;
      
      return std::move(pes);
    }



    inline ParallelExecutionWrapper::~ParallelExecutionWrapper()
    {
      if( m_pec != nullptr )
      {
        ParallelExecutionStreamQueue{m_pec->m_default_stream} << ParallelExecutionWrapper{m_pec};
        m_pec = nullptr;
      }
    }

    
  }

}

