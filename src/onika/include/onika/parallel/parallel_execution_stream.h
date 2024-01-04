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

    // GPU execution kernel for fixed size grid, using workstealing element assignment to blocks
    ONIKA_DEVICE_KERNEL_FUNC
    ONIKA_DEVICE_KERNEL_BOUNDS(ONIKA_CU_MAX_THREADS_PER_BLOCK,ONIKA_CU_MIN_BLOCKS_PER_SM)
    [[maybe_unused]] static
    void block_parallel_for_gpu_kernel_workstealing( uint64_t N, GPUKernelExecutionScratch* scratch )
    {
      // avoid use of compute buffer when possible
      const auto & func = * reinterpret_cast<BlockParallelForGPUFunctor*>( scratch->functor_data );
/*
      if( ONIKA_CU_THREAD_IDX == 0 )
      {
        static constexpr int sz = sizeof(BlockParallelForGPUFunctor);
        printf("GPU: call(%d) functor @%p , scratch @%p , size=%d",int(ONIKA_CU_BLOCK_IDX),&func,scratch,sz);
        for(int i=0;i<sz;i++) printf("%c%02X" , ((i%16)==0) ? '\n' : ' ' , int(scratch->functor_data[i])&0xFF );
        printf("\n");
      }
      ONIKA_CU_BLOCK_SYNC();
*/
      ONIKA_CU_BLOCK_SHARED unsigned int i;
      do
      {
        if( ONIKA_CU_THREAD_IDX == 0 )
        {
          i = ONIKA_CU_ATOMIC_ADD( scratch->counters[0] , 1u );
          //printf("processing cell #%d\n",int(cell_a_no_gl));
        }
        ONIKA_CU_BLOCK_SYNC();
        if( i < N )
        {
          // if( ONIKA_CU_THREAD_IDX == 0 ) printf("processing cell #%d\n",i);
          func( i );
        }
      }
      while( i < N );
    }

    // GPU execution kernel for adaptable size grid, a.k.a. conventional Cuda kernel execution on N element blocks
    ONIKA_DEVICE_KERNEL_FUNC
    ONIKA_DEVICE_KERNEL_BOUNDS(ONIKA_CU_MAX_THREADS_PER_BLOCK,ONIKA_CU_MIN_BLOCKS_PER_SM)
    [[maybe_unused]] static
    void block_parallel_for_gpu_kernel_regulargrid( GPUKernelExecutionScratch* scratch )
    {
      const auto & func = * reinterpret_cast<BlockParallelForGPUFunctor*>( scratch->functor_data );
/*
      if( ONIKA_CU_THREAD_IDX == 0 )
      {
        static constexpr int sz = sizeof(BlockParallelForGPUFunctor);
        printf("GPU: call(%d) functor @%p , scratch @%p , size=%d",int(ONIKA_CU_BLOCK_IDX),&func,scratch->functor_data,sz);
        for(int i=0;i<sz;i++) printf("%c%02X" , ((i%16)==0) ? '\n' : ' ' , int(scratch->functor_data[i])&0xFF );
        printf("\n");
      }
      ONIKA_CU_BLOCK_SYNC();
*/      
      func( ONIKA_CU_BLOCK_IDX );
    }

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

          // OpenMP wait
          if( m_stream->m_omp_execution_count.load() > 0 )
          {
            auto * st = m_stream;
#           pragma omp task default(none) firstprivate(st) depend(in:st[0]) if(0)
            {
              if( st->m_omp_execution_count.load() > 0 )
              {
                log_err()<<"Internal error : unterminated OpenMP tasks in queue remain"<<std::endl;
                std::abort();
              }
            }
          }
          
          // Cuda wait
          if( m_stream->m_cuda_ctx != nullptr )
          {
            ONIKA_CU_STREAM_SYNCHRONIZE( m_stream->m_cu_stream );
          }
          
          // collect execution times
          auto* pec = m_queue;
          while(pec!=nullptr)
          {
            float Tgpu = 0.0;
            if( pec->m_execution_target == ParallelExecutionContext::EXECUTION_TARGET_CUDA )
            {
              ONIKA_CU_EVENT_ELAPSED(Tgpu,pec->m_start_evt,pec->m_stop_evt);
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
          if( cudaEventQuery( m_queue->m_stop_evt ) != cudaSuccess )
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

      assert( pec.m_parallel_space.m_start == 0 && pec.m_parallel_space.m_idx == nullptr );
      const size_t N = pec.m_parallel_space.m_end;
      const auto & func = * reinterpret_cast<BlockParallelForHostFunctor*>( pec.m_host_scratch.functor_data );
      
      switch( pec.m_execution_target )
      {
        case ParallelExecutionContext::EXECUTION_TARGET_OPENMP :
        {
          pes.m_stream->m_omp_execution_count.fetch_add(1);
          if( pec.m_omp_num_tasks == 0 )
          {
            const auto & func = * reinterpret_cast<BlockParallelForHostFunctor*>( pec.m_host_scratch.functor_data );
#           ifdef XSTAMP_OMP_NUM_THREADS_WORKAROUND
            omp_set_num_threads( omp_get_max_threads() );
#           endif
            const auto T0 = std::chrono::high_resolution_clock::now();
            func( block_parallel_for_prolog_t{} );
#           pragma omp parallel
            {
#             pragma omp for schedule(dynamic)
              for(uint64_t i=0;i<N;i++) { func( i ); }
            }
            func( block_parallel_for_epilog_t{} );
            pec.m_total_cpu_execution_time = ( std::chrono::high_resolution_clock::now() - T0 ).count() / 1000000.0;
            if( pec.m_execution_end_callback.m_func != nullptr )
            {
              (* pec.m_execution_end_callback.m_func) ( pec.m_execution_end_callback.m_data );
            }
            pes.m_stream->m_omp_execution_count.fetch_sub(1);
          }
          else
          {
            // preferred number of tasks : trade off between overhead (less is better) and load balancing (more is better)
            const unsigned int num_tasks = pec.m_omp_num_tasks * onika::parallel::ParallelExecutionContext::parallel_task_core_mult() + onika::parallel::ParallelExecutionContext::parallel_task_core_add() ;
            // enclose a taskgroup inside a task, so that we can wait for a single task which itself waits for the completion of the whole taskloop
            auto * ptr_pec = &pec;
            auto * ptr_str = pes.m_stream;
            // refrenced variables must be privately copied, because the task may run after this function ends
#           pragma omp task default(none) firstprivate(ptr_pec,ptr_str,num_tasks,N) depend(inout:pes.m_stream[0])
            {
              const auto & func = * reinterpret_cast<BlockParallelForHostFunctor*>( ptr_pec->m_host_scratch.functor_data );
              const auto T0 = std::chrono::high_resolution_clock::now();
              func( block_parallel_for_prolog_t{} );              
              // implicit taskgroup, ensures taskloop has completed before enclosing task ends
              // all refrenced variables can be shared because of implicit enclosing taskgroup
#             pragma omp taskloop default(none) shared(ptr_pec,num_tasks,func,N) num_tasks(num_tasks)
              for(uint64_t i=0;i<N;i++) { func( i ); }
              // here all tasks of taskloop have completed, since notaskgroup clause is not specified              
              func( block_parallel_for_epilog_t{} );            
              ptr_pec->m_total_cpu_execution_time = ( std::chrono::high_resolution_clock::now() - T0 ).count() / 1000000.0;
              if( ptr_pec->m_execution_end_callback.m_func != nullptr )
              {
                (* ptr_pec->m_execution_end_callback.m_func) ( ptr_pec->m_execution_end_callback.m_data );
              }
              ptr_str->m_omp_execution_count.fetch_sub(1);
            }
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
          checkCudaErrors( ONIKA_CU_STREAM_EVENT( pec.m_start_evt, pes.m_stream->m_cu_stream ) );

          // copy in return data intial value. mainly useful for reduction where you might want to start reduction with a given initial value
          if( pec.m_return_data_input != nullptr && pec.m_return_data_size > 0 )
          {
            checkCudaErrors( ONIKA_CU_MEMCPY( pec.m_cuda_scratch->return_data, pec.m_return_data_input , pec.m_return_data_size , pes.m_stream->m_cu_stream ) );
          }

          // sets all scratch counters to 0
          if( pec.m_reset_counters || pec.m_grid_size > 0 )
          {
            checkCudaErrors( ONIKA_CU_MEMSET( pec.m_cuda_scratch->counters, 0, GPUKernelExecutionScratch::MAX_COUNTERS * sizeof(unsigned long long int), pes.m_stream->m_cu_stream ) );
          }

          // Instantiaite device side functor : calls constructor with a placement new using scratch "functor_data" space
          // then call functor prolog if available
          func.stream_gpu_initialize( &pec , pes.m_stream );
          
          // launch compute kernel
          if( pec.m_grid_size > 0 )
          {
            ONIKA_CU_LAUNCH_KERNEL(pec.m_grid_size,pec.m_block_size,0,pes.m_stream->m_cu_stream, block_parallel_for_gpu_kernel_workstealing, N, pec.m_cuda_scratch.get() );
          }
          else
          {
            ONIKA_CU_LAUNCH_KERNEL(N,pec.m_block_size,0,pes.m_stream->m_cu_stream, block_parallel_for_gpu_kernel_regulargrid, pec.m_cuda_scratch.get() );
          }
          
          // executes prolog through functor, if available, then call device functor destructor
          ONIKA_CU_LAUNCH_KERNEL(1,pec.m_block_size,0,pes.m_stream->m_cu_stream,gpu_functor_finalize,pec.m_cuda_scratch.get());
          
          // copy out return data to host space at given pointer
          if( pec.m_return_data_output != nullptr && pec.m_return_data_size > 0 )
          {
            checkCudaErrors( ONIKA_CU_MEMCPY( pec.m_return_data_output , pec.m_cuda_scratch->return_data , pec.m_return_data_size , pes.m_stream->m_cu_stream ) );
          }
          
          // inserts a callback to stream if user passed one in
          if( pec.m_execution_end_callback.m_func != nullptr )
          {
            checkCudaErrors( cudaStreamAddCallback(pes.m_stream->m_cu_stream, ParallelExecutionContext::execution_end_callback , &pec , 0 ) );
          }
          
          // inserts stop event to account for total execution time
          assert( pec.m_stop_evt != nullptr );
          checkCudaErrors( ONIKA_CU_STREAM_EVENT( pec.m_stop_evt, pes.m_stream->m_cu_stream ) );
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

