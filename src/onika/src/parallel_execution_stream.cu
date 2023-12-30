#include <onika/parallel/parallel_execution_stream.h>

namespace onika
{
  namespace parallel
  {

    // real implementation of how a parallel operation is pushed onto a stream queue
    ParallelExecutionStreamQueue operator << ( ParallelExecutionStreamQueue && pes , ParallelExecutionContext& pec )
    {
      assert( ptr_pec->m_parallel_space.m_start == 0 && ptr_pec->m_parallel_space.m_idx == nullptr );
      const size_t N = ptr_pec->m_parallel_space.m_end;
      const auto & func = * reinterpret_cast<BlockParallelForHostFunctor*>( pec.m_host_scratch.functor_data );
      
      switch( pec->m_execution_target )
      {
        case EXECUTION_TARGET_OPENMP :
        {
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
            pec.m_total_async_cpu_execution_time = ( std::chrono::high_resolution_clock::now() - T0 ).count() / 1000000.0;
            if( pec.m_execution_end_callback.m_func != nullptr )
            {
              (* pec.m_execution_end_callback.m_func) ( ptr_pec->m_execution_end_callback.m_data );
            }
          }
          else
          {
            // enclose a taskgroup inside a task, so that we can wait for a single task which itself waits for the completion of the whole taskloop
            auto * ptr_pec = &pec;
            // refrenced variables must be privately copied, because the task may run after this function ends
#           pragma omp task default(none) firstprivate(ptr_pec,N) depend(inout:pes.m_stream[0])
            {
              const auto & func = * reinterpret_cast<BlockParallelForHostFunctor*>( ptr_pec->m_host_scratch.functor_data );
              const auto T0 = std::chrono::high_resolution_clock::now();
              func( block_parallel_for_prolog_t{} );              
              // implicit taskgroup, ensures taskloop has completed before enclosing task ends
              // all refrenced variables can be shared because of implicit enclosing taskgroup
#             pragma omp taskloop default(none) shared(ptr_pec,func,N) num_tasks(ptr_pec->m_omp_num_tasks)
              for(uint64_t i=0;i<N;i++) { func( i ); }
              // here all tasks of taskloop have completed, since notaskgroup clause is not specified              
              func( block_parallel_for_epilog_t{} );            
              ptr_pec->m_total_async_cpu_execution_time = ( std::chrono::high_resolution_clock::now() - T0 ).count() / 1000000.0;
              if( ptr_pec->m_execution_end_callback.m_func != nullptr )
              {
                (* ptr_pec->m_execution_end_callback.m_func) ( ptr_pec->m_execution_end_callback.m_data );
              }
            }
          }

        }
        break;
        
        case EXECUTION_TARGET_CUDA :
        {
          // insert start event for profiling
          checkCudaErrors( ONIKA_CU_STREAM_EVENT( pec.m_start_evt, pes.m_cu_stream ) );

          // copy in return data intial value. mainly useful for reduction where you might want to start reduction with a given initial value
          if( pec.m_return_data_input != nullptr && pec.m_return_data_size > 0 )
          {
            checkCudaErrors( ONIKA_CU_MEMCPY( pec.m_cuda_scratch->return_data, pec.m_return_data_input , pec.m_return_data_size , pes.m_cu_stream ) );
          }

          // sets all scratch counters to 0
          if( pec.m_reset_counters || pec.m_grid_size > 0 )
          {
            checkCudaErrors( ONIKA_CU_MEMSET( pec.m_cuda_scratch->counters, 0, GPUKernelExecutionScratch::MAX_COUNTERS * sizeof(unsigned long long int), pes.m_cu_stream ) );
          }

          // Instantiaite device side functor : calls constructor with a placement new using scratch "functor_data" space
          // then call functor prolog if available
          func.stream_gpu_initialize( &pec , &pes );
          
          // launch compute kernel
          if( pec.m_grid_size > 0 )
          {
            ONIKA_CU_LAUNCH_KERNEL(pec.m_grid_size,pec.m_block_size,0,pes.m_cu_stream, block_parallel_for_gpu_kernel_workstealing, N, pec.m_cuda_scratch.get() );
          }
          else
          {
            ONIKA_CU_LAUNCH_KERNEL(N,pec.m_block_size,0,pes.m_cu_stream, block_parallel_for_gpu_kernel_regulargrid, pec.m_cuda_scratch.get() );
          }
          
          // executes prolog through functor, if available, then call device functor destructor
          ONIKA_CU_LAUNCH_KERNEL(1,pec.m_block_size,0,pes.m_cu_stream,gpu_functor_finalize,pec.m_cuda_scratch.get());
          
          // copy out return data to host space at given pointer
          if( pec.m_return_data_output != nullptr && pec.m_return_data_size > 0 )
          {
            checkCudaErrors( ONIKA_CU_MEMCPY( pec.m_return_data_output , pec.m_cuda_scratch->return_data , pec.m_return_data_size , pes.m_cu_stream ) );
          }
          
          // inserts a callback to stream if user passed one in
          if( pec.m_execution_end_callback.m_func != nullptr )
          {
            checkCudaErrors( cudaStreamAddCallback(pes.m_cu_stream, ParallelExecutionContext::execution_end_callback , &pec , 0 ) );
          }
          
          // inserts stop event to account for total execution time
          checkCudaErrors( ONIKA_CU_STREAM_EVENT( pec.m_stop_evt, pes.m_cu_stream ) );
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

    ParallelExecutionStreamQueue::~ParallelExecutionStreamQueue()
    {
      if( m_stream != nullptr )
      {
        // OpenMP wait
#       pragma omp task default(none) depend(in:m_stream[0]) if(0)
        {}
        
        // Cuda wait
        ONIKA_CU_STREAM_SYNCHRONIZE( m_stream->m_cu_stream );
        
        // collect execution times
        auto* pec = m_queue;
        while(pec!=nullptr)
        {
          float Tgpu = 0.0;
          ONIKA_CU_EVENT_ELAPSED(Tgpu,pec->m_start_evt,pec->m_stop_evt);
          pec->m_total_gpu_execution_time = Tgpu;
          pec->push_execution_time();
          auto* next = pec->m_next;
          pec_free( pec );
          pec = next;
        }
        m_queue = nullptr;
        m_stream = nullptr;
      }
    }

  }
}


