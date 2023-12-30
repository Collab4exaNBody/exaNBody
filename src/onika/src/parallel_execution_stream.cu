#include <onika/parallel/parallel_execution_stream.h>

namespace onika
{
  namespace parallel
  {

    // real implementation of how a parallel operation is pushed onto a stream queue
    ParallelExecutionStreamQueue operator << ( ParallelExecutionStreamQueue && pes , ParallelExecutionContext& pec )
    {
      const auto & func = * reinterpret_cast<BlockParallelForHostFunctor*>( pec.m_host_scratch.functor_data );
      
      switch( pec->m_execution_target )
      {
        case EXECUTION_TARGET_OPENMP :
        {
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
          func.stream_init_gpu_functor( &pec , &pes );
          
          // launch compute kernel
          if( pec.m_grid_size > 0 )
          {
            ONIKA_CU_LAUNCH_KERNEL(pec.m_grid_size,pec.m_block_size,0,pes.m_cu_stream, block_parallel_for_gpu_kernel_workstealing, N, pec.m_cuda_scratch.get() );
          }
          else
          {
            ONIKA_CU_LAUNCH_KERNEL(N,pec.m_block_size,0,pes.m_cu_stream, block_parallel_for_gpu_kernel_regulargrid, pec.m_cuda_scratch.get() );
          }
          
          // call device functor destructor
          ONIKA_CU_LAUNCH_KERNEL(1,1,0,pes.m_cu_stream,finalize_functor_adapter,pec.m_cuda_scratch.get());
          
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

  }
}


