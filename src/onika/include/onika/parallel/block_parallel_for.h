#pragma once

#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_error.h>
#include <onika/cuda/device_storage.h>
#include <onika/soatl/field_id.h>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/lambda_tools.h>
#include <onika/stream_utils.h>

#ifdef XSTAMP_OMP_NUM_THREADS_WORKAROUND
#include <omp.h>
#endif

namespace onika
{

  namespace parallel
  {
    // user can add an overloaded call operator taking one of this type as its only parameter
    // an overload with block_parallel_for_prolog_t will be used both as CPU and GPU launch prolog
    // while and overload with block_parallel_for_gpu_prolog_t will e called only in case of a GPU launch
    // and similarily with block_parallel_for_cpu_prolog_t
    struct block_parallel_for_prolog_t {};
    struct block_parallel_for_gpu_prolog_t : public block_parallel_for_prolog_t {};
    struct block_parallel_for_cpu_prolog_t : public block_parallel_for_prolog_t {};

    // same as block_parallel_for_prolog_t but for end of parallel for execution
    struct block_parallel_for_epilog_t {};
    struct block_parallel_for_gpu_epilog_t : public block_parallel_for_epilog_t {};
    struct block_parallel_for_cpu_epilog_t : public block_parallel_for_epilog_t {};
  
    // this template is here to know if compute buffer must be built or computation must be ran on the fly
    template<class FuncT> struct BlockParallelForFunctorTraits
    {      
      static inline constexpr bool CudaCompatible = false;
    };

    template< class FuncT>
    ONIKA_DEVICE_KERNEL_FUNC
    ONIKA_DEVICE_KERNEL_BOUNDS(ONIKA_CU_MAX_THREADS_PER_BLOCK,ONIKA_CU_MIN_BLOCKS_PER_SM)
    void block_parallel_for_gpu_kernel_workstealing( uint64_t N, const FuncT func, GPUKernelExecutionScratch* scratch )
    {
      // avoid use of compute buffer when possible
      ONIKA_CU_BLOCK_SHARED unsigned int i;
      do
      {
        if( ONIKA_CU_THREAD_IDX == 0 )
        {
          i = ONIKA_CU_ATOMIC_ADD( scratch->counters[0] , 1u );
          //printf("processing cell #%d\n",int(cell_a_no_gl));
        }
        ONIKA_CU_BLOCK_SYNC();
        if( i < N ) { func( i ); }
      }
      while( i < N );
    }

    template< class FuncT>
    ONIKA_DEVICE_KERNEL_FUNC
    ONIKA_DEVICE_KERNEL_BOUNDS(ONIKA_CU_MAX_THREADS_PER_BLOCK,ONIKA_CU_MIN_BLOCKS_PER_SM)
    void block_parallel_for_gpu_kernel_regulargrid( const FuncT func , GPUKernelExecutionScratch* )
    {
      func( ONIKA_CU_BLOCK_IDX );
    }

    template<class FuncT>
    inline void block_parallel_for_omp_kernel(
        uint64_t N
      , const FuncT& func
      , ParallelExecutionContext * exec_ctx
      , unsigned int num_tasks = 0
      , bool async = false
      , ParallelExecutionStreamCallback* user_cb = nullptr
      )
    {
      static constexpr bool functor_has_prolog     = lambda_is_compatible_with_v<FuncT,void,ParallelExecutionContext*,block_parallel_for_prolog_t>;
      static constexpr bool functor_has_cpu_prolog = lambda_is_compatible_with_v<FuncT,void,ParallelExecutionContext*,block_parallel_for_cpu_prolog_t>;
      static constexpr bool functor_has_epilog     = lambda_is_compatible_with_v<FuncT,void,ParallelExecutionContext*,block_parallel_for_epilog_t>;
      static constexpr bool functor_has_cpu_epilog = lambda_is_compatible_with_v<FuncT,void,ParallelExecutionContext*,block_parallel_for_cpu_epilog_t>;

      if( exec_ctx != nullptr ) // for backward compatibility
      {
//        std::cout<<"notify omp parallel_for start"<<std::endl;
        exec_ctx->omp_kernel_start();
        if( num_tasks == 0 || ! async )
        {
          if      constexpr ( functor_has_cpu_prolog ) { func( exec_ctx, block_parallel_for_cpu_prolog_t{} ); }
          else if constexpr ( functor_has_prolog     ) { func( exec_ctx, block_parallel_for_prolog_t{}     ); }
        }
      }
    
      if( num_tasks == 0 )
      {
#       ifdef XSTAMP_OMP_NUM_THREADS_WORKAROUND
        omp_set_num_threads( omp_get_max_threads() );
#       endif
#       pragma omp parallel
        {
#         pragma omp for schedule(dynamic)
          for(uint64_t i=0;i<N;i++) { func( i ); }
        }
      }
      else if( async )
      {
        // enclose a taskgroup inside a task, so that we can wait for a single task which itself waits for the completion of the whole taskloop
        auto cp_func = func;
        // refrenced variables must be privately copied, because the task may run after this function ends
#       pragma omp task default(none) firstprivate(cp_func,exec_ctx,N,user_cb,num_tasks) depend(inout:exec_ctx[0])
        {
          // implicit taskgroup, ensures taskloop has completed before enclosing task ends
          // all refrenced variables can be shared because of implicit enclosing taskgroup
#         pragma omp taskloop default(none) shared(cp_func,exec_ctx,N,user_cb,num_tasks) num_tasks(num_tasks)
          for(uint64_t i=0;i<N;i++) { cp_func( i ); }
          // here all tasks of taskloop have completed
          
          if( exec_ctx != nullptr ) // for backward compatibility
          {
            if      constexpr ( functor_has_cpu_epilog ) { cp_func( exec_ctx, block_parallel_for_cpu_epilog_t{} ); }
            else if constexpr ( functor_has_epilog     ) { cp_func( exec_ctx, block_parallel_for_epilog_t{}     ); }
            exec_ctx->omp_kernel_end();
          }
          if( user_cb != nullptr ) { ( * user_cb->m_user_callback )( user_cb->m_exec_ctx , user_cb->m_user_data ); }          
        }
      }
      else
      {
          // implicit taskgroup ensures loop completed at end of structured block.
          // all refrenced variables can be shared because of implicit enclosing taskgroup
#         pragma omp taskloop default(none) shared(func,N,num_tasks) num_tasks(num_tasks) 
          for(uint64_t i=0;i<N;i++) { func( i ); }
          // here all tasks of taskloop have completed
      }

      if( num_tasks == 0 || ! async )
      {
        if( exec_ctx != nullptr ) // for backward compatibility
        {
          if      constexpr ( functor_has_cpu_epilog ) { func( exec_ctx, block_parallel_for_cpu_epilog_t{} ); }
          else if constexpr ( functor_has_epilog     ) { func( exec_ctx, block_parallel_for_epilog_t{}     ); }
          exec_ctx->omp_kernel_end();
        }
        if( user_cb != nullptr ) { ( * user_cb->m_user_callback )( user_cb->m_exec_ctx , user_cb->m_user_data ); }          
      }

    }

    template< class FuncT >
    static inline void block_parallel_for(
        uint64_t N
      , const FuncT& func
      , ParallelExecutionContext * exec_ctx = nullptr
      , bool enable_gpu = true
      , bool async = false
      , ParallelExecutionStreamCallback* user_cb = nullptr
      , void * return_data = nullptr
      , size_t return_data_size = 0)
    {
      static_assert( lambda_is_compatible_with_v<FuncT,void,uint64_t> , "Functor in argument is incompatible with void(uint64_t) call signature" );
      static constexpr bool functor_has_prolog     = lambda_is_compatible_with_v<FuncT,void,ParallelExecutionContext*,block_parallel_for_prolog_t>;
      static constexpr bool functor_has_gpu_prolog = lambda_is_compatible_with_v<FuncT,void,ParallelExecutionContext*,block_parallel_for_gpu_prolog_t>;
      static constexpr bool functor_has_epilog     = lambda_is_compatible_with_v<FuncT,void,ParallelExecutionContext*,block_parallel_for_epilog_t>;
      static constexpr bool functor_has_gpu_epilog = lambda_is_compatible_with_v<FuncT,void,ParallelExecutionContext*,block_parallel_for_gpu_epilog_t>;
    
      if( user_cb != nullptr )
      {
        user_cb->m_exec_ctx = exec_ctx;
      }
    
      if constexpr ( BlockParallelForFunctorTraits<FuncT>::CudaCompatible )
      {
        bool allow_cuda_exec = enable_gpu && ( exec_ctx != nullptr );
        if( allow_cuda_exec ) allow_cuda_exec = ( exec_ctx->m_cuda_ctx != nullptr );
        if( allow_cuda_exec ) allow_cuda_exec = exec_ctx->m_cuda_ctx->has_devices();
        if( allow_cuda_exec )
        {
          exec_ctx->check_initialize();
          const unsigned int BlockSize = std::min( static_cast<size_t>(ONIKA_CU_MAX_THREADS_PER_BLOCK) , static_cast<size_t>(onika::parallel::ParallelExecutionContext::gpu_block_size()) );
//          const unsigned int GridSize = exec_ctx->m_cuda_ctx->m_devices[0].m_deviceProp.multiProcessorCount
//                                      * onika::parallel::ParallelExecutionContext::gpu_sm_mult()
//                                      + onika::parallel::ParallelExecutionContext::gpu_sm_add();

          auto custream = exec_ctx->m_cuda_stream;        
          exec_ctx->gpu_kernel_start();
          exec_ctx->reset_counters();
          if( return_data != nullptr && return_data_size > 0 )
          {
            exec_ctx->set_return_data( return_data , return_data_size );
          }          
          auto * scratch = exec_ctx->m_cuda_scratch.get();

               if constexpr ( functor_has_gpu_prolog ) { func( exec_ctx, block_parallel_for_gpu_prolog_t{} ); }
          else if constexpr ( functor_has_prolog     ) { func( exec_ctx, block_parallel_for_prolog_t{}     ); }


//        ONIKA_CU_LAUNCH_KERNEL(GridSize,BlockSize,0,custream, block_parallel_for_gpu_kernel_workstealing, N, func, scratch );
          ONIKA_CU_LAUNCH_KERNEL(       N,BlockSize,0,custream, block_parallel_for_gpu_kernel_regulargrid ,    func, scratch );


               if constexpr ( functor_has_gpu_epilog ) { func( exec_ctx, block_parallel_for_gpu_epilog_t{} ); }
          else if constexpr ( functor_has_epilog     ) { func( exec_ctx, block_parallel_for_epilog_t{}     ); }

          if( return_data != nullptr && return_data_size > 0 )
          {
            exec_ctx->retrieve_return_data( return_data , return_data_size );
          }
          exec_ctx->gpu_kernel_end();
          exec_ctx->register_stream_callback( user_cb );
          if( ! async ) { exec_ctx->wait(); }

          return;
        }
      }

      // ================== CPU / OpenMP execution path ====================

      int prefered_num_tasks = 0;
      if( exec_ctx != nullptr ) prefered_num_tasks = exec_ctx->m_omp_num_tasks;

      // allow tasking mode, means we're in a parallel/single[/taskgroup] scope
      if( prefered_num_tasks > 0 )
      {
        prefered_num_tasks = prefered_num_tasks * onika::parallel::ParallelExecutionContext::parallel_task_core_mult() + onika::parallel::ParallelExecutionContext::parallel_task_core_add() ;
      }

      block_parallel_for_omp_kernel( N, func, exec_ctx, prefered_num_tasks, async, user_cb );
    }

  }

}

