#pragma once

#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_error.h>
#include <onika/cuda/device_storage.h>
#include <onika/soatl/field_id.h>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/block_parallel_for_functor.h>
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

    template<class FuncT>
    inline void block_parallel_for_omp_kernel(
        uint64_t N
      , const FuncT& func
      , ParallelExecutionContext * exec_ctx
      , unsigned int num_tasks = 0
      , bool async = false
      , ParallelExecutionCallback* user_cb = nullptr
      )
    {
      [[maybe_unused]] static constexpr bool functor_has_prolog     = lambda_is_compatible_with_v<FuncT,void,ParallelExecutionContext*,block_parallel_for_prolog_t>;
      static constexpr bool functor_has_cpu_prolog = lambda_is_compatible_with_v<FuncT,void,ParallelExecutionContext*,block_parallel_for_cpu_prolog_t>;
      [[maybe_unused]] static constexpr bool functor_has_epilog     = lambda_is_compatible_with_v<FuncT,void,ParallelExecutionContext*,block_parallel_for_epilog_t>;
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

    ONIKA_DEVICE_KERNEL_FUNC
    ONIKA_DEVICE_KERNEL_BOUNDS(ONIKA_CU_MAX_THREADS_PER_BLOCK,ONIKA_CU_MIN_BLOCKS_PER_SM)
    void block_parallel_for_gpu_kernel_workstealing( uint64_t N, GPUKernelExecutionScratch* scratch );

    ONIKA_DEVICE_KERNEL_FUNC
    ONIKA_DEVICE_KERNEL_BOUNDS(ONIKA_CU_MAX_THREADS_PER_BLOCK,ONIKA_CU_MIN_BLOCKS_PER_SM)
    void block_parallel_for_gpu_kernel_regulargrid( GPUKernelExecutionScratch* scratch );

    /*
     * BlockParallelForOptions holds options passed to block_parallel_for
     */
    struct BlockParallelForOptions
    {
      ParallelExecutionCallback user_cb = {};
      void * return_data = nullptr;
      size_t return_data_size = 0;
      bool enable_gpu = true;
      bool fixed_gpu_grid_size = false;
    };

    template< class FuncT >
    static inline
    ParallelExecutionWrapper
    block_parallel_for(
        uint64_t N
      , const FuncT& func
      , ParallelExecutionContext * pec
      , const BlockParallelForOptions& opts = BlockParallelForOptions{} )
    {    
      static_assert( lambda_is_compatible_with_v<FuncT,void,uint64_t> , "Functor in argument is incompatible with void(uint64_t) call signature" );
      static_assert( sizeof(BlockParallelForHostAdapter<FuncT>) <= HostKernelExecutionScratch::MAX_FUNCTOR_SIZE );

      assert( pec != nullptr );
    
      const auto [ user_cb
                 , return_data 
                 , return_data_size 
                 , enable_gpu 
                 , fixed_gpu_grid_size
                 ] = opts;

      // construct virtual functor adapter inplace, using reserved functor space
      new(pec->m_host_scratch.functor_data) BlockParallelForHostAdapter<FuncT>( func );

      pec->m_execution_end_callback = user_cb;
      pec->m_parallel_space = ParallelExecutionSpace{ 0, N, nullptr };
    
      if constexpr ( BlockParallelForFunctorTraits<FuncT>::CudaCompatible )
      {
        bool allow_cuda_exec = enable_gpu ;
        if( allow_cuda_exec ) allow_cuda_exec = ( pec->m_cuda_ctx != nullptr );
        if( allow_cuda_exec ) allow_cuda_exec = pec->m_cuda_ctx->has_devices();
        if( allow_cuda_exec )
        {
          pec->m_execution_target = ParallelExecutionContext::EXECUTION_TARGET_CUDA;
          pec->m_block_size = std::min( static_cast<size_t>(ONIKA_CU_MAX_THREADS_PER_BLOCK) , static_cast<size_t>(onika::parallel::ParallelExecutionContext::gpu_block_size()) );
          pec->m_grid_size = pec->m_cuda_ctx->m_devices[0].m_deviceProp.multiProcessorCount
                                      * onika::parallel::ParallelExecutionContext::gpu_sm_mult()
                                      + onika::parallel::ParallelExecutionContext::gpu_sm_add();
          if( ! fixed_gpu_grid_size )
          { 
            pec->m_grid_size = 0;
          }
          pec->m_reset_counters = fixed_gpu_grid_size;

          if( return_data != nullptr && return_data_size > 0 )
          {
            pec->set_return_data_input( return_data , return_data_size );
            pec->set_return_data_output( return_data , return_data_size );
          }
          else
          {
            pec->set_return_data_input( nullptr , 0 );
            pec->set_return_data_output( nullptr , 0 );
          }
          return {*pec};
        }
      }

      // ================== CPU / OpenMP execution path ====================
      pec->m_execution_target = ParallelExecutionContext::EXECUTION_TARGET_OPENMP;
      return {pec};
    }
    
  }

}

