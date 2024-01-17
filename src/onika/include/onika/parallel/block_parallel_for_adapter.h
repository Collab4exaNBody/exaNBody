#pragma once

#include <onika/parallel/block_parallel_for_functor.h>
#include <onika/parallel/parallel_execution_stream.h>

namespace onika
{
  namespace parallel
  {

    template< class FuncT>
    ONIKA_DEVICE_KERNEL_FUNC
    [[maybe_unused]] static
    void gpu_functor_initialize( ONIKA_CU_GRID_CONSTANT const FuncT func )
    {
    }


    template< class FuncT>
    ONIKA_DEVICE_KERNEL_FUNC
    [[maybe_unused]] static
    void gpu_functor_finalize( ONIKA_CU_GRID_CONSTANT const FuncT func )
    {
    }

    // GPU execution kernel for fixed size grid, using workstealing element assignment to blocks
    template< class FuncT>
    ONIKA_DEVICE_KERNEL_FUNC
    ONIKA_DEVICE_KERNEL_BOUNDS(ONIKA_CU_MAX_THREADS_PER_BLOCK,ONIKA_CU_MIN_BLOCKS_PER_SM)
    [[maybe_unused]] static
    void block_parallel_for_gpu_kernel_workstealing( uint64_t N, GPUKernelExecutionScratch* scratch, ONIKA_CU_GRID_CONSTANT const FuncT func )
    {
      // avoid use of compute buffer when possible
      ONIKA_CU_BLOCK_SHARED unsigned int i;
      do
      {
        if( ONIKA_CU_THREAD_IDX == 0 )
        {
          i = ONIKA_CU_ATOMIC_ADD( scratch->counters[0] , 1u );
        }
        ONIKA_CU_BLOCK_SYNC();
        if( i < N )
        {
          func( i );
        }
      }
      while( i < N );
    }

    // GPU execution kernel for adaptable size grid, a.k.a. conventional Cuda kernel execution on N element blocks
    template< class FuncT>
    ONIKA_DEVICE_KERNEL_FUNC
    ONIKA_DEVICE_KERNEL_BOUNDS(ONIKA_CU_MAX_THREADS_PER_BLOCK,ONIKA_CU_MIN_BLOCKS_PER_SM)
    [[maybe_unused]] static
    void block_parallel_for_gpu_kernel_regulargrid( ONIKA_CU_GRID_CONSTANT const FuncT func )
    {
      func( ONIKA_CU_BLOCK_IDX );
    }


    template<class FuncT, bool GPUSupport>
    class BlockParallelForHostAdapter : public BlockParallelForHostFunctor
    {
      static inline constexpr bool functor_has_prolog     = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_prolog_t>;
      static inline constexpr bool functor_has_cpu_prolog = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_cpu_prolog_t>;
      static inline constexpr bool functor_has_epilog     = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_epilog_t>;
      static inline constexpr bool functor_has_cpu_epilog = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_cpu_epilog_t>;
      const FuncT m_func;
      
    public:
      inline BlockParallelForHostAdapter( const FuncT& f ) : m_func(f) {}


      // ================== GPU stream based execution interface =======================

      inline void stream_gpu_initialize(ParallelExecutionContext* pec , ParallelExecutionStream* pes) const override final
      {
        if constexpr ( GPUSupport )
        {
          static constexpr bool functor_has_gpu_prolog = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_gpu_prolog_t,ParallelExecutionStream*>;
          static constexpr bool functor_has_prolog     = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_prolog_t>;
          if constexpr ( functor_has_gpu_prolog ) { m_func( block_parallel_for_gpu_prolog_t{} , pes ); }
          else if constexpr ( functor_has_prolog ) { m_func( block_parallel_for_prolog_t{} ); }
        }
        else { std::cerr << "called stream_gpu_initialize with no GPU support" << std::endl; std::abort(); }
      }
      
      inline void stream_gpu_kernel(ParallelExecutionContext* pec, ParallelExecutionStream* pes) const override final
      {
        if constexpr ( GPUSupport )
        {
          assert( pec->m_parallel_space.m_start == 0 && pec->m_parallel_space.m_idx == nullptr );
          const size_t N = pec->m_parallel_space.m_end;
          //printf("stream GPU Kernel (%s) N=%d\n",pec->m_tag,int(N));
          // launch compute kernel
          if( pec->m_grid_size > 0 )
          {
            ONIKA_CU_LAUNCH_KERNEL(pec->m_grid_size,pec->m_block_size,0,pes->m_cu_stream, block_parallel_for_gpu_kernel_workstealing, N, pec->m_cuda_scratch.get(), m_func );
          }
          else
          {
            ONIKA_CU_LAUNCH_KERNEL(N,pec->m_block_size,0,pes->m_cu_stream, block_parallel_for_gpu_kernel_regulargrid, m_func );
          }
        }
        else { std::cerr << "called stream_gpu_kernel with no GPU support" << std::endl; std::abort(); }
      }
      
      inline void stream_gpu_finalize(ParallelExecutionContext* pec, ParallelExecutionStream* pes) const override final
      {
        if constexpr ( GPUSupport )
        {
          static constexpr bool functor_has_gpu_epilog = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_gpu_epilog_t,ParallelExecutionStream*>;
          static constexpr bool functor_has_epilog     = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_epilog_t>;
          if constexpr ( functor_has_gpu_epilog ) { m_func( block_parallel_for_gpu_epilog_t{} , pes ); }
          else if constexpr ( functor_has_epilog ) { m_func( block_parallel_for_epilog_t{} ); }
        }
        else { std::cerr << "called stream_gpu_finalize with no GPU support" << std::endl; std::abort(); }
      }


      // ================ CPU OpenMP execution interface ======================
      inline void execute_prolog( ParallelExecutionContext* pec, ParallelExecutionStream* pes ) const override final
      {
        if constexpr (functor_has_cpu_prolog) { m_func(block_parallel_for_cpu_prolog_t{}); }
        else if constexpr (functor_has_prolog) { m_func(block_parallel_for_prolog_t{}); }
      }
      
      inline void execute_omp_parallel_region( ParallelExecutionContext* pec, ParallelExecutionStream* pes ) const override final
      {      
        pes->m_omp_execution_count.fetch_add(1);
        assert( pec->m_parallel_space.m_start == 0 && pec->m_parallel_space.m_idx == nullptr );
        const size_t N = pec->m_parallel_space.m_end;
#       ifdef XSTAMP_OMP_NUM_THREADS_WORKAROUND
        omp_set_num_threads( omp_get_max_threads() );
#       endif
        const auto T0 = std::chrono::high_resolution_clock::now();  
        execute_prolog( pec , pes );
#       pragma omp parallel
        {
#         pragma omp for schedule(dynamic)
          for(uint64_t i=0;i<N;i++)
          {
            m_func( i );
          }
        }
        execute_epilog( pec , pes );
        pec->m_total_cpu_execution_time = ( std::chrono::high_resolution_clock::now() - T0 ).count() / 1000000.0;
        if( pec->m_execution_end_callback.m_func != nullptr )
        {
          (* pec->m_execution_end_callback.m_func) ( pec->m_execution_end_callback.m_data );
        }
        pes->m_omp_execution_count.fetch_sub(1);
      }

      inline void execute_omp_tasks( ParallelExecutionContext* pec, ParallelExecutionStream* pes, unsigned int num_tasks ) const override final
      {
        pes->m_omp_execution_count.fetch_add(1);
        // encloses a taskgroup inside a task, so that we can wait for a single task which itself waits for the completion of the whole taskloop
        // refrenced variables must be privately copied, because the task may run after this function ends
#       pragma omp task default(none) firstprivate(pec,pes,num_tasks) depend(inout:pes[0])
        {
          assert( pec->m_parallel_space.m_start == 0 && pec->m_parallel_space.m_idx == nullptr );
          const size_t N = pec->m_parallel_space.m_end;
          const auto T0 = std::chrono::high_resolution_clock::now();
          execute_prolog( pec , pes );
          if( N > 0 )
          {
            // implicit taskgroup, ensures taskloop has completed before enclosing task ends
            // all refrenced variables can be shared because of implicit enclosing taskgroup
            const auto & func = m_func;
  #         pragma omp taskloop default(none) shared(pec,num_tasks,func,N) num_tasks(num_tasks)
            for(uint64_t i=0;i<N;i++)
            {
              func( i );
            }
          }
          // here all tasks of taskloop have completed, since notaskgroup clause is not specified              
          execute_epilog( pec , pes );          
          pec->m_total_cpu_execution_time = ( std::chrono::high_resolution_clock::now() - T0 ).count() / 1000000.0;
          if( pec->m_execution_end_callback.m_func != nullptr )
          {
            (* pec->m_execution_end_callback.m_func) ( pec->m_execution_end_callback.m_data );
          }
          pes->m_omp_execution_count.fetch_sub(1);
        }
      }

      inline void execute_epilog( ParallelExecutionContext* pec, ParallelExecutionStream* pes ) const override final
      {
        if constexpr (functor_has_cpu_epilog) { m_func(block_parallel_for_cpu_epilog_t{}); }
        else if constexpr (functor_has_epilog) { m_func(block_parallel_for_epilog_t{}); }
      }

      // ================ CPU individual task execution interface ======================
      inline void operator () (uint64_t i) const override final { m_func(i); }
      inline void operator () (uint64_t i, uint64_t end) const override final { for(;i<end;i++) m_func(i); }
      inline void operator () (const uint64_t* __restrict__ idx, uint64_t N) const override final { for(uint64_t i=0;i<N;i++) m_func(idx[i]); }
      inline ~BlockParallelForHostAdapter() override {}
    };


  }
}

