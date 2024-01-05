#pragma once

#include <onika/parallel/block_parallel_for_functor.h>

namespace onika
{
  namespace parallel
  {

    template< class FuncT>
    ONIKA_DEVICE_KERNEL_FUNC
    [[maybe_unused]] static
    void gpu_functor_initialize( __grid_constant__ const FuncT func )
    {
      static constexpr bool functor_has_gpu_prolog = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_gpu_prolog_t>;
      static constexpr bool functor_has_prolog     = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_prolog_t>;
      assert( ONIKA_CU_GRID_SIZE == 1 && ONIKA_CU_BLOCK_IDX == 0 );
      if constexpr ( functor_has_gpu_prolog ) { func( block_parallel_for_gpu_prolog_t{} ); }
      else if constexpr ( functor_has_prolog ) { func( block_parallel_for_prolog_t{} ); }
    }


    template< class FuncT>
    ONIKA_DEVICE_KERNEL_FUNC
    [[maybe_unused]] static
    void gpu_functor_finalize( __grid_constant__ const FuncT func )
    {
      static constexpr bool functor_has_gpu_epilog = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_gpu_epilog_t>;
      static constexpr bool functor_has_epilog     = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_epilog_t>;
      assert( ONIKA_CU_GRID_SIZE == 1 && ONIKA_CU_BLOCK_IDX == 0 );
      if constexpr ( functor_has_gpu_epilog ) { func( block_parallel_for_gpu_epilog_t{} ); }
      else if constexpr ( functor_has_epilog ) { func( block_parallel_for_epilog_t{} ); }
    }

    // GPU execution kernel for fixed size grid, using workstealing element assignment to blocks
    template< class FuncT>
    ONIKA_DEVICE_KERNEL_FUNC
    ONIKA_DEVICE_KERNEL_BOUNDS(ONIKA_CU_MAX_THREADS_PER_BLOCK,ONIKA_CU_MIN_BLOCKS_PER_SM)
    [[maybe_unused]] static
    void block_parallel_for_gpu_kernel_workstealing( uint64_t N, GPUKernelExecutionScratch* scratch, __grid_constant__ const FuncT func )
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
    void block_parallel_for_gpu_kernel_regulargrid( __grid_constant__ const FuncT func )
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
      inline void operator () (block_parallel_for_prolog_t) const override final
      {
        if constexpr (functor_has_cpu_prolog) { m_func(block_parallel_for_cpu_prolog_t{}); }
        else if constexpr (functor_has_prolog) { m_func(block_parallel_for_prolog_t{}); }
      }
      inline void operator () (block_parallel_for_epilog_t) const override final
      {
        if constexpr (functor_has_cpu_epilog) { m_func(block_parallel_for_cpu_epilog_t{}); }
        else if constexpr (functor_has_epilog) { m_func(block_parallel_for_epilog_t{}); }
      }
      inline void stream_gpu_initialize(ParallelExecutionContext* pec , ParallelExecutionStream* pes) const override final
      {
        if constexpr ( GPUSupport )
        {
          printf("stream GPU initialization (%s)\n",pec->m_tag);
          ONIKA_CU_LAUNCH_KERNEL(1,pec->m_block_size,0,pes->m_cu_stream,gpu_functor_initialize,m_func);
        }
        else { std::cerr << "called stream_gpu_initialize with no GPU support" << std::endl; std::abort(); }
      }
      
      inline void stream_gpu_kernel(ParallelExecutionContext* pec, ParallelExecutionStream* pes) const override final
      {
        if constexpr ( GPUSupport )
        {
          assert( pec->m_parallel_space.m_start == 0 && pec->m_parallel_space.m_idx == nullptr );
          const size_t N = pec->m_parallel_space.m_end;
          printf("stream GPU Kernel (%s) N=%d\n",pec->m_tag,int(N));
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
          printf("stream functor initialization (%s)\n",pec->m_tag);
          ONIKA_CU_LAUNCH_KERNEL(1,pec->m_block_size,0,pes->m_cu_stream,gpu_functor_finalize,m_func);
        }
        else { std::cerr << "called stream_gpu_finalize with no GPU support" << std::endl; std::abort(); }
      }
      
      inline void operator () (uint64_t i) const override final { m_func(i); }
      inline void operator () (uint64_t i, uint64_t end) const override final { for(;i<end;i++) m_func(i); }
      inline void operator () (const uint64_t* __restrict__ idx, uint64_t N) const override final { for(uint64_t i=0;i<N;i++) m_func(idx[i]); }
      inline ~BlockParallelForHostAdapter() override final {}
    };


  }
}

