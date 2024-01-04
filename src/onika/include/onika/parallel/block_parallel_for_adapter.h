#pragma once

#include <onika/parallel/block_parallel_for_functor.h>

namespace onika
{
  namespace parallel
  {

    template<class FuncT>
    class BlockParallelForGPUAdapter : public BlockParallelForGPUFunctor
    {
      static inline constexpr bool functor_has_prolog     = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_prolog_t>;
      static inline constexpr bool functor_has_gpu_prolog = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_gpu_prolog_t>;
      static inline constexpr bool functor_has_epilog     = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_epilog_t>;
      static inline constexpr bool functor_has_gpu_epilog = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_gpu_epilog_t>;
      const FuncT m_func;
    public:
      ONIKA_DEVICE_FUNC inline BlockParallelForGPUAdapter( const FuncT& f ) : m_func(f) {}
      ONIKA_DEVICE_FUNC inline void operator () (block_parallel_for_prolog_t) const override final
      {
        //if( ONIKA_CU_THREAD_IDX == 0 ) printf("GPU: functor prolog\n");
        if constexpr (functor_has_gpu_prolog) { m_func(block_parallel_for_gpu_prolog_t{}); }
        else if constexpr (functor_has_prolog) { m_func(block_parallel_for_prolog_t{}); }
        //if( ONIKA_CU_THREAD_IDX == 0 ) printf("GPU: functor prolog done\n");
      }
      ONIKA_DEVICE_FUNC inline void operator () (block_parallel_for_epilog_t) const override final
      {
        //if( ONIKA_CU_THREAD_IDX == 0 ) printf("GPU: functor epilog\n");
        if constexpr (functor_has_gpu_epilog) { m_func(block_parallel_for_gpu_epilog_t{}); }
        else if constexpr (functor_has_epilog) { m_func(block_parallel_for_epilog_t{}); }
        //if( ONIKA_CU_THREAD_IDX == 0 ) printf("GPU: functor epilog done\n");
      }
      ONIKA_DEVICE_FUNC inline void operator () (uint64_t i) const override final
      {
        //if( ONIKA_CU_THREAD_IDX == 0 ) printf("GPU: functor call i=%d\n",int(i));
        m_func(i);
      }
      ONIKA_DEVICE_FUNC inline void operator () (uint64_t i, uint64_t end) const override final { for(;i<end;i++) m_func(i); }
      ONIKA_DEVICE_FUNC inline void operator () (const uint64_t* __restrict__ idx, uint64_t N) const override final { for(uint64_t i=0;i<N;i++) m_func(idx[i]); }
      ONIKA_DEVICE_FUNC inline ~BlockParallelForGPUAdapter() override final {}
    };

    template< class FuncT>
    ONIKA_DEVICE_KERNEL_FUNC
    [[maybe_unused]] static
    void gpu_functor_initialize( const FuncT func , GPUKernelExecutionScratch* scratch )
    {
      assert( ONIKA_CU_GRID_SIZE == 1 && ONIKA_CU_BLOCK_IDX == 0 );
      ONIKA_CU_BLOCK_SHARED BlockParallelForGPUFunctor* func_adapter;
      if( ONIKA_CU_THREAD_IDX == 0 )
      {
        func_adapter = new(scratch->functor_data) BlockParallelForGPUAdapter<FuncT>( func );
        /*
        assert( (void*)scratch->functor_data == (void*)func_adapter );
        static constexpr int isz = sizeof(BlockParallelForGPUFunctor);
        static constexpr int sz = sizeof(BlockParallelForGPUAdapter<FuncT>);
        int i=0;
        printf("GPU: construct functor @%p , scratch @%p , size = %d / %d",func_adapter,scratch->functor_data,isz,sz);
        for(;i<isz;i++) printf("%c%02X" , ((i%16)==0) ? '\n' : ' ' , int(scratch->functor_data[i])&0xFF );
        printf(" / ");
        for(;i<sz;i++) printf("%c%02X" , ((i%16)==0) ? '\n' : ' ' , int(scratch->functor_data[i])&0xFF );
        printf("\n");
        */
      }
      ONIKA_CU_BLOCK_SYNC();
      (*func_adapter) ( block_parallel_for_prolog_t{} );
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
          //printf("stream functor initialization (%s)\n",pec->m_tag);
          ONIKA_CU_LAUNCH_KERNEL(1,pec->m_block_size,0,pes->m_cu_stream,gpu_functor_initialize,m_func,pec->m_cuda_scratch.get());
        }
        else
        {
          std::cerr << "Internal error: called stream_gpu_initialize with no GPU support" << std::endl;
          std::abort();
        }
      }
      inline void operator () (uint64_t i) const override final { m_func(i); }
      inline void operator () (uint64_t i, uint64_t end) const override final { for(;i<end;i++) m_func(i); }
      inline void operator () (const uint64_t* __restrict__ idx, uint64_t N) const override final { for(uint64_t i=0;i<N;i++) m_func(idx[i]); }
      inline ~BlockParallelForHostAdapter() override final {}
    };

  }
}

