#pragma once

#include <onika/cuda/cuda.h>
#include <onika/parallel/parallel_execution_context.h>

namespace onika
{
  namespace parallel
  {

    class BlockParallelForGPUFunctor
    {
    public:
      ONIKA_DEVICE_FUNC virtual inline void operator () (block_parallel_for_prolog_t) const {}
      ONIKA_DEVICE_FUNC virtual inline void operator () (block_parallel_for_epilog_t) const {}
      ONIKA_DEVICE_FUNC virtual inline void operator () (size_t i) const {}
      ONIKA_DEVICE_FUNC virtual inline void operator () (size_t i, size_t end) const { for(;i<end;i++) this->operator () (i); }
      ONIKA_DEVICE_FUNC virtual inline void operator () (const size_t* __restrict__ idx, size_t N) const { for(size_t i=0;i<N;i++) this->operator () (idx[i]); }
      ONIKA_DEVICE_FUNC virtual inline ~BlockParallelForGPUFunctor() {}
    };

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
        if constexpr (functor_has_gpu_prolog) { m_func(block_parallel_for_gpu_prolog_t{}); }
        else if constexpr (functor_has_prolog) { m_func(block_parallel_for_prolog_t{}); }
      }
      ONIKA_DEVICE_FUNC inline void operator () (block_parallel_for_epilog_t) const override final
      {
        if constexpr (functor_has_gpu_epilog) { m_func(block_parallel_for_gpu_epilog_t{}); }
        else if constexpr (functor_has_epilog) { m_func(block_parallel_for_epilog_t{}); }
      }
      ONIKA_DEVICE_FUNC inline void operator () (size_t i) const override final { m_func(i); }
      ONIKA_DEVICE_FUNC inline void operator () (size_t i, size_t end) const override final { for(;i<end;i++) m_func(i); }
      ONIKA_DEVICE_FUNC inline void operator () (const size_t* __restrict__ idx, size_t N) const override final { for(size_t i=0;i<N;i++) m_func(idx[i]); }
      ONIKA_DEVICE_FUNC inline ~BlockParallelForGPUAdapter() override final {}
    };

    template<long long FunctorSize, long long MaxSize, class FuncT>
    struct AssertFunctorSizeFitIn
    {
      std::enable_if_t< (FunctorSize <= MaxSize) , int > x = 0;
    };

    template< class FuncT>
    ONIKA_DEVICE_KERNEL_FUNC
    [[maybe_unused]]
    static void initialize_functor_adapter( const FuncT func , GPUKernelExecutionScratch* scratch )
    {
      [[maybe_unused]] static constexpr AssertFunctorSizeFitIn< sizeof(BlockParallelForGPUAdapter<FuncT>) , GPUKernelExecutionScratch::MAX_FUNCTOR_SIZE , FuncT > _check_functor_size = {};
      if( ONIKA_CU_THREAD_IDX == 0 && ONIKA_CU_BLOCK_IDX == 0 )
      {
        BlockParallelForGPUFunctor* func_adapter = new(scratch->functor_data) BlockParallelForGPUAdapter<FuncT>( func );
        (*func_adapter) ( /* what ParallelExecutionContext ? it is not GPU accessible*/ ... );
        assert( (void*)scratch->functor_data == (void*)func_adapter );
      }
    }

    ONIKA_DEVICE_KERNEL_FUNC
    void finalize_functor_adapter( GPUKernelExecutionScratch* scratch );

    class BlockParallelForHostFunctor
    {
    public:
      virtual inline void operator () (block_parallel_for_prolog_t) const {}
      virtual inline void operator () (block_parallel_for_epilog_t) const {}
      virtual inline void stream_init_gpu_functor(ParallelExecutionContext* pec , ParrallelExecutionStream* pes) const {}
      virtual inline void operator () (size_t i) const {}
      virtual inline void operator () (size_t i, size_t end) const { for(;i<end;i++) this->operator () (i); }
      virtual inline void operator () (const size_t* __restrict__ idx, size_t N) const { for(size_t i=0;i<N;i++) this->operator () (idx[i]); }
      virtual ~BlockParallelForHostFunctor() {}
    };

    template<class FuncT>
    class BlockParallelForHostAdapter : public BlockParallelForHostFunctor
    {
      static inline constexpr bool functor_has_prolog     = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_prolog_t>;
      static inline constexpr bool functor_has_cpu_prolog = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_cpu_prolog_t>;
      static inline constexpr bool functor_has_epilog     = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_epilog_t>;
      static inline constexpr bool functor_has_cpu_epilog = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_cpu_epilog_t>;
      const FuncT m_func;
    public:
      inline BlockParallelForGPUAdapter( const FuncT& f ) : m_func(f) {}
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
      inline void stream_init_gpu_functor(ParallelExecutionContext* pec , ParrallelExecutionStream* pes) const override final
      {
        ONIKA_CU_LAUNCH_KERNEL(1,1,0,pes->m_cu_stream,initialize_functor_adapter,m_func,pec->m_cuda_scratch.get());        
      }      
      inline void operator () (size_t i) const override final { m_func(i); }
      inline void operator () (size_t i, size_t end) const override final { for(;i<end;i++) m_func(i); }
      inline void operator () (const size_t* __restrict__ idx, size_t N) const override final { for(size_t i=0;i<N;i++) m_func(idx[i]); }
      inline ~BlockParallelForHostAdapter() override final {}
    };

  }
}

