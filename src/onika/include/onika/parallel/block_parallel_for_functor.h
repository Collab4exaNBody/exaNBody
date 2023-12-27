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
      ONIKA_DEVICE_FUNC virtual inline void operator () (size_t i) const =0;
      ONIKA_DEVICE_FUNC virtual inline void operator () (size_t start, size_t end) const
      {
        for(;start<end;start++) this->operator () (start);
      }
      ONIKA_DEVICE_FUNC virtual inline void operator () (const size_t* __restrict__ indices, size_t count) const
      {
        for(size_t i=0;i<count;i++) this->operator () (indices[i]);
      }
      ONIKA_DEVICE_FUNC virtual ~BlockParallelForGPUFunctor() {}

    };

    template<class FuncT>
    class BlockParallelForGPUAdapter : public BlockParallelForGPUFunctor
    {
      const FuncT m_func;
    public:
      ONIKA_DEVICE_FUNC inline BlockParallelForGPUAdapter( const FuncT& f ) : m_func(f) {}
      ONIKA_DEVICE_FUNC virtual inline void operator () (size_t i) const { m_func(i); }
      ONIKA_DEVICE_FUNC virtual inline void operator () (size_t start, size_t end) const
      {
        for(;start<end;start++) m_func(start);
      }
      ONIKA_DEVICE_FUNC virtual inline void operator () (const size_t* __restrict__ indices, size_t count) const
      {
        for(size_t i=0;i<count;i++) m_func(indices[i]);
      }
      ONIKA_DEVICE_FUNC virtual ~BlockParallelForGPUAdapter() {}
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
        assert( (void*)scratch->functor_data == (void*)func_adapter );
      }
    }

    ONIKA_DEVICE_KERNEL_FUNC
    [[maybe_unused]]
    static void finalize_functor_adapter( GPUKernelExecutionScratch* scratch )
    {
      if( ONIKA_CU_THREAD_IDX == 0 && ONIKA_CU_BLOCK_IDX == 0 )
      {
        reinterpret_cast<BlockParallelForGPUFunctor*>( scratch->functor_data ) -> ~BlockParallelForGPUFunctor();
      }
    }


  }
}

