#pragma once

#include <onika/cuda/cuda.h>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/lambda_tools.h>

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

    class BlockParallelForGPUFunctor
    {
    public:
      ONIKA_DEVICE_FUNC virtual inline void operator () (block_parallel_for_prolog_t) const {}
      ONIKA_DEVICE_FUNC virtual inline void operator () (block_parallel_for_epilog_t) const {}
      ONIKA_DEVICE_FUNC virtual inline void operator () (uint64_t i) const {}
      ONIKA_DEVICE_FUNC virtual inline void operator () (uint64_t i, uint64_t end) const { for(;i<end;i++) this->operator () (i); }
      ONIKA_DEVICE_FUNC virtual inline void operator () (const uint64_t* __restrict__ idx, size_t N) const { for(uint64_t i=0;i<N;i++) this->operator () (idx[i]); }
      ONIKA_DEVICE_FUNC virtual inline ~BlockParallelForGPUFunctor() {}
    };

    template<long long FunctorSize, long long MaxSize, class FuncT>
    struct AssertFunctorSizeFitIn
    {
      std::enable_if_t< (FunctorSize <= MaxSize) , int > x = 0;
    };

    class BlockParallelForHostFunctor
    {
    public:
      virtual inline void operator () (block_parallel_for_prolog_t) const {}
      virtual inline void operator () (block_parallel_for_epilog_t) const {}
      virtual inline void stream_gpu_initialize(ParallelExecutionContext*,ParallelExecutionStream*) const {}
      virtual inline void operator () (uint64_t i) const {}
      virtual inline void operator () (uint64_t i, uint64_t end) const { for(;i<end;i++) this->operator () (i); }
      virtual inline void operator () (const uint64_t * __restrict__ idx, uint64_t N) const { for(uint64_t i=0;i<N;i++) this->operator () (idx[i]); }
      virtual inline ~BlockParallelForHostFunctor() {}
    };

//    ONIKA_DEVICE_KERNEL_FUNC
//    void gpu_functor_finalize( GPUKernelExecutionScratch* scratch );

    ONIKA_DEVICE_KERNEL_FUNC
    [[maybe_unused]] static
    void gpu_functor_finalize( GPUKernelExecutionScratch* scratch )
    {
      assert( ONIKA_CU_GRID_SIZE == 1 && ONIKA_CU_BLOCK_IDX == 0 );
      const auto & func = * reinterpret_cast<BlockParallelForGPUFunctor*>( scratch->functor_data );
      //if( ONIKA_CU_THREAD_IDX == 0 ) printf("GPU: call epilog\n");
      func( block_parallel_for_epilog_t{} );
      //if( ONIKA_CU_THREAD_IDX == 0 ) printf("GPU: call epilog done\n");
      ONIKA_CU_BLOCK_SYNC();
      if( ONIKA_CU_THREAD_IDX == 0 )
      {
        reinterpret_cast<BlockParallelForGPUFunctor*>( scratch->functor_data ) -> ~BlockParallelForGPUFunctor();
      }
      //if( ONIKA_CU_THREAD_IDX == 0 ) printf("GPU: destructor done\n");      
    }


  }
}

