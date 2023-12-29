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
      ONIKA_DEVICE_FUNC virtual void operator () (size_t i) const =0;
      ONIKA_DEVICE_FUNC virtual inline void operator () (size_t start, size_t end) const
      {
        for(;start<end;start++) this->operator () (start);
      }
      ONIKA_DEVICE_FUNC virtual inline void operator () (const size_t* __restrict__ indices, size_t count) const
      {
        for(size_t i=0;i<count;i++) this->operator () (indices[i]);
      }
      ONIKA_DEVICE_FUNC virtual inline ~BlockParallelForGPUFunctor() {}
    };

    template<class FuncT>
    class BlockParallelForGPUAdapter : public BlockParallelForGPUFunctor
    {
      const FuncT m_func;
    public:
      ONIKA_DEVICE_FUNC inline BlockParallelForGPUAdapter( const FuncT& f ) : m_func(f) {}
      ONIKA_DEVICE_FUNC inline void operator () (size_t i) const override final
      {
        m_func(i);
      }
      ONIKA_DEVICE_FUNC inline void operator () (size_t start, size_t end) const override final
      {
        for(;start<end;start++) m_func(start);
      }
      ONIKA_DEVICE_FUNC inline void operator () (const size_t* __restrict__ indices, size_t count) const override final
      {
        for(size_t i=0;i<count;i++) m_func(indices[i]);
      }
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
        assert( (void*)scratch->functor_data == (void*)func_adapter );
      }
    }

    ONIKA_DEVICE_KERNEL_FUNC
    void finalize_functor_adapter( GPUKernelExecutionScratch* scratch );

    class BlockParallelForHostFunctor
    {
    public:
      virtual void to_stream(size_t N, ParallelExecutionContext* pec , ParrallelExecutionStream* pes) const =0;
      virtual void operator () (size_t i) const =0;
      virtual inline void operator () (size_t start, size_t end) const
      {
        for(;start<end;start++) this->operator () (start);
      }
      virtual inline void operator () (const size_t* __restrict__ indices, size_t count) const
      {
        for(size_t i=0;i<count;i++) this->operator () (indices[i]);
      }
      virtual ~BlockParallelForHostFunctor() {}
    };

    template<class FuncT>
    class BlockParallelForHostAdapter : public BlockParallelForHostFunctor
    {
      const FuncT m_func;
    public:
      inline BlockParallelForGPUAdapter( const FuncT& f ) : m_func(f) {}
      // this version will work for stream based execution, another version will be needed for graph based execution
      inline void to_stream(size_t N, ParallelExecutionContext* pec , ParrallelExecutionStream* pes) const override final
      {
        switch( pec->m_execution_target )
        {
          case EXECUTION_TARGET_OPENMP :
          {
          }
          break;
          case EXECUTION_TARGET_CUDA :
          {
            checkCudaErrors( ONIKA_CU_STREAM_EVENT( pec-m_start_evt, pes->m_cu_stream ) );
            if( pec->m_return_data_input != nullptr && pec->m_return_data_size > 0 )
            {
              checkCudaErrors( ONIKA_CU_MEMCPY( pec->m_cuda_scratch->return_data, pec->m_return_data_input , pec->m_return_data_size , pes->m_cu_stream ) );
            }
            ONIKA_CU_LAUNCH_KERNEL(1,1,0,pes->m_cu_stream,initialize_functor_adapter,m_func,pec->m_cuda_scratch.get());
            if( pec->m_grid_size > 0 )
            {
              ONIKA_CU_LAUNCH_KERNEL(pec->m_grid_size,pec->m_block_size,0,pes->m_cu_stream, block_parallel_for_gpu_kernel_workstealing, N, pec->m_cuda_scratch.get() );
            }
            else
            {
              ONIKA_CU_LAUNCH_KERNEL(N,pec->m_block_size,0,pes->m_cu_stream, block_parallel_for_gpu_kernel_regulargrid, pec->m_cuda_scratch.get() );
            }
            ONIKA_CU_LAUNCH_KERNEL(1,1,0,pes->m_cu_stream,finalize_functor_adapter,pec->m_cuda_scratch.get());
            if( pec->m_return_data_output != nullptr && pec->m_return_data_size > 0 )
            {
              checkCudaErrors( ONIKA_CU_MEMCPY( pec->m_return_data_output , pec->m_cuda_scratch->return_data , pec->m_return_data_size , pes->m_cu_stream ) );
            }
            checkCudaErrors( ONIKA_CU_STREAM_EVENT( pec-m_stop_evt, pes->m_cu_stream ) );
            
            ... optional user end of execution callback .. must be called right after end of execution
            ... profiling information gather callback ... can be called at the synchronization point of the stream
          }
          break;          
        }
      }
      
      inline void operator () (size_t i) const override final
      {
        m_func(i);
      }
      
      inline void operator () (size_t start, size_t end) const override final
      {
        for(;start<end;start++) m_func(start);
      }
      
      inline void operator () (const size_t* __restrict__ indices, size_t count) const override final
      {
        for(size_t i=0;i<count;i++) m_func(indices[i]);
      }
      inline ~BlockParallelForHostAdapter() override final {}
    };

  }
}

