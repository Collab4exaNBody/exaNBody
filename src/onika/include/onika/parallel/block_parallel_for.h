/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/
#pragma once

#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_error.h>
#include <onika/cuda/device_storage.h>
#include <onika/soatl/field_id.h>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/parallel_execution_stream.h>
#include <onika/parallel/block_parallel_for_adapter.h>
#include <onika/lambda_tools.h>
#include <onika/stream_utils.h>

namespace onika
{

  namespace parallel
  {  
    // this template is here to know if compute buffer must be built or computation must be ran on the fly
    template<class FuncT> struct BlockParallelForFunctorTraits
    {      
      static inline constexpr bool CudaCompatible = false;
    };

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
      using HostFunctorAdapter = BlockParallelForHostAdapter< FuncT , BlockParallelForFunctorTraits<FuncT>::CudaCompatible >;
      [[maybe_unused]] static constexpr AssertFunctorSizeFitIn< sizeof(HostFunctorAdapter) , HostKernelExecutionScratch::MAX_FUNCTOR_SIZE , FuncT > _check_cpu_functor_size = {};
      assert( pec != nullptr );
    
      const auto [ user_cb
                 , return_data 
                 , return_data_size 
                 , enable_gpu
                 , fixed_gpu_grid_size
                 ] = opts;

      // construct virtual functor adapter inplace, using reserved functor space
      static_assert( ( HostKernelExecutionScratch::MAX_FUNCTOR_ALIGNMENT % alignof(FuncT) ) == 0 , "functor_data alignment is not sufficient for user functor" );
      // printf("inplace construction of host adapter, size=%d, max=%d\n", int(sizeof(HostFunctorAdapter)) , int(HostKernelExecutionScratch::MAX_FUNCTOR_SIZE) );
      new(pec->m_host_scratch.functor_data) HostFunctorAdapter( func );

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
	    // printf("bpf: return data input=%p , output=%p , size=%d\n",return_data,return_data, int(return_data_size) );
            pec->set_return_data_input( return_data , return_data_size );
            pec->set_return_data_output( return_data , return_data_size );
          }
          else
          {
            pec->set_return_data_input( nullptr , 0 );
            pec->set_return_data_output( nullptr , 0 );
          }
          return {pec};
        }
      }

      // ================== CPU / OpenMP execution path ====================
      pec->m_execution_target = ParallelExecutionContext::EXECUTION_TARGET_OPENMP;
      return {pec};

    }
    
  }

}

