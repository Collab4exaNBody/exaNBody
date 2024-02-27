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

#include <onika/parallel/block_parallel_for.h>

namespace onika
{

  namespace parallel
  {  

    template<class FuncT> struct ParallelForFunctorTraits
    {      
      static inline constexpr bool CudaCompatible = false;
    };

    /*
     * ParallelForOptions holds options passed to block_parallel_for
     */
    struct ParallelForOptions
    {
      ParallelExecutionCallback user_cb = {};
      void * return_data = nullptr;
      size_t return_data_size = 0;
      bool enable_gpu = true;
      OMPScheduling omp_scheduling = OMP_SCHED_STATIC;
    };

    template<class FuncT>
    struct ParallelForBlockAdapter
    {
      FuncT m_func;
      size_t N = 0;
      ONIKA_HOST_DEVICE_FUNC inline void operator () ( size_t block_idx ) const
      {
        size_t i = block_idx * ONIKA_CU_BLOCK_SIZE + ONIKA_CU_THREAD_IDX;
        if( i < N ) m_func( i );
      }
    };

    template< class FuncT >
    static inline
    ParallelExecutionWrapper
    parallel_for(
        uint64_t N
      , const FuncT& func
      , ParallelExecutionContext * pec
      , const ParallelForOptions& opts = ParallelForOptions{} )
    {          
      BlockParallelForOptions bpfopts = {};
      bpfopts.user_cb = opts.user_cb;
      bpfopts.return_data = opts.return_data;
      bpfopts.return_data_size = opts.return_data_size;
      bpfopts.enable_gpu = opts.enable_gpu;
      bpfopts.omp_scheduling = opts.omp_scheduling;
      bpfopts.n_div_blocksize = true;
      
      return block_parallel_for( N , ParallelForBlockAdapter<FuncT>{func,N} , pec , bpfopts );
    }

  }

}

