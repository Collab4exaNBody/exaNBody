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

#ifdef ONIKA_CUDA_VERSION

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#define ONIKA_CU_PROF_RANGE_PUSH(s) nvtxRangePush(s)
#define ONIKA_CU_PROF_RANGE_POP() nvtxRangePop()
#define ONIKA_CU_MEM_PREFETCH(ptr,sz,d,st) cudaMemPrefetchAsync((const void*)(ptr),sz,0,st) 
#define ONIKA_CU_CREATE_STREAM_NON_BLOCKING(streamref) cudaStreamCreateWithFlags( & streamref, cudaStreamNonBlocking )

#else

struct cudaDeviceProp
{
  char name[256];
  int managedMemory = 0;
  int concurrentManagedAccess = 0;
  int totalGlobalMem = 0;
  int warpSize = 0;
  int multiProcessorCount = 0;
  int sharedMemPerBlock = 0;
};
using cudaStream_t = int;
using cudaEvent_t = int*;
using cudaError_t = int;
static inline constexpr int cudaSuccess = 0;
static inline constexpr int cudaStreamNonBlocking = 0;
static inline constexpr int cudaErrorNotReady = 0;
static inline constexpr int cudaStreamCreateWithFlags(cudaStream_t*,int){return cudaSuccess;}
template<class... AnyArgs> static inline constexpr int _fake_cuda_api_noop(AnyArgs...){return cudaSuccess;}

#define cudaEventQuery _fake_cuda_api_noop
#define cudaStreamAddCallback _fake_cuda_api_noop
#define cudaStreamCreate _fake_cuda_api_noop

#define ONIKA_CU_PROF_RANGE_PUSH            _fake_cuda_api_noop
#define ONIKA_CU_PROF_RANGE_POP             _fake_cuda_api_noop
#define ONIKA_CU_MEM_PREFETCH               _fake_cuda_api_noop
#define ONIKA_CU_CREATE_STREAM_NON_BLOCKING _fake_cuda_api_noop

#endif // ONIKA_CUDA_VERSION


#include <onika/memory/memory_usage.h>
#include <onika/cuda/cuda_error.h>
#include <functional>
#include <list>

// specializations to avoid MemoryUsage template to dig into cuda aggregates
namespace onika
{

  namespace memory
  {    
    template<> struct MemoryUsage<cudaDeviceProp>
    {
      static inline constexpr size_t memory_bytes(const cudaDeviceProp&) { return sizeof(cudaDeviceProp); }
    };
    template<> struct MemoryUsage<cudaStream_t>
    {
      static inline constexpr size_t memory_bytes(const cudaStream_t&) { return sizeof(cudaStream_t); }
    };
  }

  namespace cuda
  {

    struct CudaDevice
    {
      cudaDeviceProp m_deviceProp;
      std::list< std::function<void()> > m_finalize_destructors;
      int device_id = 0;
    };

    struct CudaContext
    {
      std::vector<CudaDevice> m_devices;
      std::vector<cudaStream_t> m_threadStream;

      bool has_devices() const;
      unsigned int device_count() const;
      cudaStream_t getThreadStream(unsigned int tid);
    };

  }

}


