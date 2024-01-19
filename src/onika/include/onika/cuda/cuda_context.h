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

template<class... AnyArgs> static inline constexpr int _fake_cuda_api_noop(AnyArgs...){return 0;}


/***************************************************************/
/************************ Cuda API calls ***********************/
/***************************************************************/

#ifdef ONIKA_CUDA_VERSION

#ifdef ONIKA_HIP_VERSION

// HIP runtime API
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#define ONIKA_CU_PROF_RANGE_PUSH            _fake_cuda_api_noop
#define ONIKA_CU_PROF_RANGE_POP             _fake_cuda_api_noop
#define ONIKA_CU_MEM_PREFETCH(ptr,sz,d,st) hipMemPrefetchAsync((const void*)(ptr),sz,0,st) 
#define ONIKA_CU_CREATE_STREAM_NON_BLOCKING(streamref) hipStreamCreateWithFlags( & streamref, hipStreamNonBlocking )
#define ONIKA_CU_STREAM_ADD_CALLBACK(stream,cb,udata) hipStreamAddCallback(stream,cb,udata,0u)
#define ONIKA_CU_STREAM_SYNCHRONIZE(STREAM)  hipStreamSynchronize(STREAM)
#define ONIKA_CU_DESTROY_STREAM(streamref) hipStreamDestroy(streamref)
#define ONIKA_CU_EVENT_QUERY(evt) hipEventQuery(evt)
#define ONIKA_CU_MALLOC(devPtrPtr,N) hipMalloc(devPtrPtr,N)
#define ONIKA_CU_MALLOC_MANAGED(devPtrPtr,N) hipMallocManaged(devPtrPtr,N)
#define ONIKA_CU_FREE(devPtr) hipFree(devPtr)
#define ONIKA_CU_CREATE_EVENT(EVT) hipEventCreate(&EVT)
#define ONIKA_CU_DESTROY_EVENT(EVT) hipEventDestroy(EVT)
#define ONIKA_CU_STREAM_EVENT(EVT,STREAM) hipEventRecord(EVT,STREAM)
#define ONIKA_CU_EVENT_ELAPSED(T,EVT1,EVT2) hipEventElapsedTime(&T,EVT1,EVT2)
#define ONIKA_CU_MEMSET(p,v,n,...) hipMemsetAsync(p,v,n OPT_COMMA_VA_ARGS(__VA_ARGS__) )
#define ONIKA_CU_MEMCPY(d,s,n,...) hipMemcpyAsync(d,s,n,hipMemcpyDefault OPT_COMMA_VA_ARGS(__VA_ARGS__) )
#define ONIKA_CU_MEMCPY_KIND(d,s,n,k,...) hipMemcpyAsync(d,s,n,k OPT_COMMA_VA_ARGS(__VA_ARGS__) )
#define ONIKA_CU_GET_DEVICE_COUNT(iPtr)	hipGetDeviceCount(iPtr)
#define ONIKA_CU_SET_DEVICE(id)	hipSetDevice(id)
#define ONIKA_CU_SET_SHARED_MEM_CONFIG(shmc) hipDeviceSetSharedMemConfig(shmc)
#define ONIKA_CU_SET_LIMIT(l,v)	hipDeviceSetLimit(l,v)
#define ONIKA_CU_GET_LIMIT(vptr,l) hipDeviceGetLimit(vptr,l)
#define ONIKA_CU_GET_DEVICE_PROPERTIES(propPtr,id) hipGetDeviceProperties(propPtr,id)
#define ONIKA_CU_DEVICE_SYNCHRONIZE() hipDeviceSynchronize()
#define ONIKA_CU_GET_ERROR_STRING(c) hipGetErrorString(code)
#define ONIKA_CU_NAME_STR "HIP "
using onikaDeviceProp_t = hipDeviceProp_t;
using onikaStream_t = hipStream_t;
using onikaEvent_t = hipEvent_t;
using onikaError_t = hipError_t;
using onikaLimit_t = hipLimit_t;
static inline constexpr auto onikaSuccess = hipSuccess;
static inline constexpr auto onikaSharedMemBankSizeFourByte = hipSharedMemBankSizeFourByte;
static inline constexpr auto onikaSharedMemBankSizeEightByte = hipSharedMemBankSizeEightByte;
static inline constexpr auto onikaSharedMemBankSizeDefault = hipSharedMemBankSizeDefault;
static inline constexpr auto onikaLimitStackSize = hipLimitStackSize;
static inline constexpr auto onikaLimitPrintfFifoSize = hipLimitPrintfFifoSize;
static inline constexpr auto onikaLimitMallocHeapSize = hipLimitMallocHeapSize;
static inline constexpr auto onikaMemcpyDeviceToHost = hipMemcpyDeviceToHost;
static inline constexpr auto onikaMemcpyHostToDevice = hipMemcpyHostToDevice;

#else

// Cuda runtime API
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>
#define ONIKA_CU_PROF_RANGE_PUSH(s) nvtxRangePush(s)
#define ONIKA_CU_PROF_RANGE_POP() nvtxRangePop()
#define ONIKA_CU_MEM_PREFETCH(ptr,sz,d,st) cudaMemPrefetchAsync((const void*)(ptr),sz,0,st) 
#define ONIKA_CU_CREATE_STREAM_NON_BLOCKING(streamref) cudaStreamCreateWithFlags( & streamref, cudaStreamNonBlocking )
#define ONIKA_CU_STREAM_ADD_CALLBACK(stream,cb,udata) cudaStreamAddCallback(stream,cb,udata,0u)
#define ONIKA_CU_STREAM_SYNCHRONIZE(STREAM)  cudaStreamSynchronize(STREAM)
#define ONIKA_CU_DESTROY_STREAM(streamref) cudaStreamDestroy(streamref)
#define ONIKA_CU_EVENT_QUERY(evt) cudaEventQuery(evt)
#define ONIKA_CU_MALLOC(devPtrPtr,N) cudaMalloc(devPtrPtr,N)
#define ONIKA_CU_MALLOC_MANAGED(devPtrPtr,N) cudaMallocManaged(devPtrPtr,N)
#define ONIKA_CU_FREE(devPtr) cudaFree(devPtr)
#define ONIKA_CU_CREATE_EVENT(EVT) cudaEventCreate(&EVT)
#define ONIKA_CU_DESTROY_EVENT(EVT) cudaEventDestroy(EVT)
#define ONIKA_CU_STREAM_EVENT(EVT,STREAM) cudaEventRecord(EVT,STREAM)
#define ONIKA_CU_EVENT_ELAPSED(T,EVT1,EVT2) cudaEventElapsedTime(&T,EVT1,EVT2)
#define ONIKA_CU_MEMSET(p,v,n,...) cudaMemsetAsync(p,v,n OPT_COMMA_VA_ARGS(__VA_ARGS__) )
#define ONIKA_CU_MEMCPY(d,s,n,...) cudaMemcpyAsync(d,s,n,cudaMemcpyDefault OPT_COMMA_VA_ARGS(__VA_ARGS__) )
#define ONIKA_CU_MEMCPY_KIND(d,s,n,k,...) cudaMemcpyAsync(d,s,n,k OPT_COMMA_VA_ARGS(__VA_ARGS__) )
#define ONIKA_CU_GET_DEVICE_COUNT(iPtr)	cudaGetDeviceCount(iPtr)
#define ONIKA_CU_SET_DEVICE(id)	cudaSetDevice(id)
#define ONIKA_CU_SET_SHARED_MEM_CONFIG(shmc) cudaDeviceSetSharedMemConfig(shmc)
#define ONIKA_CU_SET_LIMIT(l,v)	cudaDeviceSetLimit(l,v)
#define ONIKA_CU_GET_LIMIT(vptr,l) cudaDeviceGetLimit(vptr,l)
#define ONIKA_CU_GET_DEVICE_PROPERTIES(propPtr,id) cudaGetDeviceProperties(propPtr,id)
#define ONIKA_CU_DEVICE_SYNCHRONIZE() cudaDeviceSynchronize()
#define ONIKA_CU_GET_ERROR_STRING(c) cudaGetErrorString(code)
#define ONIKA_CU_NAME_STR "Cuda"
using onikaDeviceProp_t = cudaDeviceProp;
using onikaStream_t = cudaStream_t;
using onikaEvent_t = cudaEvent_t;
using onikaError_t = cudaError_t;
using onikaLimit_t = cudaLimit;
static inline constexpr auto onikaSuccess = cudaSuccess;
static inline constexpr auto onikaSharedMemBankSizeFourByte = cudaSharedMemBankSizeFourByte;
static inline constexpr auto onikaSharedMemBankSizeEightByte = cudaSharedMemBankSizeEightByte;
static inline constexpr auto onikaSharedMemBankSizeDefault = cudaSharedMemBankSizeDefault;
static inline constexpr auto onikaLimitStackSize = cudaLimitStackSize;
static inline constexpr auto onikaLimitPrintfFifoSize = cudaLimitPrintfFifoSize;
static inline constexpr auto onikaLimitMallocHeapSize = cudaLimitMallocHeapSize;
static inline constexpr auto onikaMemcpyDeviceToHost = cudaMemcpyDeviceToHost;
static inline constexpr auto onikaMemcpyHostToDevice = cudaMemcpyHostToDevice;

#endif

#else

struct onikaDeviceProp_t
{
  char name[256];
  int managedMemory = 0;
  int concurrentManagedAccess = 0;
  int totalGlobalMem = 0;
  int warpSize = 0;
  int multiProcessorCount = 0;
  int sharedMemPerBlock = 0;
};
using onikaStream_t = int;
using onikaEvent_t = int*;
using onikaError_t = int;
static inline constexpr int onikaSuccess = 0;
static inline constexpr int onikaErrorNotReady = 0;

#define cudaEventQuery _fake_cuda_api_noop
#define cudaStreamAddCallback _fake_cuda_api_noop
#define cudaStreamCreate _fake_cuda_api_noop

#define ONIKA_CU_PROF_RANGE_PUSH            _fake_cuda_api_noop
#define ONIKA_CU_PROF_RANGE_POP             _fake_cuda_api_noop
#define ONIKA_CU_MEM_PREFETCH               _fake_cuda_api_noop
#define ONIKA_CU_CREATE_STREAM_NON_BLOCKING _fake_cuda_api_noop
#define ONIKA_CU_STREAM_ADD_CALLBACK        _fake_cuda_api_noop
#define ONIKA_CU_CREATE_EVENT(EVT)          _fake_cuda_api_noop(EVT=nullptr)
#define ONIKA_CU_DESTROY_EVENT(EVT)         _fake_cuda_api_noop(EVT=nullptr)
#define ONIKA_CU_STREAM_EVENT(EVT,STREAM)   _fake_cuda_api_noop(EVT,STREAM)
#define ONIKA_CU_EVENT_ELAPSED(T,EVT1,EVT2) _fake_cuda_api_noop(T=0.0f)
#define ONIKA_CU_STREAM_SYNCHRONIZE(STREAM) _fake_cuda_api_noop(STREAM)
#define ONIKA_CU_EVENT_QUERY(EVT)           (onikaSuccess)
#define ONIKA_CU_MEMSET(p,v,n,...)          std::memset(p,v,n)
#define ONIKA_CU_MEMCPY(d,s,n,...)          std::memcpy(d,s,n)
#define ONIKA_CU_NAME_STR "GPU "
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
    template<> struct MemoryUsage<onikaDeviceProp_t>
    {
      static inline constexpr size_t memory_bytes(const onikaDeviceProp_t&) { return sizeof(onikaDeviceProp_t); }
    };
    template<> struct MemoryUsage<onikaStream_t>
    {
      static inline constexpr size_t memory_bytes(const onikaStream_t&) { return sizeof(onikaStream_t); }
    };
  }

  namespace cuda
  {

    struct CudaDevice
    {
      onikaDeviceProp_t m_deviceProp;
      std::list< std::function<void()> > m_finalize_destructors;
      int device_id = 0;
    };

    struct CudaContext
    {
      std::vector<CudaDevice> m_devices;
      std::vector<onikaStream_t> m_threadStream;

      bool has_devices() const;
      unsigned int device_count() const;
      onikaStream_t getThreadStream(unsigned int tid);
    };

  }

}


