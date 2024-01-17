#pragma once

template<class... AnyArgs> static inline constexpr int _fake_cuda_api_noop(AnyArgs...){return 0;}

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
#define ONIKA_CU_EVENT_QUERY(evt) hipEventQuery(evt)
#define ONIKA_CU_MALLOC(devPtrPtr,N) hipMalloc(devPtrPtr,N)
#define ONIKA_CU_MALLOC_MANAGED(devPtrPtr,N) hipMallocManaged(devPtrPtr,N)
#define ONIKA_CU_FREE(devPtr) hipFree(devPtr)
#define ONIKA_CU_GET_ERROR_STRING(c) hipGetErrorString(code)
#define ONIKA_CU_NAME_STR "HIP"
using onikaDeviceProp = hipDeviceProp_t;
using onikaStream_t = hipStream_t;
using onikaEvent_t = hipEvent_t;
using onikaError_t = hipError_t;
static inline constexpr auto onikaSuccess = hipSuccess;

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
#define ONIKA_CU_EVENT_QUERY(evt) cudaEventQuery(evt)
#define ONIKA_CU_MALLOC(devPtrPtr,N) cudaMalloc(devPtrPtr,N)
#define ONIKA_CU_MALLOC_MANAGED(devPtrPtr,N) cudaMallocManaged(devPtrPtr,N)
#define ONIKA_CU_FREE(devPtr) cudaFree(devPtr)
#define ONIKA_CU_GET_ERROR_STRING(c) cudaGetErrorString(code)
#define ONIKA_CU_NAME_STR "Cuda"
using onikaDeviceProp = cudaDeviceProp;
using onikaStream_t = cudaStream_t;
using onikaEvent_t = cudaEvent_t;
using onikaError_t = cudaError_t;
static inline constexpr auto onikaSuccess = cudaSuccess;

#endif

#else

struct onikaDeviceProp
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
    template<> struct MemoryUsage<onikaDeviceProp>
    {
      static inline constexpr size_t memory_bytes(const onikaDeviceProp&) { return sizeof(onikaDeviceProp); }
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
      onikaDeviceProp m_deviceProp;
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


