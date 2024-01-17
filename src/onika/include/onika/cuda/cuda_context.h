#pragma once

template<class... AnyArgs> static inline constexpr int _fake_cuda_api_noop(AnyArgs...){return 0;}

#ifdef ONIKA_CUDA_VERSION

#ifdef ONIKA_HIP_VERSION
#include <hip_runtime.h>
#define ONIKA_CU_PROF_RANGE_PUSH            _fake_cuda_api_noop
#define ONIKA_CU_PROF_RANGE_POP             _fake_cuda_api_noop
#define ONIKA_CU_MEM_PREFETCH(ptr,sz,d,st) hipMemPrefetchAsync((const void*)(ptr),sz,0,st) 
#define ONIKA_CU_CREATE_STREAM_NON_BLOCKING(streamref) hipStreamCreateWithFlags( & streamref, hipStreamNonBlocking )
using onikaDeviceProp = hipDeviceProp;
using onikaStream_t = hipStream_t;
using onikaEvent_t = hipEvent_t;
using onikaError_t = hipError_t;
#else
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#define ONIKA_CU_PROF_RANGE_PUSH(s) nvtxRangePush(s)
#define ONIKA_CU_PROF_RANGE_POP() nvtxRangePop()
#define ONIKA_CU_MEM_PREFETCH(ptr,sz,d,st) cudaMemPrefetchAsync((const void*)(ptr),sz,0,st) 
#define ONIKA_CU_CREATE_STREAM_NON_BLOCKING(streamref) cudaStreamCreateWithFlags( & streamref, cudaStreamNonBlocking )
using onikaDeviceProp = cudaDeviceProp;
using onikaStream_t = cudaStream_t;
using onikaEvent_t = cudaEvent_t;
using onikaError_t = cudaError_t;
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
static inline constexpr int onikaErrorNotReady = 0;

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


