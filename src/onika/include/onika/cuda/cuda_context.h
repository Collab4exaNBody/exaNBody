#pragma once

#ifdef ONIKA_CUDA_VERSION

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#define ONIKA_CU_PROF_RANGE_PUSH(s) nvtxRangePush(s)
#define ONIKA_CU_PROF_RANGE_POP() nvtxRangePop()
#define ONIKA_CU_MEM_PREFETCH(ptr,sz,d,st) cudaMemPrefetchAsync((const void*)(ptr),sz,0,st) 

#else

#define ONIKA_CU_PROF_RANGE_PUSH(s) do{}while(false)
#define ONIKA_CU_PROF_RANGE_POP() do{}while(false)
#define ONIKA_CU_MEM_PREFETCH(ptr,sz,d,st) do{}while(false)

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
      inline bool has_devices() const { return ! m_devices.empty(); }
      inline unsigned int device_count() const { return m_devices.size(); }
      inline cudaStream_t getThreadStream(unsigned int tid)
      {
        if( tid >= m_threadStream.size() )
        {
          unsigned int i = m_threadStream.size();
          m_threadStream.resize( tid+1 , 0 );
          for(;i<m_threadStream.size();i++)
          {
            checkCudaErrors( cudaStreamCreateWithFlags( & m_threadStream[i], cudaStreamNonBlocking ) );
          }
        }
        return m_threadStream[tid];
      }
    };

  }

}


