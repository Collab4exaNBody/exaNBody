#pragma once

#include <type_traits>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <thread>
#include <onika/macro_utils.h>

#ifndef ONIKA_CU_MAX_THREADS_PER_BLOCK
#define ONIKA_CU_MAX_THREADS_PER_BLOCK 128
#endif

#ifndef ONIKA_CU_MIN_BLOCKS_PER_SM
#define ONIKA_CU_MIN_BLOCKS_PER_SM 6
#endif

#ifndef ONIKA_CU_ENABLE_KERNEL_BOUNDS
#define ONIKA_CU_ENABLE_KERNEL_BOUNDS 1
#endif

#define ONIKA_ALWAYS_INLINE inline __attribute__((always_inline))

namespace onika
{
  namespace cuda
  {

    struct onika_cu_atomic_flag_t
    {
      std::atomic_flag m_flag = ATOMIC_FLAG_INIT;
      uint8_t m_pad[3] = { 0, 0, 0 };
    };
    static_assert( sizeof(onika_cu_atomic_flag_t) == sizeof(uint32_t) , "Atomic flag type must be 4 bytes large" );
#   define ONIKA_CU_ATOMIC_FLAG_INIT ::onika::cuda::onika_cu_atomic_flag_t{ ATOMIC_FLAG_INIT , {0,0,0} }

    using onika_cu_memory_order_t = std::memory_order;


    /************** start of Cuda code definitions ***************/
#   ifdef __CUDA_ARCH__

    using gpu_device_execution_t = std::true_type;

    [[ noreturn ]] __host__ __device__ inline void __onika_cu_abort() { __threadfence(); __trap(); __builtin_unreachable(); }
# if __CUDA_ARCH__ < 600
#error atomicAdd(double) not available
# endif


#   define ONIKA_CU_GRID_CONSTANT       __grid_constant__
#   define ONIKA_DEVICE_CONSTANT_MEMORY __constant__
#   define ONIKA_CU_THREAD_LOCAL        __local__
#   define ONIKA_CU_BLOCK_SHARED        __shared__

#   define ONIKA_CU_BLOCK_SIMD_FOR(T,i,s,e)           for(T i=s+threadIdx.x ; i<e ; i+=blockDim.x )
#   define ONIKA_CU_BLOCK_SIMD_FOR_UNGUARDED(T,i,s,e) for(T _onika_tmp_j=s , i=s+threadIdx.x ; _onika_tmp_j<e ; _onika_tmp_j+=blockDim.x, i+=blockDim.x )

#   define ONIKA_CU_BLOCK_SYNC()   __syncthreads()
#   define ONIKA_CU_BLOCK_FENCE()  __threadfence_block()

#   define ONIKA_CU_DEVICE_FENCE() __threadfence()
#   define ONIKA_CU_SYSTEM_FENCE() __threadfence_system()

#   define ONIKA_CU_WARP_SYNC()    __syncwarp()
#   define ONIKA_CU_WARP_SHFL_DOWN_SYNC(mask,var,delta,width) __shfl_down_sync(mask,var,delta,width)
#   define ONIKA_CU_BLOCK_ACTIVE_MASK() __activemask()

#   define ONIKA_CU_MEM_ORDER_RELEASE std::memory_order_release
#   define ONIKA_CU_MEM_ORDER_RELAXED std::memory_order_relaxed
#   define ONIKA_CU_MEM_ORDER_ACQUIRE std::memory_order_acquire
#   define ONIKA_CU_MEM_ORDER_SEQ_CST std::memory_order_seq_cst

#   define ONIKA_CU_ATOMIC_STORE(x,a,...) ( * (volatile std::remove_reference_t<decltype(x)> *) &(x) ) = (a)
#   define ONIKA_CU_ATOMIC_LOAD(x,...)   ( * (volatile const std::remove_reference_t<decltype(x)> *) &(x) )
#   define ONIKA_CU_ATOMIC_ADD(x,a,...)  atomicAdd( &(x) , static_cast<std::remove_reference_t<decltype(x)> >(a) )
#   define ONIKA_CU_ATOMIC_SUB(x,a,...)  atomicSub( &(x) , static_cast<std::remove_reference_t<decltype(x)> >(a) )
#   define ONIKA_CU_ATOMIC_MIN(x,a,...)  atomicMin( &(x) , static_cast<std::remove_reference_t<decltype(x)> >(a) )
#   define ONIKA_CU_ATOMIC_MAX(x,a,...)  atomicMax( &(x) , static_cast<std::remove_reference_t<decltype(x)> >(a) )
#   define ONIKA_CU_BLOCK_ATOMIC_ADD(x,a) atomicAdd( &(x) , static_cast<std::remove_reference_t<decltype(x)> >(a) )

#   define ONIKA_CU_ATOMIC_FLAG_TEST_AND_SET(f) ( ! atomicCAS((uint32_t*)&(f),0,1) )
#   define ONIKA_CU_ATOMIC_FLAG_CLEAR(f) * (volatile uint32_t *) &(f) = 0

#   define ONIKA_CU_NANOSLEEP(ns) __nanosleep(ns)

#   define ONIKA_CU_CLOCK() clock64()
#   define ONIKA_CU_CLOCK_ELAPSED(a,b) ((b)-(a))
    using onika_cu_clock_t = long long int;

#   define ONIKA_CU_GRID_SIZE  gridDim.x
#   define ONIKA_CU_BLOCK_IDX  blockIdx.x
#   define ONIKA_CU_BLOCK_SIZE blockDim.x
#   define ONIKA_CU_THREAD_IDX threadIdx.x

#   define ONIKA_CU_VALUE_IF_CUDA(a,b) (a)

#   define ONIKA_CU_ABORT() ::onika::cuda::__onika_cu_abort()


/************** end of Cuda code definitions ***************/
#else 
/************** start of HOST code definitions ***************/

    template<class T>
    static inline T onika_omp_fetch_add(T* x , const T& a)
    {
      T r;
#     pragma omp atomic capture
      { r = *x ; *x += a; }
      return r;
    }

    using onika_cu_memory_order_t = std::memory_order;
    using gpu_device_execution_t = std::false_type;

#   define ONIKA_CU_GRID_CONSTANT       /**/
#   define ONIKA_DEVICE_CONSTANT_MEMORY /**/
#   define ONIKA_CU_BLOCK_SHARED        /**/
#   define ONIKA_CU_THREAD_LOCAL        /**/

#   define ONIKA_CU_BLOCK_SIMD_FOR_UNGUARDED ONIKA_CU_BLOCK_SIMD_FOR
#   define ONIKA_CU_BLOCK_SIMD_FOR(T,i,s,e) _Pragma("omp simd") for(T i=s ; i<e ; ++i)

#   define ONIKA_CU_BLOCK_SYNC()      (void)0
#   define ONIKA_CU_BLOCK_FENCE()     (void)0
#   define ONIKA_CU_BLOCK_WARP_SYNC() (void)0

#   define ONIKA_CU_DEVICE_FENCE() std::atomic_thread_fence(ONIKA_CU_MEM_ORDER_SEQ_CST)
#   define ONIKA_CU_SYSTEM_FENCE() std::atomic_thread_fence(ONIKA_CU_MEM_ORDER_SEQ_CST)

#   define ONIKA_CU_WARP_SHFL_DOWN_SYNC(mask,var,delta,width) (var)
#   define ONIKA_CU_BLOCK_ACTIVE_MASK() 1u

#   define ONIKA_CU_MEM_ORDER_RELEASE std::memory_order_release
#   define ONIKA_CU_MEM_ORDER_RELAXED std::memory_order_relaxed
#   define ONIKA_CU_MEM_ORDER_ACQUIRE std::memory_order_acquire
#   define ONIKA_CU_MEM_ORDER_SEQ_CST std::memory_order_seq_cst

#   define ONIKA_CU_ATOMIC_STORE(x,a,...) reinterpret_cast<std::atomic<std::remove_reference_t<decltype(x)> >*>(&(x))->store(a OPT_COMMA_VA_ARGS(__VA_ARGS__) )
#   define ONIKA_CU_ATOMIC_LOAD(x,...) reinterpret_cast< const std::atomic<std::remove_cv_t<std::remove_reference_t<decltype(x)> > > * >(&(x))->load( __VA_ARGS__ )
#   define ONIKA_CU_ATOMIC_ADD(x,a,...) ::onika::cuda::onika_omp_fetch_add( &(x) , static_cast<std::remove_reference_t<decltype(x)> >(a) )
//reinterpret_cast<std::atomic<std::remove_reference_t<decltype(x)> >*>(&(x))->fetch_add(a OPT_COMMA_VA_ARGS(__VA_ARGS__) )
#   define ONIKA_CU_ATOMIC_SUB(x,a,...) reinterpret_cast<std::atomic<std::remove_reference_t<decltype(x)> >*>(&(x))->fetch_sub(a OPT_COMMA_VA_ARGS(__VA_ARGS__) )
#   define ONIKA_CU_ATOMIC_MIN(x,a,...) _Pragma("omp atomic update") (x) = ( ((x)<=(a))?(x):(a) )
#   define ONIKA_CU_ATOMIC_MAX(x,a,...) _Pragma("omp atomic update") (x) = ( ((x)>=(a))?(x):(a) )
#   define ONIKA_CU_BLOCK_ATOMIC_ADD(x,a) (x) += (a)

#   define ONIKA_CU_ATOMIC_FLAG_TEST_AND_SET(f) ( ! (f).m_flag.test_and_set(std::memory_order_acquire) )
#   define ONIKA_CU_ATOMIC_FLAG_CLEAR(f) (f).m_flag.clear(std::memory_order_release)

#   define ONIKA_CU_NANOSLEEP(ns) std::this_thread::sleep_for(std::chrono::nanoseconds(ns))

#   define ONIKA_CU_CLOCK() std::chrono::high_resolution_clock::now()
#   define ONIKA_CU_CLOCK_ELAPSED(a,b) std::chrono::duration<double,std::micro>((b)-(a)).count()
    using onika_cu_clock_t = decltype(std::chrono::high_resolution_clock::now());

/*
namespace onika { namespace cuda { namespace _details {
  static unsigned int __onika_cu_grid_size = 1;
  static unsigned int __onika_cu_block_idx = 0;
//  inline unsigned int __onika_cu_gvars_use(){ return __onika_cu_grid_size + __onika_cu_block_idx; }
} } }
*/
#   define ONIKA_CU_GRID_SIZE  1 //((const unsigned int&)(::onika::cuda::_details::__onika_cu_grid_size))
#   define ONIKA_CU_BLOCK_IDX  0 //((const unsigned int&)(::onika::cuda::_details::__onika_cu_block_idx))
#   define ONIKA_CU_BLOCK_SIZE 1
#   define ONIKA_CU_THREAD_IDX 0

#   define ONIKA_CU_VALUE_IF_CUDA(a,b) (b)

#   define ONIKA_CU_ABORT() std::abort()

/************** end of HOST code definitions ***************/
#endif 






/***************************************************************/
/******************** Cuda language support ********************/
/***************************************************************/


/************** begin cuda-c code definitions ***************/
#   ifdef __CUDACC__

#   define ONIKA_DEVICE_KERNEL_FUNC __global__
#   if ONIKA_CU_ENABLE_KERNEL_BOUNDS == 1
#   define ONIKA_DEVICE_KERNEL_BOUNDS(MaxThreadsPerBlock,MinBlocksPerSM) __launch_bounds__(MaxThreadsPerBlock,MinBlocksPerSM)
#   else
#   define ONIKA_DEVICE_KERNEL_BOUNDS(MaxThreadsPerBlock,MinBlocksPerSM) /**/
#   endif
#   define ONIKA_HOST_DEVICE_FUNC __host__ __device__
#   define ONIKA_DEVICE_FUNC __device__
#   define ONIKA_BUILTIN_ASSUME_ALIGNED(p,a) p
#   define ONIKA_CU_LAUNCH_KERNEL(GDIM,BDIM,SHMEM,STREAM,FUNC, ... ) FUNC <<< GDIM , BDIM , SHMEM , STREAM >>> ( __VA_ARGS__ )

/************** end cuda-c code definitions ***************/
#   else
/************** begin of HOST code definitions ***************/

#   define ONIKA_DEVICE_FUNC /**/
#   define ONIKA_DEVICE_KERNEL_FUNC /**/
#   define ONIKA_DEVICE_KERNEL_BOUNDS(MaxThreadsPerBlock,MinBlocksPerSM) /**/
#   define ONIKA_HOST_DEVICE_FUNC /**/
#   define ONIKA_BUILTIN_ASSUME_ALIGNED(p,a) __builtin_assume_aligned(p,a)
#   define ONIKA_CU_LAUNCH_KERNEL(GDIM,BDIM,SHMEM,STREAM,FUNC, ... ) do{ \
	FAKE_USE_OF_VARIABLES(__VA_ARGS__) \
	printf("Illegal call to Kernel %s<<<%d,%d,%d,%ld>>>(...) with no GPU support, in %s at %s:%d\n",#FUNC,int(GDIM),int(BDIM),int(SHMEM),long(STREAM),__FUNCTION__,__FILE__,__LINE__); \
	std::abort(); \
    }while(false)
#   endif
/************** end host code definitions ***************/




/***************************************************************/
/************************ Cuda API calls ***********************/
/***************************************************************/
# ifdef ONIKA_CUDA_VERSION
#   define ONIKA_CU_CREATE_EVENT(EVT) cudaEventCreate(&EVT)
#   define ONIKA_CU_DESTROY_EVENT(EVT) cudaEventDestroy(EVT)
#   define ONIKA_CU_STREAM_EVENT(EVT,STREAM) cudaEventRecord(EVT,STREAM)
#   define ONIKA_CU_EVENT_ELAPSED(T,EVT1,EVT2) cudaEventElapsedTime(&T,EVT1,EVT2)
#   define ONIKA_CU_STREAM_SYNCHRONIZE(STREAM)  cudaStreamSynchronize(STREAM)
#   define ONIKA_CU_SET_DEVICE(dev)  cudaSetDevice(dev)
#   define ONIKA_CU_MEMSET(p,v,n,...) cudaMemsetAsync(p,v,n OPT_COMMA_VA_ARGS(__VA_ARGS__) )
#   define ONIKA_CU_MEMCPY(d,s,n,...) cudaMemcpyAsync(d,s,n,cudaMemcpyDefault OPT_COMMA_VA_ARGS(__VA_ARGS__) )
# else
#   define ONIKA_CU_CREATE_EVENT(EVT) _fake_cuda_api_noop(EVT=nullptr)
#   define ONIKA_CU_DESTROY_EVENT(EVT) _fake_cuda_api_noop(EVT=nullptr)
#   define ONIKA_CU_STREAM_EVENT(EVT,STREAM) _fake_cuda_api_noop(EVT,STREAM)
#   define ONIKA_CU_EVENT_ELAPSED(T,EVT1,EVT2) _fake_cuda_api_noop(T=0.0f)
#   define ONIKA_CU_STREAM_SYNCHRONIZE(STREAM) _fake_cuda_api_noop(STREAM)
#   define ONIKA_CU_SET_DEVICE(dev) _fake_cuda_api_noop(dev)
#   define ONIKA_CU_MEMSET(p,v,n,...) std::memset(p,v,n)
#   define ONIKA_CU_MEMCPY(d,s,n,...) std::memcpy(d,s,n)
# endif




/***************************************************************/
/****************** Cuda hardware intrinsics *******************/
/***************************************************************/

#   ifdef __CUDA_ARCH__
    __noinline__  __device__ inline unsigned int get_smid(void)
    {
      unsigned int ret;
      asm("mov.u32 %0, %smid;" : "=r"(ret) );
      return ret;
    }
#   else
    inline constexpr unsigned int get_smid(void) { return 0; }
#   endif

    template<class T>
    struct alignas( alignof(T) ) UnitializedPlaceHolder
    {
      static_assert( sizeof(unsigned char) == 1 , "expected char to be 1 byte" );
      unsigned char byte[sizeof(T)];
    };

  }
}

