#pragma once

#include <cstdlib>
#include <onika/memory/simd.h>
#include <vector>
#include <algorithm>
#include <onika/cuda/cuda.h>

namespace onika { namespace memory
{
  /*
    Host allocation kinds
  */
  enum class HostAllocationPolicy
  {
    MALLOC    = 0 ,
    CUDA_HOST = 1
  };

# ifdef ONIKA_CUDA_VERSION
  static inline constexpr HostAllocationPolicy CUDA_FALLBACK_ALLOC_POLICY = HostAllocationPolicy::CUDA_HOST;
# else
  static inline constexpr HostAllocationPolicy CUDA_FALLBACK_ALLOC_POLICY = HostAllocationPolicy::MALLOC;
# endif

  // prefered default alignment and chunk size on this machine
  static inline constexpr size_t DEFAULT_ALIGNMENT = SimdRequirements<double>::alignment;
  static inline constexpr size_t DEFAULT_CHUNK_SIZE = SimdRequirements<float>::chunksize;
  static inline constexpr size_t MINIMUM_CUDA_ALIGNMENT = 64;

  struct MemoryChunkInfo
  {
    void * alloc_base = nullptr;
    size_t alloc_size = 0;
    uint32_t alloc_flags = 0;  
       
    inline HostAllocationPolicy mem_type() const { return static_cast<HostAllocationPolicy>( alloc_flags & 0xFF ); }
    inline unsigned int alignment() const { return alloc_flags >> 8; }
    inline void * base_ptr() const { return alloc_base; }
    inline size_t size() const { return alloc_size; }
  };

  /*
    Configurable host allocator, either based on malloc or cudaMallocManaged
  */
  struct GenericHostAllocator
  {
    static inline constexpr size_t DefaultAlignBytes = std::max( MINIMUM_CUDA_ALIGNMENT , DEFAULT_ALIGNMENT );

#   ifndef NDEBUG
    static constexpr size_t add_info_size = sizeof(size_t) + sizeof(uint32_t);
    static bool s_enable_debug_log;
    static void set_debug_log(bool b);
#   else
    static constexpr size_t add_info_size = sizeof(uint32_t);
    static inline constexpr void set_debug_log(bool){}
#   endif

    MemoryChunkInfo memory_info( void* ptr , size_t s ) const;
    void* allocate( size_t s , size_t a ) const;
    void deallocate( void* ptr , size_t s ) const;
    bool is_gpu_addressable( void* ptr , size_t s ) const;
    bool operator == (const GenericHostAllocator& other) const;
    HostAllocationPolicy get_policy() const;
    bool allocates_gpu_addressable() const;
    void set_gpu_addressable_allocation(bool yn );
    
    HostAllocationPolicy m_alloc_policy = HostAllocationPolicy::MALLOC;

#   ifdef ONIKA_CUDA_VERSION
    static bool s_enable_cuda;
    static bool cuda_enabled();
    static void set_cuda_enabled(bool yn);
#   else
    static inline constexpr bool cuda_enabled() { return false; }
    static inline constexpr void set_cuda_enabled(bool) {}
#   endif
  };

  // STL compatible host memory allocator.
  template <class T>
  struct CudaManagedAllocator
  {
    typedef T value_type;

    inline T* allocate (std::size_t n)
    {
      constexpr size_t al = (CUDA_FALLBACK_ALLOC_POLICY==HostAllocationPolicy::CUDA_HOST) ? std::max( alignof(T) , MINIMUM_CUDA_ALIGNMENT ) : alignof(T);
      return static_cast<T*>( GenericHostAllocator{CUDA_FALLBACK_ALLOC_POLICY} .allocate( sizeof(T) * n , al ) );
    }

    inline void deallocate (T* p, std::size_t n)
    {
      GenericHostAllocator{CUDA_FALLBACK_ALLOC_POLICY} .deallocate( p , sizeof(T) * n );
    }

    template<class U> inline bool operator == (const U&) const { return false; }
    inline bool operator == (const CudaManagedAllocator<T>&) const { return true; }

    template<class U> inline bool operator != (const U& other) const { return ! (*this == other); }
  };

  // default allocator used if none provided
  using DefaultAllocator = GenericHostAllocator;

  template <class T>
  struct NullAllocator
  {
    typedef T value_type;
    ONIKA_HOST_DEVICE_FUNC inline T* allocate (std::size_t n) { return nullptr; }
    ONIKA_HOST_DEVICE_FUNC inline void deallocate (T* p, std::size_t n) { }

    template<class U> ONIKA_HOST_DEVICE_FUNC inline bool operator == (const U&) const { return false; }
    ONIKA_HOST_DEVICE_FUNC inline bool operator == (const NullAllocator<T>&) const { return true; }

    template<class U> ONIKA_HOST_DEVICE_FUNC inline bool operator != (const U& other) const { return ! (*this == other); }    
  };

  // Host-Device compatible STL vectors
  template<class T> using CudaMMVector = std::vector< T , CudaManagedAllocator<T> >;

  // useful macro to indicate the compiler a pointer is aligned
# ifndef __CUDACC__
# define ONIKA_ASSUME_ALIGNED(x) x = ( decltype(x) __restrict__ ) __builtin_assume_aligned( x , ::onika::memory::DEFAULT_ALIGNMENT )
# define ONIKA__bultin_assume_aligned(x,a) __builtin_assume_aligned( x , a )
# else
# define ONIKA_ASSUME_ALIGNED(x) while(false)
# define ONIKA__bultin_assume_aligned(x,a) (x)
# endif

}

}

