#pragma once

#ifdef ONIKA_CUDA_VERSION
#include <cuda_runtime.h>
#endif // ONIKA_CUDA_VERSION

#include <onika/memory/memory_usage.h>
#include <onika/cuda/cuda_context.h>

// specializations to avoid MemoryUsage template to dig into cuda aggregates
namespace onika
{

  namespace cuda
  {

    struct CudaDeviceStorageShared
    {
      size_t m_array_size = 0;
      size_t m_refcount = 0;
    };

    template<class T>
    struct CudaDeviceStorage
    {
      T* m_ptr = nullptr;
      CudaDeviceStorageShared* m_shared = nullptr;

      CudaDeviceStorage() = default;
      inline CudaDeviceStorage(const CudaDeviceStorage& other) : m_ptr(other.m_ptr), m_shared(other.m_shared) { ++ m_shared->m_refcount; }
      inline CudaDeviceStorage(CudaDeviceStorage && other) : m_ptr(other.m_ptr), m_shared(other.m_shared) { other.m_ptr=nullptr; other.m_shared=nullptr; }

      static inline CudaDeviceStorage<T> New(CudaDevice& dev, size_t n = 1)
      {
        T* devPtr = nullptr;
#       ifdef ONIKA_CUDA_VERSION
          cudaMalloc( & devPtr , sizeof(T) * n );
#       else
          devPtr = new T [ n ];
#       endif
        CudaDeviceStorageShared* shctx = new CudaDeviceStorageShared { n , 1 };
        return CudaDeviceStorage<T>( devPtr ,shctx  );
      }

      inline void reset()
      {
        if( m_shared != nullptr )
        {
          -- m_shared->m_refcount;
          if( m_shared->m_refcount == 0 )
          {
            delete m_shared;
            if( m_ptr != nullptr )
            {
#           ifdef ONIKA_CUDA_VERSION
              cudaFree( m_ptr );
#           else
              delete [] m_ptr;
#           endif
            }
          }
        }
        m_ptr = nullptr;
        m_shared = nullptr;
      }

      inline ~CudaDeviceStorage()
      {
        reset();
      }
      
      inline CudaDeviceStorage& operator = (const CudaDeviceStorage& other)
      {
        if( m_shared != other.m_shared )
        {
          reset();     
          m_ptr = other.m_ptr;
          m_shared = other.m_shared;
          ++ m_shared->m_refcount;
        }
        else
        {
          assert( m_ptr == other.m_ptr );
        }
        return *this;
      }
      
      inline CudaDeviceStorage& operator = (CudaDeviceStorage&& other)
      {
        m_ptr = other.m_ptr;
        m_shared = other.m_shared;
        other.m_ptr = nullptr;
        other.m_shared = nullptr;
        return *this;
      }

      inline bool operator == (const CudaDeviceStorage& other) const { return m_ptr == other.m_ptr; }
      inline bool operator == (const T* ptr) const { return m_ptr == ptr; }
      inline bool operator == (std::nullptr_t) const { return m_ptr == nullptr; }

      inline bool operator != (const CudaDeviceStorage& other) const { return m_ptr != other.m_ptr; }
      inline bool operator != (const T* ptr) const { return m_ptr != ptr; }
      inline bool operator != (std::nullptr_t) const { return m_ptr != nullptr; }


      inline const T& operator *  () const { assert(m_ptr != nullptr); return *m_ptr; }
      inline       T& operator *  ()       { assert(m_ptr != nullptr); return *m_ptr; }
      inline const T* operator -> () const { assert(m_ptr != nullptr); return  m_ptr; }
      inline       T* operator -> ()       { assert(m_ptr != nullptr); return  m_ptr; }

      inline const T* get()  const { return  m_ptr; }
      inline       T* get()        { return  m_ptr; }

    private:
      CudaDeviceStorage(T* p,CudaDeviceStorageShared* s) : m_ptr(p) , m_shared(s) {}
    };

  }

}


