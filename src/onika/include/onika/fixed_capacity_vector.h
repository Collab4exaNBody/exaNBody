#pragma once

#include <cstdlib>
#include <onika/cuda/cuda.h>

namespace onika
{

  /*
  Vector like container, with a fixed, pre-allocated memory pool passed in the form of pointer and size
  */

  template<class T, class SizeType = uint32_t>
  struct FixedCapacityVector
  {
    T* m_storage_ptr = nullptr;
    SizeType m_capacity = 0;
    SizeType m_size = 0;

    FixedCapacityVector() = default;
    FixedCapacityVector(const FixedCapacityVector&) = default;
    FixedCapacityVector(FixedCapacityVector&&) = default;
    FixedCapacityVector& operator = (const FixedCapacityVector&) = default;
    FixedCapacityVector& operator = (FixedCapacityVector&&) = default;
    
    ONIKA_HOST_DEVICE_FUNC inline FixedCapacityVector( T* ptr, size_t cap, size_t sz = 0 )
      : m_storage_ptr(ptr)
      , m_capacity(cap)
      , m_size(sz)
    {
      int al = ( (int64_t)( ((uint8_t*)ptr) - (uint8_t*)nullptr ) ) % alignof(T);
      if( al != 0 )
      {
        printf("Misaligned address for FixedCapacityVector<T>: alignof(T)=%d , sizeof(T)=%d, mod=%d\n",int(alignof(T)),int(sizeof(T)),al);
        ONIKA_CU_ABORT();
      }
    }

    ONIKA_HOST_DEVICE_FUNC inline size_t size() const
    {
      return m_size;
    }
    
    ONIKA_HOST_DEVICE_FUNC inline size_t capacity() const
    {
      return m_capacity;
    }

    ONIKA_HOST_DEVICE_FUNC inline size_t available() const
    {
      return m_capacity - m_size;
    }
    
    ONIKA_HOST_DEVICE_FUNC inline bool full() const
    {
      return m_size == m_capacity;
    }
    
    ONIKA_HOST_DEVICE_FUNC inline T* data() const { return m_storage_ptr; }

    ONIKA_HOST_DEVICE_FUNC inline void resize(size_t n)
    {
      assert( n <= m_capacity );
      m_size = n;
    }

    ONIKA_HOST_DEVICE_FUNC inline void assign(size_t n, const T& x)
    {
      assert( n <= m_capacity );
      m_size = n;
      //if( n > m_capacity ) n = m_capacity;
      for(size_t i=0;i<n;i++) { m_storage_ptr[i] = x; }
    }

    ONIKA_HOST_DEVICE_FUNC inline void clear() { m_size = 0; }

    ONIKA_HOST_DEVICE_FUNC inline void push_back( T && x )
    {
      assert( m_size < m_capacity );
      /* if( m_size < m_capacity ) */ m_storage_ptr[m_size] = std::move(x);
      ++ m_size;
    }
    
    ONIKA_HOST_DEVICE_FUNC inline void push_back( const T & x )
    {
      assert( m_size < m_capacity );
      /* if( m_size < m_capacity ) */ m_storage_ptr[m_size] = x;
      ++ m_size;
    }

    ONIKA_HOST_DEVICE_FUNC inline T& operator [] ( size_t i )
    {
      assert( i < m_size );
      return m_storage_ptr[i];
    }
    
    ONIKA_HOST_DEVICE_FUNC inline const T& operator [] ( size_t i ) const
    {
      assert( i < m_size );
      return m_storage_ptr[i];
    }
    
  };

}


