#pragma once

#include <cstdlib>
#include <cassert>
#include <vector>
#include <onika/cuda/cuda.h>

namespace onika
{

  namespace cuda
  {

    template<class T> struct VectorShallowCopy
    {
      T* m_data = nullptr;
      size_t m_size = 0;
      
      VectorShallowCopy() = default;
      VectorShallowCopy(const VectorShallowCopy&) = default;
      VectorShallowCopy(VectorShallowCopy&&) = default;
      VectorShallowCopy& operator = (const VectorShallowCopy&) = default;
      VectorShallowCopy& operator = (VectorShallowCopy&&) = default;
      
      ONIKA_HOST_DEVICE_FUNC inline T* data() { return m_data; }
      ONIKA_HOST_DEVICE_FUNC inline const T* data() const { return m_data; }
      
      ONIKA_HOST_DEVICE_FUNC inline bool empty() const { return m_size==0; }
      ONIKA_HOST_DEVICE_FUNC inline size_t size() const { return m_size; }
      ONIKA_HOST_DEVICE_FUNC inline void resize(size_t n) const { assert( n == m_size ); }
      
      ONIKA_HOST_DEVICE_FUNC inline T& operator [] (size_t i) { return m_data[i]; }
      ONIKA_HOST_DEVICE_FUNC inline const T& operator [] (size_t i) const { return m_data[i]; }
    };

    template<class T> struct ReadOnlyShallowCopyType { using type = T; };
    template<class T, class A> struct ReadOnlyShallowCopyType< std::vector<T,A> > { using type = VectorShallowCopy<T>; };

    template<class T> using ro_shallow_copy_t = typename ReadOnlyShallowCopyType<T>::type;
    
  }

}

