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

