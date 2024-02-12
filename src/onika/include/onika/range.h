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

//#include <iterator>
#include <onika/cuda/cuda.h>

namespace onika
{
    template<class Iterator>
    struct IteratorRangeView
    {
      Iterator m_begin;
      Iterator m_end;
      ONIKA_HOST_DEVICE_FUNC inline Iterator begin() const { return m_begin; }
      ONIKA_HOST_DEVICE_FUNC inline Iterator end() const { return m_end; }
      ONIKA_HOST_DEVICE_FUNC inline auto size() const { return end() - begin(); /*std::distance(begin(),end());*/ }
    };

    template<class Iterator>
    static inline IteratorRangeView<Iterator> make_iterator_range_view(Iterator b, Iterator e) { return {b,e}; }

    template<class T>
    struct SingleValueRangeView
    {
      T m_value;
      inline T * begin() const { return &m_value; }
      inline T * end() const { return begin()+1; }
    };

    template<class T>
    struct EmptyRangeView
    {
      static inline constexpr T* begin() { return nullptr; }
      static inline constexpr T* end() { return nullptr; }
    };

}
