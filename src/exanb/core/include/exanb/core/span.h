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

namespace exanb
{
  template<class T>
  struct span
  {
    inline T* begin() const { return m_begin; }
    inline T* end() const { return m_begin+m_size; }
    T* m_begin = nullptr;
    size_t m_size = 0;
  };
  
  template<class T> inline span<T> make_span(T* p, size_t n) { return span<T>{p,n}; }
  template<class T,class A> inline span<T> make_span(std::vector<T,A>& v, size_t n) { return span<T>{v.data(),n}; }

  template<class T> inline span<const T> make_const_span(const T* p, size_t n) { return span<const T>{p,n}; }
  template<class T,class A> inline span<const T> make_const_span(const std::vector<T,A>& v) { return span<const T>{v.data(),v.size()}; }
}

