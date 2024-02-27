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

#include <onika/parallel/parallel_for.h>

namespace onika
{

  namespace parallel
  {  

    template<class T>
    struct MemSetFunctor
    {
      T * __restrict__ m_data = nullptr;
      T m_value = {};  
      ONIKA_HOST_DEVICE_FUNC inline void operator () ( size_t i ) const
      {
        m_data[i] = m_value;
      }
    };

    template<class T> struct ParallelForFunctorTraits< MemSetFunctor<T> >
    {      
      static inline constexpr bool CudaCompatible = true;
    };


    template<class T>
    static inline ParallelExecutionWrapper parallel_memset( T * data , uint64_t N, const T& value, ParallelExecutionContext * pec )
    {
      return parallel_for( N , MemSetFunctor<T>{ data , value } , pec );
    }

  }

}

