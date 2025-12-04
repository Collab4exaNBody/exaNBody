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

#include <onika/cuda/cuda.h>

namespace md
{
  
  template<bool UseAtomic>
  struct SwitchableAtomicAccumFunctor
  {
    template<class T>
    ONIKA_HOST_DEVICE_FUNC
    ONIKA_ALWAYS_INLINE
    void operator () ( T & x , const T & y ) const
    {
      if constexpr ( UseAtomic ) { ONIKA_CU_BLOCK_ATOMIC_ADD( x , y ); }
      else { x += y; }
    }
  };

  using SimpleAccumFunctor = SwitchableAtomicAccumFunctor<false>;
  using AtomicAccumFunctor = SwitchableAtomicAccumFunctor<true>;
}
