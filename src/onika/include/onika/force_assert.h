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

#include <iostream>
#include <cstdlib>
#include <onika/cuda/cuda.h>

namespace onika
{
  ONIKA_HOST_DEVICE_FUNC inline void _onika_force_assert( bool t , const char* cond, const char* func , const char* file, int line)
  {
    if( !t )
    {
      printf("Assertion '%s' failed\n\tin function '%s'\n\tat %s:%d\n",cond,func,file,line);
      ONIKA_CU_ABORT();
    }
  }
}

#ifdef __INTEL_COMPILER
#define ONIKA_FORCE_ASSERT(cond) ::onika::_onika_force_assert(cond,#cond,__func__,__FILE__,__LINE__)
#else
#define ONIKA_FORCE_ASSERT(cond) ::onika::_onika_force_assert(cond,#cond,"<unknown>",__FILE__,__LINE__)
#endif
