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
#include <iostream>

#ifdef __CUDACC__

#include <cuda_runtime_api.h>

#ifndef checkCudaErrors
#define checkCudaErrors( _expr_ ) ::onika::cuda::assertCudaSuccess((_expr_), __FILE__, __LINE__)
#endif

namespace onika
{
  namespace cuda
  {
    inline void assertCudaSuccess(cudaError_t code, const char *file, int line, bool abort_on_failure=true)
    {
       if( code != cudaSuccess ) 
       {
          std::cerr << "Cuda error : " << cudaGetErrorString(code) <<"\n"<< file << ":"<< line << "\n";
          if( abort_on_failure ) std::abort();
       }
    }
  } 
}

#else

//using cudaError_t = int;
#ifndef checkCudaErrors
#define checkCudaErrors( _expr_ ) do{ auto x=(_expr_); auto y=x; x=y; }while(0)
#endif

#endif


