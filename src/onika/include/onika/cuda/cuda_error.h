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


