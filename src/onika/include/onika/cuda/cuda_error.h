#pragma once

#include <cstdlib>
#include <iostream>
#include <onika/cuda/cuda_context.h>

#if defined(ONIKA_CUDA_VERSION) || defined(ONIKA_HIP_VERSION)

#ifndef ONIKA_CU_CHECK_ERRORS
#define ONIKA_CU_CHECK_ERRORS( _expr_ ) ::onika::cuda::assertSuccess((_expr_), __FILE__, __LINE__)
#endif

namespace onika
{
  namespace cuda
  {
    inline void assertSuccess(onikaError_t code, const char *file, int line, bool abort_on_failure=true)
    {
       if( code != hipSuccess ) 
       {
          std::cerr << ONIKA_CU_NAME_STR << " error : " << ONIKA_CU_GET_ERROR_STRING(code) <<"\n"<< file << ":"<< line << "\n";
          if( abort_on_failure ) std::abort();
       }
    }
  } 
}

#else

#ifndef ONIKA_CU_CHECK_ERRORS
#define ONIKA_CU_CHECK_ERRORS( _expr_ ) do{ auto x=(_expr_); auto y=x; x=y; }while(0)
#endif

#endif


