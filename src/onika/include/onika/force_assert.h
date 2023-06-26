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

#define ONIKA_FORCE_ASSERT(cond) ::onika::_onika_force_assert(cond,#cond,__func__,__FILE__,__LINE__)

