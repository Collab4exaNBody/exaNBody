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

#include <omp.h>

#include <onika/macro_utils.h>

#define OPENMP_DETACH_WARNING 1

/*
  Basic rules for data sharing :
  {
    int x = 3;
    #pragma omp task
    {
      x = 5; // x is firstprivate
    }
  }
  {
    static int x = 3;
    #pragma omp task
    {
      x = 5; // x is shared
    }
  }
  ==> use only local variables for firstprivate copîes, and pointers for shared.
*/

#ifdef ONIKA_HAVE_OPENMP_DETACH

# if defined(__clang__)

#   define OMP_TASK_DETACH(sharing_clauses,other_clauses,evt) _Pragma( ONIKA_STR(omp task sharing_clauses other_clauses firstprivate(evt) detach(evt)) )

# elif defined(__INTEL_COMPILER)

// Intel compiler frontend does not support detach clause yet
#   ifdef OPENMP_DETACH_WARNING
#     warning ONIKA_HAVE_OPENMP_DETACH disabled, compiler support missing
#   endif
#   undef ONIKA_HAVE_OPENMP_DETACH

# elif defined(__GNUC__) && __GNUC__>=11

#   ifdef OPENMP_DETACH_WARNING
#     warning GCC-11 omp task detach support is partial, data sharing clauses are ignored
#   endif

#   define OMP_TASK_DETACH(sharing_clauses,other_clauses,evt) _Pragma( ONIKA_STR(omp task /*sharing_clauses*/ other_clauses detach(evt)) )

# else

#   ifdef OPENMP_DETACH_WARNING
#     warning ONIKA_HAVE_OPENMP_DETACH disabled, compiler support missing
#   endif
#   undef ONIKA_HAVE_OPENMP_DETACH

# endif

#endif


// fallback, will compile but may be less robust (or fail) at run time
#ifndef ONIKA_HAVE_OPENMP_DETACH

# include<atomic>
namespace onika
{

  struct _omp_event_t
  {
    std::atomic<uint32_t> count = 0;
    std::atomic<uint32_t> flag = 0;
  };

  struct _omp_event_handle_t
  {
    _omp_event_handle_t() = default;
    
    inline _omp_event_handle_t( _omp_event_handle_t && e) noexcept
    {
      if( evt != nullptr ) { delete evt; }
      evt = e.evt;
      e.evt = nullptr;
    }
 
    inline _omp_event_handle_t& operator = ( _omp_event_handle_t && e) noexcept
    {
      if( evt != nullptr ) { delete evt; }
      evt = e.evt;
      e.evt = nullptr;
      return *this;
    }

    // heart of the method : we assume that the only legit copy construction will occure during task firstprivate capture of _omp_event_handle_t var
    inline _omp_event_handle_t(const _omp_event_handle_t& e) noexcept
    {
      assert( e.evt != nullptr );
      assert( e.evt->count.load(std::memory_order_acquire) == 0 );
      evt = e.evt;
      evt->count.fetch_add(1,std::memory_order_release);
    }

    inline void init() noexcept
    {
      if(evt==nullptr)
      {
        evt = new _omp_event_t{};
      }
      else
      {
        evt->count.store(0,std::memory_order_release);
        evt->flag.store(0,std::memory_order_release);
      }
    }
    inline ~_omp_event_handle_t()
    {
      if( evt != nullptr )
      {
        auto cnt = evt->count.load(std::memory_order_acquire);
        if( cnt )
        {
          assert( cnt == 1 );
          while( ! evt->flag.load(std::memory_order_acquire) )
          {
            _Pragma("omp taskyield")
          }
        }
        else
        {
          delete evt;
        }
      }
    }
    _omp_event_t* evt = nullptr;
  };
  inline void _omp_fulfill_event(const onika::_omp_event_handle_t& e)
  {
    assert( e.evt != nullptr );
    assert( e.evt->count.load(std::memory_order_acquire) == 1 );
    assert( e.evt->flag.load(std::memory_order_acquire) == 0 );
    e.evt->flag.store(1,std::memory_order_release);
  }
}
# undef omp_event_handle_t
# define omp_event_handle_t ::onika::_omp_event_handle_t
# undef omp_fulfill_event
# define omp_fulfill_event ::onika::_omp_fulfill_event
# define OMP_TASK_DETACH(sharing_clauses,other_clauses,evt) evt.init(); _Pragma( ONIKA_STR(omp task sharing_clauses other_clauses firstprivate(evt)) )

#endif

namespace onika
{
  namespace _compile_test
  {
    inline void test_omp_detach_compiler_frontend()
    {
      static_assert( sizeof(omp_event_handle_t) == sizeof(void*) );
      int i = 3;
      omp_event_handle_t evt{}; if( ((&evt)-(omp_event_handle_t*)nullptr) == 0 ){}
      static_assert( sizeof(evt) == sizeof(omp_event_handle_t) );
      OMP_TASK_DETACH( default(none) firstprivate(i) , untied , evt )
      {
        int* pi = &i;
        i = *pi;
        //OMP_TASK_DETACH_WAIT(evt);
      }
      //auto * _evt = &evt; evt = *_evt;
    }
  }
}

