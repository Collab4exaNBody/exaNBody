#pragma once

#include <chrono>
#include <iostream>

namespace exanb
{
  using ProfilingTimer = decltype( std::chrono::high_resolution_clock::now() );

  static inline void profiling_timer_start(ProfilingTimer& t0)
  {
    t0 = std::chrono::high_resolution_clock::now();
  }
  
  static inline double profiling_timer_elapsed_restart(ProfilingTimer& t0)
  {
    ProfilingTimer T1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double,std::milli>( T1 - t0 ).count();
    t0 = T1;
    return elapsed;
  }

  struct ProfilingAccountTimeNullFunc
  {
    inline void operator () (double) const {}
  };

}

