#pragma once

#include <chrono>
#include <cstdint>

// #include <time.h> // clock_gettime

namespace onika { namespace omp
{

  struct OpenMPToolTaskTiming
  {
    void* ctx = nullptr;  // application specific context pointer
    const char* tag = nullptr; // task kernel id
    ssize_t cpu_id = -1; // resource id (actualy OpenMP thread id)
    std::chrono::nanoseconds timepoint {0};
    std::chrono::nanoseconds end {0};
    size_t resume = 0;
    
    inline std::chrono::nanoseconds elapsed() const { return end - timepoint; }

    static inline std::chrono::nanoseconds wall_clock_time()
    {
      static const auto wall_clock_ref = std::chrono::high_resolution_clock::now();
      return std::chrono::high_resolution_clock::now() - wall_clock_ref;
    }

  };

} }


