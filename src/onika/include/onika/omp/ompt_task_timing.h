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


