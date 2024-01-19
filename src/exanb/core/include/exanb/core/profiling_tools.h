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

