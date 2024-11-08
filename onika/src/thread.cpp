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

#include <onika/thread.h>
#include <onika/log.h>

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <chrono>

namespace onika
{

  // --------- per thread timing functions ---------------
  std::chrono::nanoseconds get_thread_cpu_time()
  {
    struct timespec T;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID,&T);
    return std::chrono::nanoseconds { T.tv_sec * 1000000000ull + T.tv_nsec };
  }

  std::chrono::nanoseconds wall_clock_time()
  {
    static const auto wall_clock_ref = std::chrono::high_resolution_clock::now();
    return std::chrono::high_resolution_clock::now() - wall_clock_ref;
  }
  
  // --------- reliable thread indexing ---------------
  static std::atomic<int64_t> g_global_thread_index = 0;
  static std::unordered_map< std::thread::id , int64_t > initialize_thread_indices()
  {
    std::unordered_map< std::thread::id , int64_t > im;
#   pragma omp parallel
    {
#     pragma omp critical(initialize_thread_indices_cs)
      {
        im[ std::this_thread::get_id() ] = g_global_thread_index;
        ++ g_global_thread_index;
      }
    }
    return im;
  }
  static std::unordered_map< std::thread::id , int64_t > g_thread_index_map = initialize_thread_indices();
  static std::unordered_map< std::thread::id , int64_t > g_dynamic_thread_index_map;
  static std::mutex g_thread_index_mutex;
  size_t get_thread_index()
  {
    std::thread::id tid = std::this_thread::get_id();
    auto it = g_thread_index_map.find( tid );
    if( it == g_thread_index_map.end() )
    {
      std::scoped_lock{ g_thread_index_mutex };
      it = g_dynamic_thread_index_map.find( tid );
      if( it == g_dynamic_thread_index_map.end() )
      {
        int i = g_global_thread_index;
        ++ g_global_thread_index;
        g_dynamic_thread_index_map[ tid ] = g_global_thread_index;
        return i;
      }
      else
      {
        return it->second;
      }
    }
    else
    {
      return it->second;
    }
  }
  // ---------------------------------------------------

}

