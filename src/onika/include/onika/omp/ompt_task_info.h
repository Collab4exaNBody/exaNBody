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

namespace onika { namespace omp
{

  struct OpenMPToolThreadContext;

  struct OpenMPTaskInfo
  {
    const char* tag = nullptr;
    void* app_ctx = nullptr;
    OpenMPToolThreadContext* thread_ctx = nullptr;
    OpenMPTaskInfo* prev = nullptr;
    const char* explicit_task_tag = nullptr;
    bool dyn_alloc = false;
    bool allocated = true;

#   ifndef NDEBUG
    uint8_t _padding[6];
    static inline constexpr uint64_t DYN_ALLOCATED_TASK = uint64_t(0x26101976);
    static inline constexpr uint64_t INITIAL_TASK = uint64_t(0x13072007);
    static inline constexpr uint64_t TEMPORARY_TASK = uint64_t(0x04042010);
    uint64_t magic = TEMPORARY_TASK;
    inline ~OpenMPTaskInfo() { allocated=false; magic=0; }
#   endif

  };

} }

