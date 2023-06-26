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

