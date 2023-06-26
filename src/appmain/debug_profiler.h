#pragma once

#include <onika/omp/ompt_task_timing.h>

namespace exanb
{
  namespace main
  {
    extern const char * g_profiler_current_tag;
    extern void profiler_record_tag( const onika::omp::OpenMPToolTaskTiming& e );
  }
}

