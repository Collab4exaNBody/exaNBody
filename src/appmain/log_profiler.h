#pragma once

#include <onika/omp/ompt_task_timing.h>

namespace exanb
{
  namespace main
  {
    extern void log_profiler_stop_event( const onika::omp::OpenMPToolTaskTiming& e );
  }
}

