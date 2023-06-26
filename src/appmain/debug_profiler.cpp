#include <exanb/core/operator.h>
#include <exanb/core/log.h>
#include <exanb/core/print_utils.h>

#include <onika/omp/ompt_task_timing.h>

#include <iostream>
#include <string>

#include "debug_profiler.h"

using namespace exanb;

namespace exanb
{
  namespace main
  {

    const char * g_profiler_current_tag = nullptr;

    void profiler_record_tag( const onika::omp::OpenMPToolTaskTiming& e )
    {
#     ifndef ONIKA_HAVE_OPENMP_TOOLS
      g_profiler_current_tag = e.tag;
#     endif
    }
    
  }
}

