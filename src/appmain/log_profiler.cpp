#include <exanb/core/operator.h>
#include <exanb/core/log.h>
#include <exanb/core/print_utils.h>

#include <onika/omp/ompt_task_timing.h>

#include <iostream>
#include <string>

#include "log_profiler.h"

using namespace exanb;

namespace exanb
{
  namespace main
  {

    void log_profiler_stop_event( const onika::omp::OpenMPToolTaskTiming& e )
    {
#     ifdef ONIKA_HAVE_OPENMP_TOOLS
      if( std::string(e.tag) != "sequential" || e.ctx==nullptr ) return;
#     else
      if( e.tag==nullptr || e.ctx==nullptr ) return;
#     endif

      OperatorNode* op = reinterpret_cast<OperatorNode*>( e.ctx );
      if( op->is_terminal() )
      {
        std::chrono::duration<double,std::milli> elapsed_ms = e.elapsed();
#       ifdef ONIKA_HAVE_OPENMP_TOOLS
        lout << op->pathname() << " "<< elapsed_ms.count() << " ms" << std::endl ;
#       else
        lout << e.tag << " "<< elapsed_ms.count() << " ms" << std::endl ;
#       endif
      }

    }

  }
}

