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
#include <exanb/core/operator.h>
#include <onika/log.h>
#include <onika/print_utils.h>

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

