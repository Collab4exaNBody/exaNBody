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

