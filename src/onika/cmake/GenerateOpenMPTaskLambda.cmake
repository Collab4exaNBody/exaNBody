# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
# 
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# function to generate OpenMP helper for variable number of dependencies
#    // helper method to get nth ref among a list
#    template<class T0> static inline auto& get_nth_arg( std::integral_constant<int,0> , T0& x ) { return x; }

function(GenerateOpenMPTaskLambda outfile maxdeps)
  set(code "#pragma once\n#include <tuple>\n#include <utility>\n#include <onika/omp/ompt_interface.h>\nnamespace onika\n{\nnamespace task\n{\n")
  set(code "${code}  template<int... Is> using iseq = std::integer_sequence<int,Is...>;\n")
  set(code "${code}  template< class ArgsOrdering , class ArgsTuple > struct OpenMPLambdaTaskScheduler;\n\n")

  set(code "${code}#ifdef ONIKA_ENABLE_TASK_PROFILING\n")
  set(code "${code}#define __ONIKA_TASK_PROFILE_BEGIN onika_ompt_begin_task(tag);\n")
  set(code "${code}#define __ONIKA_TASK_PROFILE_END onika_ompt_end_task(tag);\n")
  set(code "${code}#else\n")
  set(code "${code}#define __ONIKA_TASK_PROFILE_BEGIN /**/\n")
  set(code "${code}#define __ONIKA_TASK_PROFILE_END /**/\n")
  set(code "${code}#endif\n")

#  set(code "${code}#ifndef __ONIKA_TASK_SCHED_DBG\n")
#  set(code "${code}#define __ONIKA_TASK_SCHED_DBG /**/\n")
#  set(code "${code}#endif\n")

  set(code "${code}#ifndef __ONIKA_TASK_BEGIN_DBG\n")
  set(code "${code}#define __ONIKA_TASK_BEGIN_DBG /**/\n")
  set(code "${code}#endif\n")

  set(code "${code}#ifndef __ONIKA_TASK_END_DBG\n")
  set(code "${code}#define __ONIKA_TASK_END_DBG /**/\n")
  set(code "${code}#endif\n")

  set(code "${code}#pragma GCC diagnostic push\n")
  set(code "${code}#pragma GCC diagnostic ignored \"-Wunused-variable\"\n")   
   
  foreach(ndeps RANGE ${maxdeps})
    math(EXPR ndeps_1 "${ndeps}-1")
    set(code "${code}  // specializations for ${ndeps} arguments\n")

    foreach(nindeps RANGE ${ndeps})
      math(EXPR noutdeps "${ndeps}-${nindeps}")
      math(EXPR noutdeps_1 "${noutdeps}-1")
      math(EXPR nindeps_1 "${nindeps}-1")

      set(code "${code}  template<")
      if(${ndeps} GREATER 0)
        foreach(i RANGE ${ndeps_1})
          if(${i} GREATER 0)
            set(code "${code},")
          endif()
          set(code "${code}int I${i},typename P${i}")
        endforeach()
      endif()
      set(code "${code}> struct OpenMPLambdaTaskScheduler< iseq<${nindeps}")
      if(${ndeps} GREATER 0)
        foreach(i RANGE ${ndeps_1})
          set(code "${code},I${i}")
        endforeach()
      endif()
      set(code "${code}> , std::tuple<")
      if(${ndeps} GREATER 0)
        foreach(i RANGE ${ndeps_1})
          if(${i} GREATER 0)
            set(code "${code},")
          endif()
          set(code "${code}P${i}*")
        endforeach()
      endif()
      set(code "${code}> >\n  {\n    const char* m_tag=\"<unknown>\"; std::tuple<")
      if(${ndeps} GREATER 0)
        foreach(i RANGE ${ndeps_1})
          if(${i} GREATER 0)
            set(code "${code},")
          endif()
          set(code "${code}P${i}*")
        endforeach()
      endif()
      set(code "${code}> m_args;\n")
      set(code "${code}    template<class FuncT> inline void operator << (FuncT && func)\n    {\n      auto tag = m_tag; tag=(tag!=nullptr)?tag:\"<unknown>\";\n#     ifdef __ONIKA_TASK_SCHED_DBG\n      using plist_t=iseq<${nindeps}")

      if(${ndeps} GREATER 0)
        foreach(i RANGE ${ndeps_1})
          set(code "${code},I${i}")
        endforeach()
      endif()
      set(code "${code}>; __ONIKA_TASK_SCHED_DBG\n#     endif\n")

      if(${ndeps} GREATER 0)
        set(code "${code}      int")
        foreach(i RANGE ${ndeps_1})
          if(${i} GREATER 0)
            set(code "${code},")
          endif()
          set(code "${code} *_d${i} = (int*)(std::get<I${i}>(m_args))")
        endforeach()
        set(code "${code};")
      endif()

      if(${ndeps} GREATER 0)
        set(code "${code}\n     ")
        foreach(i RANGE ${ndeps_1})
          set(code "${code} auto a${i} = std::get<${i}>(m_args);")
        endforeach()
      endif()
      
      set(code "${code}\n#     pragma omp task default(none) firstprivate(tag,func")
      if(${ndeps} GREATER 0)
        foreach(i RANGE ${ndeps_1})
          set(code "${code},a${i}")
        endforeach()
      endif()
      set(code "${code})")

      if(${nindeps} GREATER 0)
        set(code "${code} depend(in:")
        foreach(depi RANGE ${nindeps_1})
          if(${depi} GREATER 0)
            set(code "${code},")
          endif()
          set(code "${code}_d${depi}[0]")
        endforeach()
        set(code "${code})")
      endif()

      if(${ndeps} GREATER ${nindeps})
        set(code "${code} depend(inout:")
        foreach(i RANGE ${noutdeps_1})
          if(${i} GREATER 0)
            set(code "${code},")
          endif()
          math(EXPR j "${i}+${nindeps}")
          set(code "${code}_d${j}[0]")
        endforeach()
        set(code "${code})")
      endif()

      set(code "${code} untied\n      { __ONIKA_TASK_BEGIN_DBG __ONIKA_TASK_PROFILE_BEGIN func(")
      if(${ndeps} GREATER 0)
        foreach(i RANGE ${ndeps_1})
          if(${i} GREATER 0)
            set(code "${code},")
          endif()
          set(code "${code}*a${i}")
        endforeach()
      endif()
      set(code "${code}); __ONIKA_TASK_PROFILE_END __ONIKA_TASK_END_DBG }\n    }\n  };\n")

    endforeach()
    set(code "${code}\n")
    
  endforeach()

  set(code "${code}\n}\n}\n#undef __ONIKA_TASK_PROFILE_BEGIN\n#undef __ONIKA_TASK_PROFILE_END\n")
  set(code "${code}#pragma GCC diagnostic pop\n")

  message(STATUS "generate file ${outfile}")
  file(WRITE ${outfile} "${code}")
endfunction()

