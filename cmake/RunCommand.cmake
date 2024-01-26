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
find_program(SHELL_ENV_CMD env DOC "path to shell command 'env'")

# allow user to customize mpi commands to help debug
if(NOT MPIEXEC_EXECUTABLE_DBG)
  set(MPIEXEC_EXECUTABLE_DBG ${MPIEXEC_EXECUTABLE})
endif()
if(NOT MPIEXEC_NUMPROC_FLAG_DBG)
  set(MPIEXEC_NUMPROC_FLAG_DBG ${MPIEXEC_NUMPROC_FLAG})
endif()
if(NOT MPIEXEC_NUMCORE_FLAG_DBG)
  set(MPIEXEC_NUMCORE_FLAG_DBG ${MPIEXEC_NUMCORE_FLAG})
endif()
if(NOT MPIEXEC_PREFLAGS_DBG)
  set(MPIEXEC_PREFLAGS_DBG ${MPIEXEC_PREFLAGS})
endif()
if(NOT MPIEXEC_POSTFLAGS_DBG)
  set(MPIEXEC_POSTFLAGS_DBG ${MPIEXEC_POSTFLAGS})
endif()
if(NOT DEBUGGER_COMMAND)
  set(DEBUGGER_COMMAND gdb -ex run --args CACHE STRING "Debugger command prefix")
endif()

#build an encapsulating command that restricts the number of threads
function(MakeMultiThreadCommand CMDVAR RESULTCMDVAR NUMCORES)
  set(${RESULTCMDVAR} ${SHELL_ENV_CMD} OMP_NUM_THREADS=${NUMCORES} ${${CMDVAR}} PARENT_SCOPE)
endfunction()

# 2 optional additional arguments are NUMPROCS and NUMCORES
function(MakeRunCommand_internal ENABLE_DEBUGGER CMDVAR NUMPROCS NUMCORES ENABLE_PREALLOC USES_OMP RESULTCMDVAR)

  if(${NUMCORES} GREATER 0 AND ${USES_OMP})
    MakeMultiThreadCommand(${CMDVAR} MTCommand ${NUMCORES})
  else()
    set(MTCommand ${${CMDVAR}})
  endif()
  
  if(${NUMPROCS} GREATER 0 OR HOST_ALWAYS_USE_MPIRUN)

    if(ENABLE_DEBUGGER)
      set(MPIDBG _DBG)
      set(DBG_PREFIX ${DEBUGGER_COMMAND})
    endif()

    if(${NUMPROCS} LESS 1)
      set(NUMPROCS 1)
    endif()
    
    if(MPIEXEC_NUMCORE_FLAG AND ${NUMCORES} GREATER 0)
      set(MPIEXEC_NUMCORES_OPT ${MPIEXEC_NUMCORE_FLAG${MPIDBG}} ${NUMCORES})
    endif()

    if(ENABLE_PREALLOC)
      set(CMD_PREALLOC_FLAG ${MPIEXEC_PREALLOC_FLAG})
    endif()
    
    set(${RESULTCMDVAR} ${MPIEXEC_EXECUTABLE${MPIDBG}} ${MPIEXEC_NUMPROC_FLAG${MPIDBG}} ${NUMPROCS} ${MPIEXEC_NUMCORES_OPT} ${CMD_PREALLOC_FLAG} ${MPIEXEC_PREFLAGS${MPIDBG}} ${DBG_PREFIX} ${MTCommand} ${MPIEXEC_POSTFLAGS${MPIDBG}} PARENT_SCOPE)

  else()

    if(ENABLE_DEBUGGER)
      set(${RESULTCMDVAR} ${DEBUGGER_COMMAND} ${MTCommand} PARENT_SCOPE)
    else()
      set(${RESULTCMDVAR} ${MTCommand} PARENT_SCOPE)
    endif()

  endif()

endfunction()

function(MakeRunCommand CMDVAR NUMPROCS NUMCORES RESULTCMDVAR)
  MakeRunCommand_internal(OFF ${CMDVAR} ${NUMPROCS} ${NUMCORES} OFF ON ${RESULTCMDVAR})
  set(${RESULTCMDVAR} ${${RESULTCMDVAR}} PARENT_SCOPE)
endfunction()

function(MakeRunCommandNoOMP CMDVAR NUMPROCS NUMCORES RESULTCMDVAR)
  MakeRunCommand_internal(OFF ${CMDVAR} ${NUMPROCS} ${NUMCORES} OFF OFF ${RESULTCMDVAR})
  set(${RESULTCMDVAR} ${${RESULTCMDVAR}} PARENT_SCOPE)
endfunction()

function(MakePreallocRunCommand CMDVAR NUMPROCS NUMCORES RESULTCMDVAR)
  MakeRunCommand_internal(OFF ${CMDVAR} ${NUMPROCS} ${NUMCORES} ON ON ${RESULTCMDVAR})
  set(${RESULTCMDVAR} ${${RESULTCMDVAR}} PARENT_SCOPE)
endfunction()

function(MakeDebugRunCommand CMDVAR NUMPROCS NUMCORES RESULTCMDVAR)
  MakeRunCommand_internal(ON ${CMDVAR} ${NUMPROCS} ${NUMCORES} OFF ON ${RESULTCMDVAR})
  set(${RESULTCMDVAR} ${${RESULTCMDVAR}} PARENT_SCOPE)
endfunction()

