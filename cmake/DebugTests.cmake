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

# ===========================
# === CTest customization ===
# ===========================
enable_testing()
function(AddTestWithDebugTarget TEST_NAME CMDVAR NUMPROCS NUMCORES)
  # message(STATUS "AddTestWithDebugTarget ${TEST_NAME} | ${CMDVAR} (${${CMDVAR}}) | ${NUMPROCS} | ${NUMCORES}")
  MakeRunCommand(${CMDVAR} ${NUMPROCS} ${NUMCORES} FULL_COMMAND)
  MakeDebugRunCommand(${CMDVAR} ${NUMPROCS} ${NUMCORES} FULL_COMMAND_DBG)
  #  message(STATUS "${TEST_NAME} | ${FULL_COMMAND} | ${FULL_COMMAND_DBG}")
  add_test(NAME ${TEST_NAME} COMMAND ${FULL_COMMAND})
  add_custom_target(${TEST_NAME} COMMAND ${FULL_COMMAND_DBG})
endfunction()

function(xstamp_add_test TestName)
      set(TestName xs_test_${TestName})
      list(REMOVE_AT ARGV 0)
      set(${TestName}Command ${ARGV})
#      message(STATUS "xstamp_add_test: ${TestName} : ${${TestName}Command}")
      AddTestWithDebugTarget(${TestName} ${TestName}Command -1 -1)
endfunction()

function(xstamp_add_test_par TestName NP NC)
      set(TestName xs_test_${TestName})
      list(REMOVE_AT ARGV 0 1 2)
      set(${TestName}Command ${ARGV})
#      message(STATUS "xstamp_add_test_par ${TestName} (parallel np=${NP} nc=${NC}) : ${${TestName}Command}")
      AddTestWithDebugTarget(${TestName} ${TestName}Command ${NP} ${NC})
endfunction()

