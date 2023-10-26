
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

