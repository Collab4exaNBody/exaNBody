function(AddRegressionTestDir REGRESSION_DIR)
  file(GLOB REGRESSION_TESTS RELATIVE ${REGRESSION_DIR} ${REGRESSION_DIR}/*)
  set(USTAMP_REGRESSION_TEST_LIST)
  foreach(testcase ${REGRESSION_TESTS})
    install(DIRECTORY ${REGRESSION_DIR}/${testcase} DESTINATION share/examples)

    file(GLOB testruns RELATIVE ${REGRESSION_DIR}/${testcase} ${REGRESSION_DIR}/${testcase}/*.msp)
    foreach(testrun ${testruns})
    
      string(REGEX REPLACE ".msp$" "" testvariant ${testrun})

      # scan file for special keywords on the first line : no-mpi no-seq no-mt
      file(STRINGS ${REGRESSION_DIR}/${testcase}/${testrun} test_lines)
      list(GET test_lines 0 test_header)
      string(FIND "${test_header}" "no-mpi" NO_MPI)
      string(FIND "${test_header}" "no-seq" NO_SEQ)
      string(FIND "${test_header}" "no-mt" NO_MT)    
      if(NO_MPI EQUAL -1)
        set(ENABLE_TEST_MPI ON)
      else()
        set(ENABLE_TEST_MPI OFF)
      endif()
      if(NO_SEQ EQUAL -1)
        set(ENABLE_TEST_SEQ ON)
      else()
        set(ENABLE_TEST_SEQ OFF)
      endif()
      if(NO_MT EQUAL -1)
        set(ENABLE_TEST_MT ON)
      else()
        set(ENABLE_TEST_MT OFF)
      endif()

      string(REGEX MATCHALL "enable-if-[^ ]*" ALL_COND_VAR "${test_header}")
      set(${testcase}_${testvariant}_ENABLED ON)
      
      foreach(COND_VAR ${ALL_COND_VAR})
        if(COND_VAR)
          string(REPLACE "enable-if-" "" COND_VAR "${COND_VAR}")
          string(REGEX MATCH "=.*" COND_VAR_VALUE "${COND_VAR}")
          if(COND_VAR_VALUE)
            string(REGEX REPLACE "=.*" "" COND_VAR "${COND_VAR}")
            string(REPLACE "=" "" COND_VAR_VALUE "${COND_VAR_VALUE}")
            if(NOT "${${COND_VAR}}" STREQUAL "${COND_VAR_VALUE}" )
              xstamp_message("Variable ${COND_VAR} (${${COND_VAR}}) doesn't match '${COND_VAR_VALUE}'")
              set(XSTAMP_COND_FALSE OFF)
              set(COND_VAR XSTAMP_COND_FALSE)
            endif()
          endif()
          
  #         message(STATUS "${testcase}/${testrun} condition ${COND_VAR} = ${${COND_VAR}}")
          if(NOT ${COND_VAR})
            if(${testcase}_${testvariant}_ENABLED)
              xstamp_message("disable test ${testcase}_${testvariant} (${COND_VAR}=${${COND_VAR}})")
              set(ENABLE_TEST_MPI OFF)
              set(ENABLE_TEST_SEQ OFF)
              set(ENABLE_TEST_MT OFF)
              set(${testcase}_${testvariant}_ENABLED OFF)
            endif()
          endif()
        endif()
      endforeach()

      set(MT_NTHREADS 4)
      string(REGEX MATCH "nthreads=[^ ]*" NTHREADS_VAR "${test_header}")
      if(NTHREADS_VAR)
        string(REPLACE "nthreads=" "" NTHREADS_VAR "${NTHREADS_VAR}")
        if("${NTHREADS_VAR}" STREQUAL "max")
          set(MT_NTHREADS ${HOST_HW_CORES})
        elseif(${NTHREADS_VAR} GREATER 1)
          set(MT_NTHREADS ${NTHREADS_VAR})
        endif()
        xstamp_message("${testcase}_${testvariant} : NTHREADS overriden to ${MT_NTHREADS}")
      endif()

      # qmessage(STATUS "${testcase}_${testvariant} : SEQ=${ENABLE_TEST_SEQ} MT=${ENABLE_TEST_MT} MPI=${ENABLE_TEST_MPI} HOST_ALWAYS_USE_MPIRUN=${HOST_ALWAYS_USE_MPIRUN}")

      list(APPEND USTAMP_REGRESSION_TEST_LIST "${testcase}_${testvariant}")

      if(USTAMP_TEST_SEQ AND ENABLE_TEST_SEQ)
        set(TestName ${XNB_APP}_${testcase}_${testvariant}_seq)
        set(${TestName}Command ${USTAMP_APPS_DIR}/${XNB_APP} ${REGRESSION_DIR}/${testcase}/${testrun} ${EXASTAMP_TEST_ADDITIONAL_ARGS})
        AddTestWithDebugTarget(${TestName} ${TestName}Command 1 1)
      endif()

      if(USTAMP_TEST_MT AND ENABLE_TEST_MT)
        set(TestName ${XNB_APP}_${testcase}_${testvariant}_mt)
        set(${TestName}Command ${USTAMP_APPS_DIR}/${XNB_APP} ${REGRESSION_DIR}/${testcase}/${testrun} ${EXASTAMP_TEST_ADDITIONAL_ARGS})
        AddTestWithDebugTarget(${TestName} ${TestName}Command 1 ${MT_NTHREADS})
      endif()

      if(USTAMP_TEST_PAR AND ENABLE_TEST_MPI)
        set(TestName ${XNB_APP}_${testcase}_${testvariant}_par)
        set(${TestName}Command ${USTAMP_APPS_DIR}/${XNB_APP} ${REGRESSION_DIR}/${testcase}/${testrun} ${EXASTAMP_TEST_ADDITIONAL_ARGS})
        AddTestWithDebugTarget(${TestName} ${TestName}Command 4 ${MT_NTHREADS})
      endif()

    endforeach()
  endforeach()

  list(LENGTH USTAMP_REGRESSION_TEST_LIST USTAMP_REGRESSION_TEST_COUNT)
  message(STATUS "found ${USTAMP_REGRESSION_TEST_COUNT} regression tests in ${REGRESSION_DIR}")
endfunction()

