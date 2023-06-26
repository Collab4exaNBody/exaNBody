# benchmark generate macro
macro(GenerateBenchmark A C DPS SIMD OMPTOGGLE)
  if("${DPS}" STREQUAL "d")
    set(DP 1)
  else()
    set(DP 0)
  endif()
  if(${SIMD})
    set(VEC 1)
    set(VECS "_vec")
  else()
    set(VEC 0)
    set(VECS "")
  endif()
  if(${OMPTOGGLE})
    set(OMPVAL 1)
    set(OMPS "_omp")
  else()
    set(OMPS "")
    set(OMPVAL 0)
  endif()
  set(SUFFIX ${A}_${C}_${DPS}${VECS}${OMPS})
  add_executable(soatlbenchmark_${SUFFIX} src/tests/benchmark.cpp)
  target_include_directories(soatlbenchmark_${SUFFIX} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${ONIKA_GEN_INC_DIRS})
  target_compile_options(soatlbenchmark_${SUFFIX} PUBLIC ${OpenMP_CXX_FLAGS})
  target_compile_definitions(soatlbenchmark_${SUFFIX} PUBLIC ${ONIKA_DEFINITIONS} -DTEST_USE_SIMD=${VEC} -DTEST_ALIGNMENT=${A} -DTEST_CHUNK_SIZE=${C} -DTEST_DOUBLE_PRECISION=${DP} -DTEST_ENABLE_OPENMP=${OMPVAL})
  target_link_libraries(soatlbenchmark_${SUFFIX} onika ${OpenMP_CXX_LIBRARIES})

  # add perf tests
  xstamp_add_test(soatlbenchmark_${SUFFIX}_hfa ${CMAKE_CURRENT_BINARY_DIR}/soatlbenchmark_${SUFFIX} 10000000)

  # assembly analysis
  if(SOATL_OBJDUMP)
    add_custom_target(vecreport_${SUFFIX}
                  COMMAND ${CMAKE_COMMAND} -DBINARY_FILE="$<TARGET_FILE:soatlbenchmark_${SUFFIX}>" -DSOATL_OBJDUMP="${SOATL_OBJDUMP}"
                  -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/vecreport.cmake
                  DEPENDS soatlbenchmark_${SUFFIX})
    add_dependencies(vecreport vecreport_${SUFFIX})
  endif()
endmacro()

