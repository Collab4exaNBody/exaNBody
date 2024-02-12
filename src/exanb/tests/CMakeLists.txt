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
# =======================
# ===   tests         ===
# =======================

option(USTAMP_ENABLE_TESTS "enable exanb tests" ON)

if(USTAMP_ENABLE_TESTS)

  enable_testing()

  add_executable(omp_nested_parallelism omp_nested_parallelism.cpp)
  target_compile_options(omp_nested_parallelism PRIVATE ${USTAMP_CXX_FLAGS})
  target_compile_definitions(omp_nested_parallelism PRIVATE ${USTAMP_COMPILE_DEFINITIONS})
  target_include_directories(omp_nested_parallelism PRIVATE ${USTAMP_INCLUDE_DIRS})
  target_link_libraries(omp_nested_parallelism ${USTAMP_CORE_LIBRARIES})
  xstamp_add_test(omp_nested_parallelism_1 ${CMAKE_CURRENT_BINARY_DIR}/omp_nested_parallelism 1)
  xstamp_add_test_par(omp_nested_parallelism_2 1 4 ${CMAKE_CURRENT_BINARY_DIR}/omp_nested_parallelism 2)
  xstamp_add_test_par(omp_nested_parallelism_3 1 4 ${CMAKE_CURRENT_BINARY_DIR}/omp_nested_parallelism 3)
  xstamp_add_test_par(omp_nested_parallelism_4 1 4 ${CMAKE_CURRENT_BINARY_DIR}/omp_nested_parallelism 4)

  add_executable(test_geometry test_geometry.cpp)
  target_compile_options(test_geometry PRIVATE ${USTAMP_CXX_FLAGS})
  target_compile_definitions(test_geometry PRIVATE ${USTAMP_COMPILE_DEFINITIONS})
  target_include_directories(test_geometry PRIVATE ${USTAMP_INCLUDE_DIRS})
  target_link_libraries(test_geometry ${USTAMP_CORE_LIBRARIES})
  xstamp_add_test(test_geometry ${CMAKE_CURRENT_BINARY_DIR}/test_geometry)

  add_executable(test_grid test_grid.cpp)
  target_compile_options(test_grid PRIVATE ${USTAMP_CXX_FLAGS})
  target_compile_definitions(test_grid PRIVATE ${USTAMP_COMPILE_DEFINITIONS})
  target_include_directories(test_grid PRIVATE ${USTAMP_INCLUDE_DIRS})
  target_link_libraries(test_grid ${USTAMP_CORE_LIBRARIES})
  add_test(test_grid test_grid)

  add_executable(benchmark_spin_mutex benchmark_spin_mutex.cpp)
  target_compile_options(benchmark_spin_mutex PRIVATE ${USTAMP_CXX_FLAGS})
  target_compile_definitions(benchmark_spin_mutex PRIVATE ${USTAMP_COMPILE_DEFINITIONS})
  target_include_directories(benchmark_spin_mutex PRIVATE ${USTAMP_INCLUDE_DIRS})
  target_link_libraries(benchmark_spin_mutex ${USTAMP_CORE_LIBRARIES})
  xstamp_add_test(benchmark_spin_mutex ${CMAKE_CURRENT_BINARY_DIR}/benchmark_spin_mutex 2 10000000)

  add_executable(test_yaml test_yaml.cpp)
  target_compile_options(test_yaml PRIVATE ${USTAMP_CXX_FLAGS})
  target_compile_definitions(test_yaml PRIVATE ${USTAMP_COMPILE_DEFINITIONS})
  target_include_directories(test_yaml PRIVATE ${USTAMP_INCLUDE_DIRS})
  target_link_libraries(test_yaml ${USTAMP_CORE_LIBRARIES})
  xstamp_add_test(test_yaml ${CMAKE_CURRENT_BINARY_DIR}/test_yaml)

  add_executable(test_defbox test_defbox.cpp)
  target_compile_options(test_defbox PRIVATE ${USTAMP_CXX_FLAGS})
  target_compile_definitions(test_defbox PRIVATE ${USTAMP_COMPILE_DEFINITIONS})
  target_include_directories(test_defbox PRIVATE ${USTAMP_INCLUDE_DIRS})
  target_link_libraries(test_defbox ${USTAMP_CORE_LIBRARIES})
  xstamp_add_test(test_defbox ${CMAKE_CURRENT_BINARY_DIR}/test_defbox)

  add_executable(test_wave test_wave_grid.cpp)
  target_compile_options(test_wave PRIVATE ${USTAMP_CXX_FLAGS})
  target_compile_definitions(test_wave PRIVATE ${USTAMP_COMPILE_DEFINITIONS})
  target_include_directories(test_wave PRIVATE ${USTAMP_INCLUDE_DIRS})
  target_link_libraries(test_wave ${USTAMP_CORE_LIBRARIES})
  xstamp_add_test(test_wave ${CMAKE_CURRENT_BINARY_DIR}/test_wave)

endif()

