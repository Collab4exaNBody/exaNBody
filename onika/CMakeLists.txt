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
# ======================================
# === Parallel Data Movement library ===
# ======================================
cmake_minimum_required (VERSION 3.18)

# ========================================
# === Compiler toolchain configuration ===
# ========================================
# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# for export to onika-config file
set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}set(CMAKE_CXX_STANDARD ${CMAKE_CXX_STANDARD})\n")
set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}set(CMAKE_CXX_STANDARD_REQUIRED ${CMAKE_CXX_STANDARD_REQUIRED})\n")
set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}set(CMAKE_CXX_EXTENSIONS ${CMAKE_CXX_EXTENSIONS})\n")

# ==========================
# === project definition ===
# ==========================
project(onika VERSION 1.2 LANGUAGES C CXX)


# =======================================
# === CMake & toolchain configuration ===
# =======================================
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(RunCommand)
include(DebugTests)
include(GenerateBenchmarkSOATL)

SET(CMAKE_SKIP_BUILD_RPATH FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
SET(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(ONIKA_COMPILE_FEATURES cxx_std_17)

# ============================
# === OpenMP configuration ===
# ============================
# OpenMP with task and ompt support is appreciated
# packages and tools
find_package(OpenMP REQUIRED)

# for export to onika-config file)
set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}find_package(OpenMP REQUIRED)\n")

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  set(ONIKA_OMP_NUM_THREADS_WORKAROUND_DEFAULT ON)
else()
  set(ONIKA_OMP_NUM_THREADS_WORKAROUND_DEFAULT OFF)
endif()
option(ONIKA_OMP_NUM_THREADS_WORKAROUND "Enable OpenMP num_threads bug workaround" ${ONIKA_OMP_NUM_THREADS_WORKAROUND_DEFAULT})
if(ONIKA_OMP_NUM_THREADS_WORKAROUND)
  list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_OMP_NUM_THREADS_WORKAROUND=1)
endif()

if(OpenMP_CXX_VERSION GREATER_EQUAL 5.0)
  option(ONIKA_HAVE_OPENMP_TOOLS "Eanble use of OMPT (OpenMP ver=${OpenMP_CXX_VERSION})" ON)
  option(ONIKA_HAVE_OPENMP_DETACH "Enable use of OMP 5.0 detach clause (OpenMP ver=${OpenMP_CXX_VERSION})" ON)
else()
  option(ONIKA_HAVE_OPENMP_TOOLS "Force use of OMPT (OpenMP ver=${OpenMP_CXX_VERSION})" OFF)
  option(ONIKA_HAVE_OPENMP_DETACH "Force use of OMP 5.0 detach clause (OpenMP ver=${OpenMP_CXX_VERSION})" OFF)
endif()

if(ONIKA_HAVE_OPENMP_TOOLS)
  message(STATUS "OpenMP Tools enabled")
  list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_HAVE_OPENMP_TOOLS=1)
  option(ONIKA_ENABLE_TASK_PROFILING "Enable Onika task profiling feature" ON)
  if(ONIKA_ENABLE_TASK_PROFILING)
    list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_ENABLE_TASK_PROFILING=1)
  endif()
endif()

if(ONIKA_HAVE_OPENMP_DETACH)
  message(STATUS "OpenMP detach enabled")
  list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_HAVE_OPENMP_DETACH=1)
endif()

# configure maximum number of openmp task dependences
if(NOT ONIKA_OMP_MAX_DEPENDS_DEFAULT)
  set(ONIKA_OMP_MAX_DEPENDS_DEFAULT 10)
endif()
set(ONIKA_OMP_MAX_DEPENDS ${ONIKA_OMP_MAX_DEPENDS_DEFAULT} CACHE STRING "Maximum number of OpenMP task dynamic dependences")
list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_OMP_MAX_DEPENDS=${ONIKA_OMP_MAX_DEPENDS})

#configure default OpenMP scheduling
set(ONIKA_PARFOR_OMPSCHED_DEFAULT OMP_SCHED_GUIDED CACHE STRING "Set Onika's parallel_for scheduling : OMP_SCHED_STATIC OMP_SCHED_GUIDED OMP_SCHED_DYNAMIC")
set(ONIKA_BLKPARFOR_OMPSCHED_DEFAULT OMP_SCHED_GUIDED CACHE STRING "Set Onika's block_parallel_for scheduling : OMP_SCHED_STATIC OMP_SCHED_GUIDED OMP_SCHED_DYNAMIC")
list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_BLKPARFOR_OMPSCHED_DEFAULT=${ONIKA_BLKPARFOR_OMPSCHED_DEFAULT} ONIKA_PARFOR_OMPSCHED_DEFAULT=${ONIKA_PARFOR_OMPSCHED_DEFAULT})


# ====================================
# === local SMP node configuration ===
# ====================================
find_program(LSCPU_EXECUTABLE lscpu)
if(LSCPU_EXECUTABLE)
  execute_process(COMMAND ${LSCPU_EXECUTABLE} --parse=CPU,CORE OUTPUT_VARIABLE LSCPU_DATA)
  string(REGEX REPLACE "#[^\n]*\n" "" LSCPU_DATA "{${LSCPU_DATA}}")
  string(REPLACE "," ";" LSCPU_DATA "${LSCPU_DATA}")
  string(REPLACE "\n" ";" LSCPU_DATA "${LSCPU_DATA}")
  string(REPLACE ";}" "" LSCPU_DATA "${LSCPU_DATA}")
  string(REPLACE "{" "" LSCPU_DATA "${LSCPU_DATA}")
  list(GET LSCPU_DATA -2 ONIKA_HOST_HW_THREADS)
  list(GET LSCPU_DATA -1 ONIKA_HOST_HW_CORES)
  math(EXPR ONIKA_HOST_HW_THREADS "1+${ONIKA_HOST_HW_THREADS}")
  math(EXPR ONIKA_HOST_HW_CORES "1+${ONIKA_HOST_HW_CORES}")
  set(ONIKA_ADVISED_HW_THREADS ${ONIKA_HOST_HW_THREADS})
  message(STATUS "lscpu : threads=${ONIKA_HOST_HW_THREADS} cores=${ONIKA_HOST_HW_CORES}")
else()
  set(ONIKA_HOST_HW_CORES 4)
  set(ONIKA_HOST_HW_THREADS 8)
  set(ONIKA_ADVISED_HW_THREADS 128)
endif()
list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_ADVISED_HW_THREADS=${ONIKA_ADVISED_HW_THREADS})
# for export to onika-config file)
set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}set(ONIKA_HOST_HW_CORES ${ONIKA_HOST_HW_CORES})\n")
set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}set(ONIKA_HOST_HW_THREADS ${ONIKA_HOST_HW_THREADS})\n")

# ===================================
# ============ Cuda =================
# ===================================
option(ONIKA_BUILD_CUDA "Enable GPU Acceleration" OFF)
option(ONIKA_ENABLE_HIP "Use HIP instead of Cuda" OFF)
if(ONIKA_ENABLE_HIP)
  set(ONIKA_USE_CUDA OFF)
  set(ONIKA_USE_HIP ${ONIKA_BUILD_CUDA})
  if(ONIKA_BUILD_CUDA)
    if(NOT ROCM_INSTALL_ROOT)
      set(ROCM_INSTALL_ROOT "/opt/rocm")
    endif()
    list(APPEND CMAKE_MODULE_PATH "${ROCM_INSTALL_ROOT}/share/rocm/cmake")
    find_package(ROCM REQUIRED)
    enable_language(HIP)
    file(REAL_PATH "${ROCM_INSTALL_ROOT}" XNB_ROCM_ROOT)
    string(REGEX MATCH "[0-9].[0-9].[0-9]" XNB_ROCM_VERSION "${XNB_ROCM_ROOT}")
    if(XNB_ROCM_VERSION)
      set(ONIKA_HIP_VERSION "${XNB_ROCM_VERSION}")
    else()
      set(ONIKA_HIP_VERSION "${CMAKE_HIP_COMPILER_VERSION}")
    endif()
    set(ONIKA_CUDA_COMPILE_DEFINITIONS ONIKA_CUDA_VERSION=${CMAKE_HIP_COMPILER_VERSION} ONIKA_HIP_VERSION=${ONIKA_HIP_VERSION})

    # for export to onika-config file
    set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}list(APPEND CMAKE_MODULE_PATH \"${ROCM_INSTALL_ROOT}/share/rocm/cmake\")\n")
    set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}set(CMAKE_HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES})\n")
    set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}find_package(ROCM REQUIRED)\n")
    set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}enable_language(HIP)\n")
  endif()
else()
  set(ONIKA_USE_HIP OFF)
  set(ONIKA_USE_CUDA ${ONIKA_BUILD_CUDA})
  if(ONIKA_BUILD_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "Cuda target architecture(s)")
    find_package(CUDA REQUIRED)
    enable_language(CUDA)
    set(ONIKA_CUDA_COMPILE_DEFINITIONS ONIKA_CUDA_VERSION=${CMAKE_CUDA_COMPILER_VERSION})

    # for export to onika-config file
    set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}set(CUDA_SDK_ROOT_DIR ${CUDA_SDK_ROOT_DIR})\n")
    set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}set(CMAKE_CUDA_COMPILER ${CMAKE_CUDA_COMPILER})\n")
    set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})\n")
    set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}find_package(CUDA REQUIRED)\n")
    set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}enable_language(CUDA)\n")
  endif()
endif()

list(APPEND ONIKA_COMPILE_DEFINITIONS ${ONIKA_CUDA_COMPILE_DEFINITIONS})

if(ONIKA_USE_HIP)
  if(NOT CMAKE_HIP_COMPILER)
    enable_language(HIP)
  endif()
  set(ONIKA_HIP_COMPILE_FLAGS "-Werror=return-local-addr;-Werror=return-stack-address;-Werror=return-type" CACHE STRING "HIP extra flags")
  set(ONIKA_LIBRARIES ${ONIKA_LIBRARIES} roctx64)
  set(ONIKA_COMPILE_FEATURES ${ONIKA_COMPILE_FEATURES} hip_std_17)
  message(STATUS "Onika uses HIP ${CMAKE_HIP_COMPILER_VERSION}")
endif()

if(ONIKA_USE_CUDA)
  if(NOT CMAKE_CUDA_COMPILER_VERSION)
    find_package(CUDA REQUIRED)
    enable_language(CUDA)
    message(STATUS "Cuda version : ${CMAKE_CUDA_COMPILER_VERSION}")
  endif()
  get_filename_component(CUDA_BIN ${CMAKE_CUDA_COMPILER} DIRECTORY)
  set(CUDA_ROOT ${CUDA_BIN}/..)
  set(ONIKA_INCLUDE_DIRS ${ONIKA_INCLUDE_DIRS} ${CUDA_ROOT}/include ${CUDA_ROOT}/samples/common/inc)
  set(ONIKA_LIBRARY_DIRS ${CUDA_ROOT}/lib64)
  set(ONIKA_LIBRARIES ${ONIKA_LIBRARIES} cudart)
  set(ONIKA_CUDA_COMPILE_FLAGS "--Werror;cross-execution-space-call;--extended-lambda" CACHE STRING "Cuda extra flags")
  set(ONIKA_COMPILE_FEATURES ${ONIKA_COMPILE_FEATURES} cuda_std_17)
  message(STATUS "onika uses CUDA v${CMAKE_CUDA_COMPILER_VERSION} , arch=${CMAKE_CUDA_ARCHITECTURES} , root=${CUDA_ROOT}")
endif()


# Memory alignment settings
set(ONIKA_DEFAULT_ALIGNMENT "-1" CACHE STRING "Default field array alignment (-1 for autodetect)")
if(${ONIKA_DEFAULT_ALIGNMENT} GREATER 0)
  list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_DEFAULT_ALIGNMENT=${ONIKA_DEFAULT_ALIGNMENT})
endif()
set(ONIKA_DEFAULT_CHUNK_SIZE "-1" CACHE STRING "Default field array vector size (-1 for autodetect)")
if(${ONIKA_DEFAULT_CHUNK_SIZE} GREATER 0)
  list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_DEFAULT_CHUNK_SIZE=${ONIKA_DEFAULT_CHUNK_SIZE})
endif()
set(ONIKA_MINIMUM_CUDA_ALIGNMENT "-1" CACHE STRING "Default GPU memory alignment (-1 for autodetect)")
if(${ONIKA_MINIMUM_CUDA_ALIGNMENT} GREATER 0)
  list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_MINIMUM_CUDA_ALIGNMENT=${ONIKA_MINIMUM_CUDA_ALIGNMENT})
endif()

# Parallel execution configuration
option(ONIKA_CU_ENABLE_KERNEL_BOUNDS "Use NVCC directive to restrict kernel launch bounds" OFF)
set(ONIKA_TASKS_PER_CORE "4" CACHE STRING "Number of OpenMP tasks per thread")
set(ONIKA_CU_MAX_THREADS_PER_BLOCK "256" CACHE STRING "Maximum number of threads per Cuda block")
set(ONIKA_CU_MIN_BLOCKS_PER_SM "6" CACHE STRING "Minimum number of blocks per SM")
if(ONIKA_CU_ENABLE_KERNEL_BOUNDS)
  list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_CU_ENABLE_KERNEL_BOUNDS=1)
else()
  list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_CU_ENABLE_KERNEL_BOUNDS=0)
endif()
list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_CU_MAX_THREADS_PER_BLOCK=${ONIKA_CU_MAX_THREADS_PER_BLOCK} ONIKA_CU_MIN_BLOCKS_PER_SM=${ONIKA_CU_MIN_BLOCKS_PER_SM} ONIKA_TASKS_PER_CORE=${ONIKA_TASKS_PER_CORE})

# SOATL options
option(SOATL_SIZE_TYPE_32BITS "SOATL use 32bits array size (instead of 64bits)" ON)
if(SOATL_SIZE_TYPE_32BITS)
  message(STATUS "SOATL uses 32 bits size type")
  list(APPEND ONIKA_COMPILE_DEFINITIONS SOATL_SIZE_TYPE_32BITS=1)
else()
  message(STATUS "SOATL uses 64 bits size type")
endif()

# option to reset new allocations to zero
option(ONIKA_MEMORY_ZERO_ALLOC "Onika clears allocation to zero" OFF)
if(ONIKA_MEMORY_ZERO_ALLOC)
  list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_MEMORY_ZERO_ALLOC=1)  
endif()



# ============================
# === MPI configuration ===
# ============================
if(MPI_CXX_INCLUDE_PATH AND MPI_CXX_LIBRARIES)
  message(STATUS "Found MPI_CXX_INCLUDE_PATH and MPI_CXX_LIBRARIES variables, skip find_package(MPI REQUIRED)")
  message(STATUS "MPI_CXX_INCLUDE_PATH = ${MPI_CXX_INCLUDE_PATH}")
  message(STATUS "MPI_CXX_LIBRARIES    = ${MPI_CXX_LIBRARIES}")
  set(MPI_CXX_FOUND ON)
  # for export to onika-config file)
  if(MPIEXEC_EXECUTABLE)
    set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}set(MPIEXEC_EXECUTABLE ${MPIEXEC_EXECUTABLE})\n")
  endif()
  set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}set(MPI_CXX_INCLUDE_PATH ${MPI_CXX_INCLUDE_PATH})\n")
  set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}set(MPI_CXX_LIBRARIES ${MPI_CXX_LIBRARIES})\n")
  set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}set(MPI_CXX_FOUND ON)\n")
else()
  find_package(MPI REQUIRED)
  message(STATUS "MPIEXEC_EXECUTABLE = ${MPIEXEC_EXECUTABLE}")
  # for export to onika-config file)
  if(MPIEXEC_EXECUTABLE)
    set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}set(MPIEXEC_EXECUTABLE ${MPIEXEC_EXECUTABLE})\n")
  endif()
  set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}find_package(OpenMP REQUIRED)\n")
endif()
set(ONIKA_INCLUDE_DIRS ${ONIKA_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_PATH})



# ===================================
# ====== Global parameters ==========
# ===================================
if(ONIKA_INTERNAL_UNIT_SYSTEM)
  list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_INTERNAL_UNIT_SYSTEM=${ONIKA_INTERNAL_UNIT_SYSTEM})
endif()
set(ONIKA_DEFAULT_CONFIG_DIR "${CMAKE_INSTALL_PREFIX}/share/config")
set(ONIKA_DEFAULT_PLUGIN_DIR "${CMAKE_INSTALL_PREFIX}/lib")
list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_VERSION="${CMAKE_PROJECT_VERSION}" ONIKA_DEFAULT_CONFIG_DIR="${ONIKA_DEFAULT_CONFIG_DIR}" ONIKA_DEFAULT_DATA_DIRS=".:${CMAKE_INSTALL_PREFIX}/share/data" ONIKA_DEFAULT_PLUGIN_DIR="${ONIKA_DEFAULT_PLUGIN_DIR}")


# =====================================
# === third party and external libs ===
# =====================================
# tinyexpr
add_library(tinyexpr STATIC thirdparty/tinyexpr/tinyexpr.c)
target_compile_options(tinyexpr PRIVATE -fPIC)

# YAML
if(YAML_CPP_INCLUDE_DIR AND YAML_CPP_LIBRARIES)
  message(STATUS "YAML manually configured :")
  message(STATUS "\tYAML_CPP_INCLUDE_DIR = ${YAML_CPP_INCLUDE_DIR}")
  message(STATUS "\tYAML_CPP_LIBRARIES   = ${YAML_CPP_LIBRARIES}")
else()
  find_package(yaml-cpp REQUIRED)
endif()
if(yaml-cpp_DIR)
  message(STATUS "yaml-cpp_DIR = ${yaml-cpp_DIR}")
endif()


# =====================================
# === Onika lib build configuration ===
# =====================================

# setup include dirs
set(ONIKA_INCLUDE_DIRS ${ONIKA_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/tinyexpr)

# setup thirdparty and additional libraries used by onika
set(ONIKA_LIBRARIES ${ONIKA_LIBRARIES} ${YAML_CPP_LIBRARIES} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} dl)
set(ONIKA_PRIVATE_LIBRARIES tinyexpr)

# compile options 
set(ONIKA_COMPILE_OPTIONS -Wall ${OpenMP_CXX_FLAGS})

# compile main lib
file(GLOB onika_SRCS src/*.cpp src/*.cu)
add_library(onika SHARED ${onika_SRCS})
target_include_directories(onika PUBLIC ${ONIKA_INCLUDE_DIRS})
target_compile_definitions(onika PUBLIC ${ONIKA_COMPILE_DEFINITIONS})
target_compile_options(onika PUBLIC ${ONIKA_COMPILE_OPTIONS})
target_compile_features(onika PUBLIC ${ONIKA_COMPILE_FEATURES})
target_link_directories(onika PUBLIC ${ONIKA_LIBRARY_DIRS})
target_link_libraries(onika PUBLIC ${ONIKA_LIBRARIES} PRIVATE ${ONIKA_PRIVATE_LIBRARIES})

# Generate config file
list(APPEND ONIKA_INCLUDE_DIRS_USER ${CMAKE_INSTALL_PREFIX}/include ${ONIKA_INCLUDE_DIRS})
list(APPEND ONIKA_LIBRARY_DIRS_USER ${CMAKE_INSTALL_PREFIX}/lib ${ONIKA_LIBRARY_DIRS})
list(APPEND ONIKA_LIBRARIES_USER onika ${ONIKA_LIBRARIES})
set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}set(ONIKA_COMPILE_OPTIONS ${ONIKA_COMPILE_OPTIONS})\n")
set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}set(ONIKA_COMPILE_FEATURES ${ONIKA_COMPILE_FEATURES})\n")
set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}set(ONIKA_COMPILE_DEFINITIONS ${ONIKA_COMPILE_DEFINITIONS})\n")
set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}set(ONIKA_INCLUDE_DIRS ${ONIKA_INCLUDE_DIRS_USER})\n")
set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}set(ONIKA_LIBRARY_DIRS ${ONIKA_LIBRARY_DIRS_USER})\n")
set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}set(ONIKA_LIBRARIES ${ONIKA_LIBRARIES_USER})\n")
set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}set(ONIKA_BUILD_CUDA ${ONIKA_BUILD_CUDA})\n")
set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}set(ONIKA_ENABLE_HIP ${ONIKA_ENABLE_HIP})\n")
set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}set(ONIKA_RUN ${CMAKE_INSTALL_PREFIX}/bin/onikarun)\n")
set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}set(onika_DIR ${CMAKE_INSTALL_PREFIX})\n")
set(ONIKA_CONFIG_DATA "${ONIKA_CONFIG_DATA}set(onika_FOUND TRUE)\n")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/onika-config.cmake "${ONIKA_CONFIG_DATA}")

# compile tools and apps
file(GLOB app_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/apps ${CMAKE_CURRENT_SOURCE_DIR}/apps/*.cpp)
unset(ONIKA_APPLICATIONS)
foreach(APPSRC ${app_sources})
  get_filename_component(APPNAME ${APPSRC} NAME_WLE)
  message(STATUS "Onika found application ${APPNAME}")
  list(APPEND ONIKA_APPLICATIONS ${APPNAME})
  add_executable(${APPNAME} apps/${APPNAME}.cpp)
  target_compile_options(${APPNAME} PUBLIC ${ONIKA_COMPILE_OPTIONS})
  target_compile_definitions(${APPNAME} PUBLIC ${ONIKA_COMPILE_DEFINITIONS})
  target_include_directories(${APPNAME} PUBLIC ${ONIKA_INCLUDE_DIRS_USER})
  target_link_directories(${APPNAME} PUBLIC ${ONIKA_LIBRARY_DIRS_USER})
  target_link_libraries(${APPNAME} PUBLIC ${ONIKA_LIBRARIES_USER})
endforeach()

# install files
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/onika-config.cmake DESTINATION ${CMAKE_INSTALL_PREFIX})
install(TARGETS onika DESTINATION lib)
install(TARGETS ${ONIKA_APPLICATIONS} DESTINATION bin)
install(DIRECTORY include/onika DESTINATION include)

# ==================
# === test suite ===
# ==================
option(ONIKA_ENABLE_TESTS "ONIKA enable tests" ON)
if(ONIKA_ENABLE_TESTS)

  file(GLOB test_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/src/tests ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/*.cpp)
  foreach(TESTSRC ${test_sources})
    get_filename_component(TESTNAME ${TESTSRC} NAME_WLE)
    message(STATUS "onika found test ${TESTNAME}")
    add_executable(${TESTNAME} src/tests/${TESTNAME}.cpp)
    target_compile_options(${TESTNAME} PUBLIC ${ONIKA_COMPILE_OPTIONS})
    target_compile_definitions(${TESTNAME} PUBLIC ${ONIKA_COMPILE_DEFINITIONS})
    target_include_directories(${TESTNAME} PUBLIC ${ONIKA_INCLUDE_DIRS_USER})
    target_link_directories(${TESTNAME} PUBLIC ${ONIKA_LIBRARY_DIRS_USER})
    target_link_libraries(${TESTNAME} PUBLIC ${ONIKA_LIBRARIES_USER})
  endforeach()

  enable_testing()

  ######### OpenMP tests ##################
  foreach(tn 1;2;3;4;5;6;7;8;9;10;11)
    #message(STATUS "omp_tasks test #${tn}")
    onika_add_test(omp_tasks_${tn} ${CMAKE_CURRENT_BINARY_DIR}/omp_tasks ${tn})
  endforeach()



  ############ SOATL tests ##################
#  onika_add_test(soatl_fa_pfa_aliasing ${CMAKE_CURRENT_BINARY_DIR}/soatl_fa_pfa_aliasing)
  onika_add_test(soatl_test1 ${CMAKE_CURRENT_BINARY_DIR}/soatltest 1000 0)
  onika_add_test(soatl_test2 ${CMAKE_CURRENT_BINARY_DIR}/soatltest 1000 34523452)
  onika_add_test(soatl_test3 ${CMAKE_CURRENT_BINARY_DIR}/soatltest 1000 1976)
  onika_add_test(soatl_test4 ${CMAKE_CURRENT_BINARY_DIR}/soatltest 1000 234234234)
#  onika_add_test(soatl_compute1 ${CMAKE_CURRENT_BINARY_DIR}/soatlcomputetest 1000 0)
#  onika_add_test(soatl_compute2 ${CMAKE_CURRENT_BINARY_DIR}/soatlcomputetest 1000 34523452)
#  onika_add_test(soatl_compute3 ${CMAKE_CURRENT_BINARY_DIR}/soatlcomputetest 1000 1976)
#  onika_add_test(soatl_compute4 ${CMAKE_CURRENT_BINARY_DIR}/soatlcomputetest 1000 234234234)
  onika_add_test(soatl_serialize ${CMAKE_CURRENT_BINARY_DIR}/soatlserializetest 10000)
  onika_add_test(soatl_tuple ${CMAKE_CURRENT_BINARY_DIR}/soatupletest 10)

  option(SOATL_ENABLE_BENCHMARKS "SOATL benchmarks" ON)
  if(SOATL_ENABLE_BENCHMARKS)
    set(SOATL_BENCHMARKS_SIZE 10000000 CACHE STRING "SOATL benchmarks number of elements")

    # find objdump
    get_filename_component(BINUTILS_DIR ${CMAKE_LINKER} DIRECTORY)
    find_file(SOATL_OBJDUMP objdump HINTS ${BINUTILS_DIR})

    # create a vecreport target
    if(SOATL_OBJDUMP)
      add_custom_target(vecreport)
    endif()

    GenerateBenchmark(64 16 d ON OFF)
    GenerateBenchmark(64 16 d ON ON)
    GenerateBenchmark(64 16 f ON OFF)
    GenerateBenchmark(64 16 f ON ON)
    GenerateBenchmark(32 8 d ON OFF)
    GenerateBenchmark(32 8 f ON OFF)
    GenerateBenchmark(16 4 d ON OFF)
    GenerateBenchmark(16 4 f ON OFF)
    GenerateBenchmark(1 1 d OFF OFF)
    GenerateBenchmark(1 1 d OFF ON)
    GenerateBenchmark(1 1 f OFF OFF)
    GenerateBenchmark(1 1 f OFF ON)
  endif()



  ###### xs data move algorithm tests ########
  set(NUMPROCS_LIST 1 7 8)
  set(IDROTATION_LIST 0 1 77)
  set(IDSTART_LIST 0 1 1023)
  set(IDCOUNT_LIST 10 110 100000)
  set(NPERMS_LIST 10 1000)
  set(SEED_LIST 0 1976)

  foreach(NUMPROCS ${NUMPROCS_LIST})
      foreach(IDROTATION ${IDROTATION_LIST})
        foreach(IDSTART ${IDSTART_LIST})
          foreach(IDCOUNT ${IDCOUNT_LIST})
            math(EXPR IDEND ${IDSTART}+${IDCOUNT})
            onika_add_test_par(xsdatamove_simple_np${NUMPROCS}_rot${IDROTATION}_st${IDSTART}_ic${IDCOUNT} ${NUMPROCS} 1 ${CMAKE_CURRENT_BINARY_DIR}/xs_data_move_simple ${MPIEXEC_POSTFLAGS} ${IDSTART} ${IDEND} ${IDROTATION})
          endforeach()
        endforeach()
      endforeach()
  endforeach()

  onika_add_test_par(xsdatamove_simple_adhoc1 3 1 ${CMAKE_CURRENT_BINARY_DIR}/xs_data_move_simple ${MPIEXEC_POSTFLAGS} 0 20 3)
  onika_add_test_par(xsdatamove_simple_adhoc2 3 1 ${CMAKE_CURRENT_BINARY_DIR}/xs_data_move_simple ${MPIEXEC_POSTFLAGS} 0 107 3)

  foreach(NUMPROCS ${NUMPROCS_LIST})
    foreach(SEED ${SEED_LIST})
      foreach(IDCOUNT ${IDCOUNT_LIST})
        foreach(NPERMS ${NPERMS_LIST})
          onika_add_test_par(xsdatamove_random_s${SEED}_ic${IDCOUNT}_permm${NPERMS}_np${NUMPROCS} ${NUMPROCS} 1 ${CMAKE_CURRENT_BINARY_DIR}/xs_data_move_random ${MPIEXEC_POSTFLAGS} ${SEED} ${IDCOUNT} ${NPERMS})
        endforeach()
      endforeach()
    endforeach()
  endforeach()

endif()


