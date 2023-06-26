# ======================================
# === Parallel Data Movement library ===
# ======================================
cmake_minimum_required (VERSION 3.10)
project(onika VERSION 1.0 LANGUAGES CXX)

# =======================================
# === CMake & toolchain configuration ===
# =======================================
# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS NO)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(GenerateBenchmarkSOATL)

# =========================================
# === project options and configuration ===
# =========================================
# OpenMP with task and ompt support is appreciated
# packages and tools
if(NOT OpenMP_FOUND)
    find_package(OpenMP REQUIRED)
endif()
if(OpenMP_CXX_VERSION GREATER_EQUAL 5.0)
  option(ONIKA_HAVE_OPENMP_TOOLS "Eanble use of OMPT (OpenMP ver=${OpenMP_CXX_VERSION})" ON)
  option(ONIKA_HAVE_OPENMP_DETACH "Enable use of OMP 5.0 detach clause (OpenMP ver=${OpenMP_CXX_VERSION})" ON)
else()
  option(ONIKA_HAVE_OPENMP_TOOLS "Force use of OMPT (OpenMP ver=${OpenMP_CXX_VERSION})" OFF)
  option(ONIKA_HAVE_OPENMP_DETACH "Force use of OMP 5.0 detach clause (OpenMP ver=${OpenMP_CXX_VERSION})" OFF)
endif()

option(ONIKA_CU_ENABLE_KERNEL_BOUNDS "Use NVCC directive to restrict kernel launch bounds" OFF)
set(ONIKA_CU_MAX_THREADS_PER_BLOCK "128" CACHE STRING "Maximum number of threads per Cuda block")
set(ONIKA_CU_MIN_BLOCKS_PER_SM "6" CACHE STRING "Minimum number of blocks per SM")
if(ONIKA_CU_ENABLE_KERNEL_BOUNDS)
  set(ONIKA_DEFINITIONS ${ONIKA_DEFINITIONS} -DONIKA_CU_ENABLE_KERNEL_BOUNDS=1)
else()
  set(ONIKA_DEFINITIONS ${ONIKA_DEFINITIONS} -DONIKA_CU_ENABLE_KERNEL_BOUNDS=0)
endif()
set(ONIKA_DEFINITIONS ${ONIKA_DEFINITIONS} -DONIKA_CU_MAX_THREADS_PER_BLOCK=${ONIKA_CU_MAX_THREADS_PER_BLOCK} -DONIKA_CU_MIN_BLOCKS_PER_SM=${ONIKA_CU_MIN_BLOCKS_PER_SM})

if(ONIKA_HAVE_OPENMP_TOOLS)
  message(STATUS "OpenMP Tools enabled")
  set(ONIKA_DEFINITIONS -DONIKA_HAVE_OPENMP_TOOLS=1)
  option(ONIKA_ENABLE_TASK_PROFILING "Enable Onika task profiling feature" ON)
  if(ONIKA_ENABLE_TASK_PROFILING)
    set(ONIKA_DEFINITIONS ${ONIKA_DEFINITIONS} -DONIKA_ENABLE_TASK_PROFILING=1)
  endif()
endif()

if(ONIKA_HAVE_OPENMP_DETACH)
  message(STATUS "OpenMP detach enabled")
  set(ONIKA_DEFINITIONS ${ONIKA_DEFINITIONS} -DONIKA_HAVE_OPENMP_DETACH=1)
endif()

# configure maximum number of openmp task dependences
if(NOT ONIKA_OMP_MAX_DEPENDS_DEFAULT)
  set(ONIKA_OMP_MAX_DEPENDS_DEFAULT 10)
endif()
set(ONIKA_OMP_MAX_DEPENDS ${ONIKA_OMP_MAX_DEPENDS_DEFAULT} CACHE STRING "Maximum number of OpenMP task dynamic dependences")
set(ONIKA_DEFINITIONS ${ONIKA_DEFINITIONS} -DONIKA_OMP_MAX_DEPENDS=${ONIKA_OMP_MAX_DEPENDS})

# SOATL options
if(DEFINED SOATL_SIZE_32BITS_FORCED)
    message(STATUS "SOATL_SIZE_TYPE_32BITS forced to ${SOATL_SIZE_32BITS_FORCED}")
    set(SOATL_SIZE_TYPE_32BITS ${SOATL_SIZE_32BITS_FORCED})
else()
    option(SOATL_SIZE_TYPE_32BITS "SOATL use 32bits array size (instead of 64bits)" OFF)
endif()
if(SOATL_SIZE_TYPE_32BITS)
  message(STATUS "SOATL uses 32 bits size type")
  set(ONIKA_DEFINITIONS ${ONIKA_DEFINITIONS} -DSOATL_SIZE_TYPE_32BITS=1)
else()
  message(STATUS "SOATL uses 64 bits size type")
endif()

# CUDA options
if(NOT ONIKA_FORCE_CUDA_OPTION)
  option(ONIKA_USE_CUDA "Enable Cuda Acceleration" OFF)
endif()
if(ONIKA_USE_CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "Cuda version : ${CMAKE_CUDA_COMPILER_VERSION}")
    get_filename_component(CUDA_BIN ${CMAKE_CUDA_COMPILER} DIRECTORY)
    set(CUDA_ROOT ${CUDA_BIN}/..)
    set(CUDA_SAMPLES_INCLUDE_DIR ${CUDA_ROOT}/samples/common/inc)
    set(CUDA_INCLUDE_DIR ${CUDA_ROOT}/include)
    set(CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIR} ${CUDA_SAMPLES_INCLUDE_DIR})
    set(CUDA_LIBRARY_DIR ${CUDA_ROOT}/lib64)
    set(CUDA_LIBRARIES cudart)
  endif()
  set(ONIKA_DEFINITIONS ${ONIKA_DEFINITIONS} -DONIKA_CUDA_VERSION=${CMAKE_CUDA_COMPILER_VERSION})
  set(ONIKA_INCLUDE_DIRS ${ONIKA_INCLUDE_DIRS} ${CUDA_INCLUDE_DIR})
  set(ONIKA_LIBRARY_DIRS ${CUDA_LIBRARY_DIR})
  set(ONIKA_LIBRARIES ${ONIKA_LIBRARIES} ${CUDA_LIBRARIES})
  message(STATUS "onika uses CUDA ${CMAKE_CUDA_COMPILER_VERSION}")
endif()

# option to reset new allocations to zero
option(ONIKA_MEMORY_ZERO_ALLOC "Onika clears allocation to zero" OFF)
if(ONIKA_MEMORY_ZERO_ALLOC)
  set(ONIKA_DEFINITIONS ${ONIKA_DEFINITIONS} -DONIKA_MEMORY_ZERO_ALLOC=1)  
endif()

# generate static openmp task scheduler
include(GenerateOpenMPTaskLambda)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gen/include/onika/task)
GenerateOpenMPTaskLambda(${CMAKE_CURRENT_BINARY_DIR}/gen/include/onika/task/static_depend_lambda.h ${ONIKA_OMP_MAX_DEPENDS})
set(ONIKA_GEN_INC_DIRS ${CMAKE_CURRENT_BINARY_DIR}/gen/include)

# generate general purpose OpenMP dynamic depend dispatcher
include(GenOMPDynamicDependDispatch)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gen/src)
GenOMPDynamicDependDispatch(${CMAKE_CURRENT_BINARY_DIR}/gen/src/dynamic_depend_dispatch.cpp ${ONIKA_OMP_MAX_DEPENDS})
set(ONIKA_GEN_SRCS ${CMAKE_CURRENT_BINARY_DIR}/gen/src/dynamic_depend_dispatch.cpp)

# setup include dirs
set(ONIKA_INCLUDE_DIRS ${ONIKA_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include ${ONIKA_GEN_INC_DIRS})

# compile main lib
file(GLOB onika_SRCS src/*.cpp)
add_library(onika SHARED ${onika_SRCS} ${ONIKA_GEN_SRCS})
target_include_directories(onika PRIVATE ${ONIKA_INCLUDE_DIRS})
target_compile_definitions(onika PRIVATE ${ONIKA_DEFINITIONS})
target_compile_options(onika PRIVATE -Wall ${OpenMP_CXX_FLAGS})
target_link_directories(onika PUBLIC ${ONIKA_LIBRARY_DIRS})
target_link_libraries(onika ${ONIKA_LIBRARIES} ${OpenMP_CXX_LIBRARIES} dl)

# compile tools
add_executable(trace_convert src/tools/trace_convert.cpp)
target_compile_options(trace_convert PRIVATE -Wall ${OpenMP_CXX_FLAGS})
target_compile_definitions(trace_convert PRIVATE ${ONIKA_DEFINITIONS})
target_include_directories(trace_convert PRIVATE ${ONIKA_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIR})
target_link_libraries(trace_convert onika ${OpenMP_CXX_LIBRARIES} ${YAML_CPP_LIBRARIES})


# install files
install(TARGETS onika DESTINATION lib)
install(DIRECTORY include/onika DESTINATION include)
install(DIRECTORY ${ONIKA_GEN_INC_DIRS}/onika DESTINATION include)


# ==================
# === test suite ===
# ==================
option(ONIKA_ENABLE_TESTS "ONIKA enable tests" ON)
if(ONIKA_ENABLE_TESTS)

  file(GLOB test_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/src/tests ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/*.cpp)
  foreach(TESTSRC ${test_sources})
    get_filename_component(TESTNAME ${TESTSRC} NAME_WLE)
    # message(STATUS "onika found test ${TESTNAME}")
    add_executable(${TESTNAME} src/tests/${TESTNAME}.cpp)
    target_compile_options(${TESTNAME} PRIVATE -Wall ${OpenMP_CXX_FLAGS})
    target_compile_definitions(${TESTNAME} PRIVATE ${ONIKA_DEFINITIONS})
    target_include_directories(${TESTNAME} PRIVATE ${ONIKA_INCLUDE_DIRS})
    target_link_libraries(${TESTNAME} onika ${OpenMP_CXX_LIBRARIES})
  endforeach()

  enable_testing()

  foreach(tn 1;2;3;4;5;6;7;8;9;10;11)
    #message(STATUS "omp_tasks test #${tn}")
    xstamp_add_test(omp_tasks_${tn} ${CMAKE_CURRENT_BINARY_DIR}/omp_tasks ${tn})
  endforeach()

#  xstamp_add_test(soatl_fa_pfa_aliasing ${CMAKE_CURRENT_BINARY_DIR}/soatl_fa_pfa_aliasing)
  xstamp_add_test(soatl_test1 ${CMAKE_CURRENT_BINARY_DIR}/soatltest 1000 0)
  xstamp_add_test(soatl_test2 ${CMAKE_CURRENT_BINARY_DIR}/soatltest 1000 34523452)
  xstamp_add_test(soatl_test3 ${CMAKE_CURRENT_BINARY_DIR}/soatltest 1000 1976)
  xstamp_add_test(soatl_test4 ${CMAKE_CURRENT_BINARY_DIR}/soatltest 1000 234234234)
#  xstamp_add_test(soatl_compute1 ${CMAKE_CURRENT_BINARY_DIR}/soatlcomputetest 1000 0)
#  xstamp_add_test(soatl_compute2 ${CMAKE_CURRENT_BINARY_DIR}/soatlcomputetest 1000 34523452)
#  xstamp_add_test(soatl_compute3 ${CMAKE_CURRENT_BINARY_DIR}/soatlcomputetest 1000 1976)
#  xstamp_add_test(soatl_compute4 ${CMAKE_CURRENT_BINARY_DIR}/soatlcomputetest 1000 234234234)
  xstamp_add_test(soatl_serialize ${CMAKE_CURRENT_BINARY_DIR}/soatlserializetest 10000)
  xstamp_add_test(soatl_tuple ${CMAKE_CURRENT_BINARY_DIR}/soatupletest 10)

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

endif()


# ================================================
# === for use when embedded in another project ===
# ================================================
if(NOT "${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
  set(ONIKA_LIBRARIES onika ${ONIKA_LIBRARIES} ${OpenMP_CXX_LIBRARIES} PARENT_SCOPE)
  set(ONIKA_INCLUDE_DIRS ${ONIKA_INCLUDE_DIRS} PARENT_SCOPE)
  set(ONIKA_DEFINITIONS ${ONIKA_DEFINITIONS} PARENT_SCOPE)
  set(ONIKA_FOUND TRUE PARENT_SCOPE)
endif()

