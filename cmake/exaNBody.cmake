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

macro(exaNBodyStartApplication)
  # =============================
  # === Application branding  ===
  # =============================
  set(XNB_APP_NAME ${CMAKE_PROJECT_NAME})
  set(XNB_APP_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)
  set(XNB_FIELD_SETS_HDR ${PROJECT_SOURCE_DIR}/include/exanb/field_sets.h)
  if(XNB_PRODUCT_VARIANT)
    include(${PROJECT_SOURCE_DIR}/addons/${XNB_PRODUCT_VARIANT}.cmake)
  endif()

  set(XNB_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/exaNBody)
  set(USTAMP_APPS_DIR ${CMAKE_CURRENT_BINARY_DIR})

  # ===========================
  # === CMake customization ===
  # ===========================
  get_filename_component(XNB_ROOT_DIR ${exaNBody_DIR} ABSOLUTE)
  message(STATUS "XNB_ROOT_DIR=${XNB_ROOT_DIR}")

  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Release RelWithDebInfo Debug)
  include(CMakeDependentOption)
  include(RunCommand)
  include(DebugTests)
  include(ExaStampFileGlob)
  include(AddRegressionTestDir)
  include(ConfigureGenerateVariants)
  include(ExaNBodyPlugin)

  SET(CMAKE_SKIP_BUILD_RPATH FALSE)
  SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
  SET(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
  SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

  # =================================
  # === CMake customization       ===
  # =================================
  if(NOT CCMAKE_COMMAND)
    get_filename_component(CCMAKE_BIN_DIR ${CMAKE_COMMAND} DIRECTORY)
    find_program(CCMAKE_COMMAND ccmake HINTS ${CCMAKE_BIN_DIR})
    if(NOT CCMAKE_COMMAND)
      set(CCMAKE_COMMAND "ccmake" CACHE STRING "ccmake command")
    endif()
    #  message(STATUS "ccmake : ${CCMAKE_COMMAND}")
  endif()

  # host platform name
  if(NOT HOST_OS)
    set(HOST_OS "Linux-generic" CACHE STRING "Compile target name")
  endif()

  # get host maximum number of threads
  if(NOT HOST_HW_THREADS)
    set(HOST_HW_THREADS 1 CACHE STRING "Host number of logical threads")
  endif()
  if(NOT HOST_HW_CORES)
    set(HOST_HW_CORES 1 CACHE STRING "Host number of physical cores")
  endif()

  if(NOT HOST_HW_PARTITION)
    set(HOST_HW_PARTITION localhost)
  endif()

  # CPU cores detection
  message(STATUS "Host hardware : partition ${HOST_HW_PARTITION} has ${HOST_HW_CORES} core(s) and ${HOST_HW_THREADS} thread(s)")

  # =======================================
  # === Third party tools and libraries ===
  # =======================================
  if(XSTAMP_THIRD_PARTY_TOOLS_ROOT)
    message(STATUS "Third party tools in ${XSTAMP_THIRD_PARTY_TOOLS_ROOT}")
    if(NOT ZOLTAN_DIR)
      set(ZOLTAN_DIR ${XSTAMP_THIRD_PARTY_TOOLS_ROOT}/zoltan-3.83)
    endif()
  endif()

  if(MPI_CXX_INCLUDE_PATH AND MPI_CXX_LIBRARIES)
    message(STATUS "skip find_package(MPI REQUIRED)")
    message(STATUS "MPI_CXX_INCLUDE_PATH = ${MPI_CXX_INCLUDE_PATH}")
    message(STATUS "MPI_CXX_LIBRARIES    = ${MPI_CXX_LIBRARIES}")
    set(MPI_CXX_FOUND ON)
  else()
    find_package(MPI REQUIRED)
  endif()

  option(EXASTAMP_USE_ZOLTAN "Use Zoltan partitioner for load balancing" OFF)
  if(EXASTAMP_USE_ZOLTAN)
    set(EXASTAMP_ZOLTAN_DEFINITIONS ${EXASTAMP_ZOLTAN_DEFINITIONS} -D__use_lib_zoltan=1)
    option(EXASTAMP_USE_SYSTEM_ZOLTAN "Use system provided Zoltan" ON)
    if(EXASTAMP_USE_SYSTEM_ZOLTAN)
      find_package(Zoltan)
      set(EXASTAMP_ZOLTAN_LIBRARIES ${EXASTAMP_ZOLTAN_LIBRARIES} ${Zoltan_LIBRARIES})
      set(EXASTAMP_ZOLTAN_INCLUDE_DIRS ${EXASTAMP_ZOLTAN_INCLUDE_DIRS} ${Zoltan_INCLUDE_DIRS})
    else()
      option(EXASTAMP_USE_METIS "Use Metis partitioner with Zoltan" OFF)
      if(EXASTAMP_USE_METIS)
	      find_package(METIS REQUIRED)
	      set(EXASTAMP_ZOLTAN_DEFINITIONS ${EXASTAMP_ZOLTAN_DEFINITIONS} -D__use_lib_metis=1)
	      set(EXASTAMP_ZOLTAN_INCLUDE_DIRS ${EXASTAMP_ZOLTAN_INCLUDE_DIRS} ${METIS_INCLUDE_DIRS})
	      set(EXASTAMP_ZOLTAN_LIBRARIES ${EXASTAMP_ZOLTAN_LIBRARIES} ${METIS_LIBRARIES})	
      endif()
      find_package(XSZoltan REQUIRED)
      set(EXASTAMP_ZOLTAN_INCLUDE_DIRS ${EXASTAMP_ZOLTAN_INCLUDE_DIRS} ${ZOLTAN_INCLUDE_DIRS})
      set(EXASTAMP_ZOLTAN_LIBRARIES ${EXASTAMP_ZOLTAN_LIBRARIES} ${ZOLTAN_LIBRARIES})
    endif()
  endif()

  # use some embedded third party tools
  add_subdirectory(${XNB_ROOT_DIR}/thirdparty ${CMAKE_CURRENT_BINARY_DIR}/thirdparty)

  # ======================================================
  # ============ compilation environment =================
  # ======================================================
  set(XSTAMP_COMPILE_PROCESSES ${HOST_HW_THREADS})
  if(${HOST_HW_THREADS} EQUAL ${HOST_HW_CORES})
    math(EXPR XSTAMP_COMPILE_PROCESSES "2*${HOST_HW_CORES}")
  endif()
  set(XSTAMP_COMPILE_PROJECT_COMMAND "make -j${XSTAMP_COMPILE_PROCESSES}")
  MakeRunCommandNoOMP(XSTAMP_COMPILE_PROJECT_COMMAND 1 ${HOST_HW_CORES} XSTAMP_COMPILE_PROJECT_COMMAND)
  set(XSTAMP_UPDATE_PLUGINS_COMMAND "make UpdatePluginDataBase")
  #  MakeRunCommandNoOMP(XSTAMP_UPDATE_PLUGINS_COMMAND 1 1 XSTAMP_UPDATE_PLUGINS_COMMAND)
  string(REPLACE ";" " " XSTAMP_COMPILE_PROJECT_COMMAND "source ${CMAKE_CURRENT_BINARY_DIR}/setup-env.sh && ${XSTAMP_COMPILE_PROJECT_COMMAND} && ${XSTAMP_UPDATE_PLUGINS_COMMAND}")

  set(XSTAMP_INSTALL_PROJECT_COMMAND make install)
  MakeRunCommand(XSTAMP_INSTALL_PROJECT_COMMAND 1 ${HOST_HW_CORES} XSTAMP_INSTALL_PROJECT_COMMAND)
  string(REPLACE ";" " " XSTAMP_INSTALL_PROJECT_COMMAND "source ${CMAKE_CURRENT_BINARY_DIR}/setup-env.sh && ${XSTAMP_INSTALL_PROJECT_COMMAND}")

  set(XSTAMP_TEST_PROJECT_COMMAND make CTEST_OUTPUT_ON_FAILURE=1 test)
  MakePreallocRunCommand(XSTAMP_TEST_PROJECT_COMMAND 4 8 XSTAMP_TEST_PROJECT_COMMAND)
  string(REPLACE ";" " " XSTAMP_TEST_PROJECT_COMMAND "source ${CMAKE_CURRENT_BINARY_DIR}/setup-env.sh && ${XSTAMP_TEST_PROJECT_COMMAND}")

  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/README.txt "\n\
  *************\n\
  *** Sources : ${PROJECT_SOURCE_DIR}\n\
  *** Build   : ${CMAKE_CURRENT_BINARY_DIR}\n\
  *** Install : ${CMAKE_INSTALL_PREFIX}\n\
  *************\n\
  \n\
  Customize configuration :\n\
  source ${CMAKE_CURRENT_BINARY_DIR}/setup-env.sh && ${CCMAKE_COMMAND} .\n\
  \n\
  Build :\n\
  ${XSTAMP_COMPILE_PROJECT_COMMAND}\n\
  \n\
  Run tests :\n\
  ${XSTAMP_TEST_PROJECT_COMMAND}\n\
  \n\
  Install :\n\
  ${XSTAMP_INSTALL_PROJECT_COMMAND}\n")

  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/setup-env.sh "${PROJECT_SETUP_ENV_COMMANDS}\ncd ${CMAKE_CURRENT_BINARY_DIR}\necho '--- Environment ready ---'\n[ -f README.txt ] && cat README.txt")

  # run script installation
  #export LD_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:\$LD_LIBRARY_PATH
  file(READ ${XNB_ROOT_DIR}/scripts/patch_library_path.sh PATCH_LIBRARY_PATH)
  set(APP_SETUP_PLUGIN_PATH "PLUGIN_PATH=${CMAKE_INSTALL_PREFIX}/lib\n${PATCH_LIBRARY_PATH}")
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${XNB_APP_NAME}-env.sh "${PROJECT_SETUP_ENV_COMMANDS}\n${APP_SETUP_PLUGIN_PATH}\n")
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${XNB_APP_NAME}-run.sh "#!/bin/bash\n${PROJECT_SETUP_ENV_COMMANDS}\n${APP_SETUP_PLUGIN_PATH}\n${CMAKE_INSTALL_PREFIX}/bin/${XNB_APP_NAME} \$*\n")
  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${XNB_APP_NAME}-run.sh DESTINATION bin)
  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${XNB_APP_NAME}-env.sh DESTINATION bin)

  # =================================
  # === Application configuration ===
  # =================================

  # global settings
  set(USTAMP_PLUGIN_DIR "${CMAKE_INSTALL_PREFIX}/lib")
  set(XNB_DEFAULT_CONFIG_FILE "main-config.msp")
  set(XNB_LOCAL_CONFIG_FILE ".build-config.msp")

  # set default maximum number of particle neighbors
  set(XSTAMP_MAX_PARTICLE_NEIGHBORS_DEFAULT "512" CACHE STRING "Maximum number of particle neighbors") 

  # where to produce output binaries
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

  # configuration files to install
  file(GLOB EXANB_APP_CONFIG_FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/config/*.msp)

  # create a marker file to identify when applications are run from build tree
  file(GENERATE OUTPUT ${USTAMP_APPS_DIR}/${XNB_LOCAL_CONFIG_FILE}
       CONTENT "configuration: { plugin_dir: '${CMAKE_LIBRARY_OUTPUT_DIRECTORY}' , config_dir: '${CMAKE_CURRENT_SOURCE_DIR}/data/config' }\n")

  # compile time definitions
  set(USTAMP_COMPILE_DEFINITIONS
    -DUSTAMP_VERSION="${EXASTAMP_VERSION}"
    -DUSTAMP_PLUGIN_DIR="${USTAMP_PLUGIN_DIR}"
    -DXNB_DEFAULT_CONFIG_FILE="${XNB_DEFAULT_CONFIG_FILE}"
    -DXNB_LOCAL_CONFIG_FILE="${XNB_LOCAL_CONFIG_FILE}"
    -DXSTAMP_ADVISED_HW_THREADS=${HOST_HW_THREADS}
    -DXSTAMP_MAX_PARTICLE_NEIGHBORS_DEFAULT=${XSTAMP_MAX_PARTICLE_NEIGHBORS_DEFAULT}
    ${XSTAMP_OMP_FLAGS}
    ${XNB_APP_DEFINITIONS}
    ${KMP_ALIGNED_ALLOCATOR_DEFINITIONS}
    ${XSTAMP_TASK_PROFILING_DEFINITIONS}
    ${ONIKA_CUDA_COMPILE_DEFINITIONS}
    ${XSTAMP_AMR_ZCURVE_DEFINITIONS}
    )

  # performance tuning : number of stored pointers for each cell
  set(XSTAMP_FIELD_ARRAYS_STORE_COUNT 1 CACHE STRING "Per cell stored pointers")
  set(USTAMP_COMPILE_DEFINITIONS ${USTAMP_COMPILE_DEFINITIONS} -DXSTAMP_FIELD_ARRAYS_STORE_COUNT=${XSTAMP_FIELD_ARRAYS_STORE_COUNT})

  set(USTAMP_INCLUDE_DIRS
    ${XNB_APP_INCLUDE_DIRS}
    ${PROJECT_BINARY_DIR}/include
    ${YAML_CPP_INCLUDE_DIR}
    ${MPI_CXX_INCLUDE_PATH}
    ${TINYEXPR_INCLUDE_DIRS}
    ${NAIVEMATRIX_INCLUDE_DIRS}
    )

  set(USTAMP_CXX_FLAGS -Wall ${OpenMP_CXX_FLAGS})
  set(USTAMP_CORE_LIBRARIES
      exanbCore
      onika
      ${MPI_CXX_LIBRARIES}
      ${XNB_APP_LIBRARIES}
      )

  # Regression tests configuration 
  option(XNB_TEST_SEQ "Enable Sequential tests" OFF)
  option(XNB_TEST_MT "Enable MultiThread only tests" OFF)
  option(XNB_TEST_MPI "Enable MPI+Threads tests" ON)

  # build components, plugins and builtin tests
  add_subdirectory(${XNB_ROOT_DIR}/src ${XNB_BINARY_DIR})

  # install various files to install dir
  #install(DIRECTORY include/ustamp DESTINATION include)
  install(FILES ${EXANB_APP_CONFIG_FILES} DESTINATION share/config)
endmacro()


# ================ Plugin data base generation ================
macro(exaNBodyFinalizeApplication)
  xstamp_generate_all_plugins_input()
  xstamp_generate_plugin_database()
  xstamp_add_unit_tests()
endmacro()

