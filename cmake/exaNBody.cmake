
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

  #message(STATUS "CMAKE_SOURCE_DIR=${CMAKE_SOURCE_DIR}")
  #message(STATUS "CMAKE_CURRENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}")
  #message(STATUS "XNB_BINARY_DIR=${XNB_BINARY_DIR}")
  #message(STATUS "XNB_APP_NAME=${XNB_APP_NAME}")
  #message(STATUS "USTAMP_APPS_DIR=${USTAMP_APPS_DIR}")

  # ========================================
  # === Compiler toolchain configuration ===
  # ========================================
  # C++ standard
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS NO)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)
  endif()

  # ===========================
  # === CMake customization ===
  # ===========================
  set(XNB_ROOT_DIR ${exaNBody_DIR})
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
    if(NOT yaml-cpp_DIR)
      set(yaml-cpp_DIR ${XSTAMP_THIRD_PARTY_TOOLS_ROOT}/yaml-cpp/share/cmake/yaml-cpp)
    endif()
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
  message(STATUS "YAML ${yaml-cpp_DIR}")
  if(YAML_CPP_INCLUDE_DIR AND YAML_CPP_LIBRARIES)
    message(STATUS "YAML manually configured :")
    message(STATUS "\tYAML_CPP_INCLUDE_DIR=${YAML_CPP_INCLUDE_DIR}")
    message(STATUS "\tYAML_CPP_LIBRARIES=${YAML_CPP_LIBRARIES}")
  else()
    find_package(yaml-cpp REQUIRED)
  endif()

  find_package(OpenMP REQUIRED)

  if(OpenMP_CXX_VERSION GREATER_EQUAL 2.0)
    set(XSTAMP_OMP_FLAGS -DXSTAMP_OMP_VERSION=${OpenMP_CXX_VERSION})
  endif()
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(XSTAMP_OMP_NUM_THREADS_WORKAROUND_DEFAULT ON)
  else()
    set(XSTAMP_OMP_NUM_THREADS_WORKAROUND_DEFAULT OFF)
  endif()
  option(XSTAMP_OMP_NUM_THREADS_WORKAROUND "Enable OpenMP num_threads bug workaround" ${XSTAMP_OMP_NUM_THREADS_WORKAROUND_DEFAULT})
  if(XSTAMP_OMP_NUM_THREADS_WORKAROUND)
    set(XSTAMP_OMP_FLAGS ${XSTAMP_OMP_FLAGS} -DXSTAMP_OMP_NUM_THREADS_WORKAROUND=1)
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

  # embedded third party tools
  set(EXASTAMP_THIRDPARTY_DIR ${XNB_ROOT_DIR}/thirdparty)
  find_path(TKSPLINE_INCLUDE_DIRS tk/spline.h)

  set(BASEN_INCLUDE_DIRS ${EXASTAMP_THIRDPARTY_DIR}/base-n/include)
  set(TINYEXPR_INCLUDE_DIRS ${EXASTAMP_THIRDPARTY_DIR}/tinyexpr)
  add_library(tinyexpr STATIC ${EXASTAMP_THIRDPARTY_DIR}/tinyexpr/tinyexpr.c)
  target_compile_options(tinyexpr PRIVATE -fPIC)
  target_include_directories(tinyexpr PRIVATE ${TINYEXPR_INCLUDE_DIRS})

  # ===================================
  # ============ Cuda =================
  # ===================================
  option(XSTAMP_BUILD_CUDA "Enable Cuda Acceleration" OFF)
  set(ONIKA_FORCE_CUDA_OPTION ON)
  set(ONIKA_USE_CUDA ${XSTAMP_BUILD_CUDA})
  if(XSTAMP_BUILD_CUDA)
    set(XSTAMP_PLUGIN_HOST_FLAGS "" CACHE STRING "Plugin host specific flags")
    set(XSTAMP_PLUGIN_CUDA_FLAGS "" CACHE STRING "Plugin Cuda only flags")
    enable_language(CUDA)
    list(GET CMAKE_CUDA_ARCHITECTURES 0 XSTAMP_CUDA_ARCH_DEFAULT)
    set(XSTAMP_CUDA_ARCH ${XSTAMP_CUDA_ARCH_DEFAULT} CACHE STRING "Cuda architecture level")
    message(STATUS "Exastamp uses CUDA ${CMAKE_CUDA_COMPILER_VERSION} (arch ${XSTAMP_CUDA_ARCH})")
    #  set(XSTAMP_CUDA_ARCH_OPTS_FMT "--gpu-architecture=sm_%a" CACHE STRING "Cuda architecture compile options")
    #  string(REPLACE "%a" "${XSTAMP_CUDA_ARCH}" XSTAMP_CUDA_ARCH_OPTS "${XSTAMP_CUDA_ARCH_OPTS_FMT}")
    #  message(STATUS "Cuda arch opts = ${XSTAMP_CUDA_ARCH_OPTS}")
    set(XSTAMP_CUDA_COMPILE_DEFINITIONS -DXSTAMP_CUDA_VERSION=${CMAKE_CUDA_COMPILER_VERSION} -DXSTAMP_CUDA_ARCH=${XSTAMP_CUDA_ARCH})
    get_filename_component(CUDA_BIN ${CMAKE_CUDA_COMPILER} DIRECTORY)
    set(CUDA_ROOT ${CUDA_BIN}/..)
    #  set(CUDA_SAMPLES_INCLUDE_DIR ${CUDA_ROOT}/samples/common/inc)
    set(CUDA_INCLUDE_DIR ${CUDA_ROOT}/include)
    #  set(CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIR} ${CUDA_SAMPLES_INCLUDE_DIR})
    set(CUDA_LIBRARY_DIR ${CUDA_ROOT}/lib64)
    set(CUDA_LIBRARIES cudart)
  endif()


  # ======================================================
  # ============ compilation environment =================
  # ======================================================
  set(XSTAMP_COMPILE_PROCESSES ${HOST_HW_THREADS})
  if(${HOST_HW_THREADS} EQUAL ${HOST_HW_CORES})
    math(EXPR XSTAMP_COMPILE_PROCESSES "2*${HOST_HW_CORES}")
  endif()
  set(XSTAMP_COMPILE_PROJECT_COMMAND "make -j${XSTAMP_COMPILE_PROCESSES}")
  MakeRunCommandNoOMP(XSTAMP_COMPILE_PROJECT_COMMAND 1 ${HOST_HW_CORES} XSTAMP_COMPILE_PROJECT_COMMAND)
  string(REPLACE ";" " " XSTAMP_COMPILE_PROJECT_COMMAND "source ${CMAKE_CURRENT_BINARY_DIR}/setup-env.sh && ${XSTAMP_COMPILE_PROJECT_COMMAND} && make UpdatePluginDataBase")

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
  file(READ ${PROJECT_SOURCE_DIR}/scripts/patch_library_path.sh PATCH_LIBRARY_PATH)
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
  set(XNB_DEFAULT_CONFIG_FILE "config_${XNB_APP_NAME}.msp")
  set(XNB_LOCAL_CONFIG_FILE "${XNB_APP_NAME}_build.msp")
  set(XNB_CONFIG_DIR "${CMAKE_INSTALL_PREFIX}/share/config")

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
    -DXNB_CONFIG_DIR="${XNB_CONFIG_DIR}"
    -DXSTAMP_DEFAULT_DATA_DIRS=".:${EXASTAMP_TEST_DATA_DIR}:${CMAKE_INSTALL_PREFIX}/share/data"
    -DXSTAMP_ADVISED_HW_THREADS=${HOST_HW_THREADS}
    -DXSTAMP_MAX_PARTICLE_NEIGHBORS_DEFAULT=${XSTAMP_MAX_PARTICLE_NEIGHBORS_DEFAULT}
    ${XSTAMP_OMP_FLAGS}
    ${XNB_APP_DEFINITIONS}
    ${KMP_ALIGNED_ALLOCATOR_DEFINITIONS}
    ${XSTAMP_TASK_PROFILING_DEFINITIONS}
    ${XSTAMP_CUDA_COMPILE_DEFINITIONS}
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
    ${TKSPLINE_INCLUDE_DIRS}
    ${TINYEXPR_INCLUDE_DIRS}
    )

  set(USTAMP_CXX_FLAGS -Wall ${OpenMP_CXX_FLAGS})
  set(USTAMP_CORE_LIBRARIES
      exanbCore
      ${YAML_CPP_LIBRARIES}
      ${OpenMP_CXX_LIBRARIES}
      ${MPI_CXX_LIBRARIES}
      ${XNB_APP_LIBRARIES}
      tinyexpr)

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

