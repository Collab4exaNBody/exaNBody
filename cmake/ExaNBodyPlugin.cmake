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
set_property(GLOBAL PROPERTY GLOBAL_XSTAMP_BUILT_PLUGINS "")

function(xstamp_register_plugin PluginName)
  get_property(XSTAMP_BUILT_PLUGINS GLOBAL PROPERTY GLOBAL_XSTAMP_BUILT_PLUGINS)
  list(APPEND XSTAMP_BUILT_PLUGINS "${PluginName}")
  set_property(GLOBAL PROPERTY GLOBAL_XSTAMP_BUILT_PLUGINS ${XSTAMP_BUILT_PLUGINS})
endfunction()

function(xstamp_add_plugin PluginName dirname)
  set(BUILD_DEFAULT ON)

  if(DEFINED XSTAMP_BUILD_${PluginName}_DEFAULT)
    set(BUILD_DEFAULT ${XSTAMP_BUILD_${PluginName}_DEFAULT})
  endif()

  option(XSTAMP_BUILD_${PluginName} "Build ${PluginName}" ${BUILD_DEFAULT})

  if(XSTAMP_BUILD_${PluginName})    

    GenerateDirectorySourceFiles(${dirname} ${PluginName}_SCRS ${PluginName}_DEPS)
    if(${PluginName}_EXTERNAL_SRCS)
      foreach(srcfile ${${PluginName}_EXTERNAL_SRCS} ${${PluginName}_EXTERNAL_HDRS})
        get_filename_component(srcfile_basename ${srcfile} NAME)
        set(dstfile ${CMAKE_CURRENT_BINARY_DIR}/${srcfile_basename})
	# message(STATUS "${PluginName} : link ${srcfile} to ${dstfile}")
        file(CREATE_LINK ${srcfile} ${dstfile} SYMBOLIC)
        list(APPEND ${PluginName}_SCRS "${dstfile}")
      endforeach()
      foreach(srcfile ${${PluginName}_EXTERNAL_HDRS})
        get_filename_component(srcfile_basename ${srcfile} NAME)
        set(dstfile ${CMAKE_CURRENT_BINARY_DIR}/${srcfile_basename})
	# message(STATUS "${PluginName} : link ${srcfile} to ${dstfile}")
        file(CREATE_LINK ${srcfile} ${dstfile} SYMBOLIC)
      endforeach()
    endif()
#    message(STATUS "${PluginName}_SCRS = ${${PluginName}_SCRS}")

    include(${dirname}/${PluginName}.cmake OPTIONAL RESULT_VARIABLE ${PluginName}_CUSTOM_CMAKE)
#    if(${PluginName}_CUSTOM_CMAKE)
#      message(STATUS "found custom cmake : ${${PluginName}_CUSTOM_CMAKE}")
#    endif()

    unset(PLUGIN_CORE_LIBS)
    if(NOT ${PluginName}_NO_CORE_LIBS)
      set(PLUGIN_CORE_LIBS ${USTAMP_CORE_LIBRARIES})
    endif()

    if(${PluginName}_SCRS_LIB)
      # message(STATUS "Plugin ${PluginName} contains a lib : ${${PluginName}_SCRS_LIB}")
      add_library(${PluginName} SHARED ${${PluginName}_SCRS_LIB})
      target_compile_options(${PluginName} PRIVATE ${USTAMP_CXX_FLAGS} ${${PluginName}_COMPILE_OPTIONS})
      target_compile_definitions(${PluginName} PRIVATE ${USTAMP_COMPILE_DEFINITIONS} PUBLIC ${${PluginName}_COMPILE_DEFINITIONS})
      target_include_directories(${PluginName} PRIVATE ${dirname}/lib ${USTAMP_INCLUDE_DIRS} PUBLIC ${${PluginName}_INCLUDE_DIRS} ${dirname}/include)
      target_link_libraries(${PluginName} ${PLUGIN_CORE_LIBS} ${${PluginName}_LINK_LIBRARIES})
      if(${PluginName}_LINK_DIRECTORIES)
        target_link_directories(${PluginName} PUBLIC ${${PluginName}_LINK_DIRECTORIES})
      endif()
      set(LibTarget ${PluginName})
      install(TARGETS ${PluginName} DESTINATION lib)
    elseif(EXISTS ${dirname}/include)
      # message(STATUS "plugin ${PluginName} has no lib sources but has includes, create interface lib")
      add_library(${PluginName} INTERFACE)
      target_include_directories(${PluginName} INTERFACE ${dirname}/include)
      target_compile_definitions(${PluginName} INTERFACE ${${PluginName}_COMPILE_DEFINITIONS})
      set(LibTarget ${PluginName})
    endif()
        
#    message(STATUS "Plugin ${PluginName}: HOST ${${PluginName}_SCRS} , target=${${PluginName}_DEPS} CUDA ${${PluginName}_SCRS_CU} , target=${${PluginName}_DEPS_CU}")
    set(${PluginName}_HAS_CUDA OFF)
    if(${PluginName}_SCRS_CU)
      if(ONIKA_BUILD_CUDA)
        set(CudaKernelTarget ${PluginName}Plugin)
        add_library(${PluginName}Plugin SHARED ${${PluginName}_SCRS_CU} ${${PluginName}_SCRS})
        target_include_directories(${PluginName}Plugin PRIVATE ${CUDA_SAMPLES_INCLUDE_DIR} ${dirname} ${USTAMP_INCLUDE_DIRS} ${${PluginName}_INCLUDE_DIRS})
        target_compile_definitions(${PluginName}Plugin PRIVATE ${USTAMP_COMPILE_DEFINITIONS} ${${PluginName}_COMPILE_DEFINITIONS})        
        if(ONIKA_ENABLE_HIP)
          set_source_files_properties(${${PluginName}_SCRS_CU} PROPERTIES LANGUAGE HIP)
        	target_compile_options(${PluginName}Plugin PRIVATE ${USTAMP_CXX_FLAGS} $<$<COMPILE_LANGUAGE:HIP>:${ONIKA_HIP_COMPILE_FLAGS}>)
          target_compile_features(${PluginName}Plugin PRIVATE hip_std_14)
        else()
          set_source_files_properties(${${PluginName}_SCRS_CU} PROPERTIES LANGUAGE CUDA)
        	target_compile_options(${PluginName}Plugin PRIVATE ${USTAMP_CXX_FLAGS} $<$<COMPILE_LANGUAGE:CUDA>:${ONIKA_CUDA_COMPILE_FLAGS}>)
          target_compile_features(${PluginName}Plugin PRIVATE cuda_std_14)
        endif()
        #set_target_properties(${PluginName}Plugin PROPERTIES CUDA_RUNTIME_LIBRARY Shared CUDA_SEPARABLE_COMPILATION ON)
        target_link_libraries(${PluginName}Plugin PRIVATE ${PLUGIN_CORE_LIBS} ${${PluginName}_LINK_LIBRARIES} ${LibTarget})
        if(${PluginName}_LINK_DIRECTORIES)
          target_link_directories(${PluginName}Plugin PRIVATE ${${PluginName}_LINK_DIRECTORIES})
        endif()
      set(${PluginName}_HAS_CUDA ON)
      else()
        message(STATUS "Warning: Cuda disabled, ${PluginName} ignores following sources : ${${PluginName}_SCRS_CU}")
      endif()
    endif()

    if(NOT ${PluginName}_HAS_CUDA AND ${PluginName}_SCRS)
      add_library(${PluginName}Plugin SHARED ${${PluginName}_SCRS})
      target_compile_options(${PluginName}Plugin PRIVATE ${USTAMP_CXX_FLAGS} ${${PluginName}_COMPILE_OPTIONS})
      target_compile_definitions(${PluginName}Plugin PRIVATE ${USTAMP_COMPILE_DEFINITIONS} ${${PluginName}_COMPILE_DEFINITIONS})
      target_include_directories(${PluginName}Plugin PRIVATE ${dirname} ${USTAMP_INCLUDE_DIRS} ${${PluginName}_INCLUDE_DIRS})
      target_link_libraries(${PluginName}Plugin PRIVATE ${PLUGIN_CORE_LIBS} ${${PluginName}_LINK_LIBRARIES} ${LibTarget})
      if(${PluginName}_LINK_DIRECTORIES)
        target_link_directories(${PluginName}Plugin PRIVATE ${${PluginName}_LINK_DIRECTORIES})
      endif()
    endif()

    install(TARGETS ${PluginName}Plugin DESTINATION lib)
    xstamp_register_plugin(${PluginName}Plugin)
    
    if(EXISTS ${dirname}/regression)
      AddRegressionTestDir(${dirname}/regression)
    endif()

    if(EXISTS ${dirname}/tests)
      add_subdirectory(${dirname}/tests)
    endif()

  endif()
endfunction()

function(xstamp_make_all_plugins_input FileName)
  file(WRITE  ${FileName} "configuration:\n")
  file(APPEND ${FileName} "  +plugins:\n")
  get_property(XSTAMP_BUILT_PLUGINS GLOBAL PROPERTY GLOBAL_XSTAMP_BUILT_PLUGINS)
  foreach(PluginName ${XSTAMP_BUILT_PLUGINS})
    # message(STATUS "Plugin DB : add ${PluginName}")
    file(APPEND ${FileName} "    - ${PluginName}\n")
  endforeach()
  file(APPEND ${FileName} "  debug:\n")
  file(APPEND ${FileName} "    plugins: true\n")
endfunction()


macro(xstamp_generate_all_plugins_input)
  set(LoadAllPluginsInputFile ${CMAKE_CURRENT_BINARY_DIR}/gen_plugins_db.msp)
  xstamp_make_all_plugins_input(${LoadAllPluginsInputFile})
endmacro()

macro(xstamp_generate_plugin_database)
  set(XstampV2PluginDBGenCommandBase ${USTAMP_APPS_DIR}/${XNB_APP_NAME} ${LoadAllPluginsInputFile} --generate_plugins_db true --logging-debug true --nogpu true)
  MakeRunCommand(XstampV2PluginDBGenCommandBase 1 ${HOST_HW_CORES} XstampV2PluginDBGenCommand)
  #message(STATUS "gen db command = ${XstampV2PluginDBGenCommand}")
  #set(XstampV2PluginDBGenCommand ${USTAMP_APPS_DIR}/${XNB_APP_NAME} ${LoadAllPluginsInputFile} --generate_plugins_db true )
  add_custom_target(UpdatePluginDataBase ${XstampV2PluginDBGenCommand} BYPRODUCTS ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/plugins_db.msp)
  get_property(XSTAMP_BUILT_PLUGINS GLOBAL PROPERTY GLOBAL_XSTAMP_BUILT_PLUGINS)
  add_dependencies(UpdatePluginDataBase ${XNB_APP_NAME} ${XSTAMP_BUILT_PLUGINS})
  install(FILES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/plugins_db.msp DESTINATION lib)
endmacro()

macro(xstamp_add_unit_tests)
  set(XStampV2UnitsTestsCommand ${USTAMP_APPS_DIR}/${XNB_APP_NAME} ${LoadAllPluginsInputFile} --run_unit_tests true)
  AddTestWithDebugTarget(XStampV2UnitsTests XStampV2UnitsTestsCommand 1 ${HOST_HW_CORES})
endmacro()

