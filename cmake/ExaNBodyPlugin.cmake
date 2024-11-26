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

    file(GLOB ${PluginName}_SCRS_CXX ${dirname}/*.cpp)
    file(GLOB ${PluginName}_SCRS_CU ${dirname}/*.cu)
    set(${PluginName}_SCRS ${${PluginName}_SCRS_CXX} ${${PluginName}_SCRS_CU})

    if(IS_DIRECTORY ${dirname}/lib)
      file(GLOB ${PluginName}_SCRS_LIB_CXX ${dirname}/lib/*.cpp)
      file(GLOB ${PluginName}_SCRS_LIB_CU ${dirname}/lib/*.cu)
      set(${PluginName}_SCRS_LIB ${${PluginName}_SCRS_LIB_CXX} ${${PluginName}_SCRS_LIB_CU})
    endif()

    if(NOT ONIKA_BUILD_CUDA)
      set_source_files_properties(${${PluginName}_SCRS_CU} PROPERTIES LANGUAGE CXX)
      set_source_files_properties(${${PluginName}_SCRS_LIB_CU} PROPERTIES LANGUAGE CXX)
    endif()
    
    list(APPEND ${PluginName}_SCRS  ${${PluginName}_EXTERNAL_SRCS})
    include(${dirname}/${PluginName}.cmake OPTIONAL RESULT_VARIABLE ${PluginName}_CUSTOM_CMAKE)

    unset(PLUGIN_CORE_LIBS)
    if(NOT ${PluginName}_NO_CORE_LIBS)
      set(PLUGIN_CORE_LIBS ${XNB_CORE_LIBS})
    endif()

    set(${PluginName}_SHARED_LIB ${PluginName})
    if(${PluginName}_SCRS_LIB)
      # message(STATUS "Plugin ${PluginName} contains a lib : ${${PluginName}_SCRS_LIB}")
      add_library(${${PluginName}_SHARED_LIB} SHARED ${${PluginName}_SCRS_LIB})
      target_compile_options(${${PluginName}_SHARED_LIB} PRIVATE ${USTAMP_CXX_FLAGS} ${${PluginName}_COMPILE_OPTIONS})
      target_compile_definitions(${${PluginName}_SHARED_LIB} PRIVATE ${XNB_COMPILE_DEFINITIONS} PUBLIC ${${PluginName}_COMPILE_DEFINITIONS})
      target_include_directories(${${PluginName}_SHARED_LIB} PRIVATE ${dirname}/lib ${XNB_INCLUDE_DIRS} PUBLIC ${${PluginName}_INCLUDE_DIRS} ${dirname}/include)
      target_link_directories(${${PluginName}_SHARED_LIB} PUBLIC ${XNB_LIBRARY_DIRS} ${${PluginName}_LINK_DIRECTORIES})
      target_link_libraries(${${PluginName}_SHARED_LIB} ${PLUGIN_CORE_LIBS} ${${PluginName}_LINK_LIBRARIES})
      install(TARGETS ${${PluginName}_SHARED_LIB} DESTINATION lib)
    elseif(EXISTS ${dirname}/include)
      # message(STATUS "plugin ${PluginName} has no lib sources but has includes, create interface lib")
      add_library(${${PluginName}_SHARED_LIB} INTERFACE)
      target_include_directories(${${PluginName}_SHARED_LIB} INTERFACE ${dirname}/include)
      target_compile_definitions(${${PluginName}_SHARED_LIB} INTERFACE ${${PluginName}_COMPILE_DEFINITIONS})
    else()
      unset(${PluginName}_SHARED_LIB)
    endif()
        
#    message(STATUS "Plugin ${PluginName}: HOST ${${PluginName}_SCRS} , target=${${PluginName}_DEPS} CUDA ${${PluginName}_SCRS_CU} , target=${${PluginName}_DEPS_CU}")
    if(${PluginName}_SCRS)
      set(${PluginName}_PLUGIN_LIB ${PluginName}Plugin)
      add_library(${${PluginName}_PLUGIN_LIB} SHARED ${${PluginName}_SCRS})
      target_include_directories(${${PluginName}_PLUGIN_LIB} PRIVATE ${CUDA_SAMPLES_INCLUDE_DIR} ${dirname} ${XNB_INCLUDE_DIRS} ${${PluginName}_INCLUDE_DIRS})
      target_compile_definitions(${${PluginName}_PLUGIN_LIB} PRIVATE ${XNB_COMPILE_DEFINITIONS} ${${PluginName}_COMPILE_DEFINITIONS})        
      if(ONIKA_ENABLE_HIP)
      	target_compile_options(${${PluginName}_PLUGIN_LIB} PRIVATE ${USTAMP_CXX_FLAGS} $<$<COMPILE_LANGUAGE:HIP>:${ONIKA_HIP_COMPILE_FLAGS}>)
      else()
      	target_compile_options(${${PluginName}_PLUGIN_LIB} PRIVATE ${USTAMP_CXX_FLAGS} $<$<COMPILE_LANGUAGE:CUDA>:${ONIKA_CUDA_COMPILE_FLAGS}>)
      endif()
      target_compile_features(${${PluginName}_PLUGIN_LIB} PRIVATE ${ONIKA_COMPILE_FEATURES})
      target_link_directories(${${PluginName}_PLUGIN_LIB} PRIVATE ${XNB_LIBRARY_DIRS} ${${PluginName}_LINK_DIRECTORIES})
      target_link_libraries(${${PluginName}_PLUGIN_LIB} PRIVATE ${PLUGIN_CORE_LIBS} ${${PluginName}_LINK_LIBRARIES} ${${PluginName}_SHARED_LIB})
      install(TARGETS ${${PluginName}_PLUGIN_LIB} DESTINATION lib)
      xstamp_register_plugin(${${PluginName}_PLUGIN_LIB})
    else()
      unset(${PluginName}_PLUGIN_LIB)
    endif()
    
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
  set(XstampV2PluginDBGenCommandBase ${ONIKA_RUN} ${LoadAllPluginsInputFile} --generate_plugins_db true --logging-debug true --nogpu true)
  MakeRunCommand(XstampV2PluginDBGenCommandBase 1 ${ONIKA_HOST_HW_CORES} XstampV2PluginDBGenCommand)
  #message(STATUS "gen db command = ${XstampV2PluginDBGenCommand}")
  #set(XstampV2PluginDBGenCommand ${USTAMP_APPS_DIR}/${XNB_APP_NAME} ${LoadAllPluginsInputFile} --generate_plugins_db true )
  add_custom_target(UpdatePluginDataBase COMMAND ${XstampV2PluginDBGenCommand} DEPENDS ${ONIKA_RUN} BYPRODUCTS ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/plugins_db.msp)
  get_property(XSTAMP_BUILT_PLUGINS GLOBAL PROPERTY GLOBAL_XSTAMP_BUILT_PLUGINS)
  install(FILES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/plugins_db.msp DESTINATION lib)
endmacro()

macro(xstamp_add_unit_tests)
  set(XStampV2UnitsTestsCommand ${ONIKA_RUN} ${LoadAllPluginsInputFile} --run_unit_tests true)
  AddTestWithDebugTarget(XStampV2UnitsTests XStampV2UnitsTestsCommand 1 ${ONIKA_HOST_HW_CORES})
endmacro()

