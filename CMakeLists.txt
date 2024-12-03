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
# ===============
# === Project ===
# ===============
cmake_minimum_required(VERSION 3.26)
project(exaNBody LANGUAGES C CXX)

find_package(onika REQUIRED)

set(XNB_APP_NAME ${CMAKE_PROJECT_NAME})
get_filename_component(XNB_ROOT_DIR ${CMAKE_INSTALL_PREFIX} ABSOLUTE)
set(XNB_CMAKE_PACKAGE ${CMAKE_CURRENT_BINARY_DIR}/exanbody-config.cmake)

file(WRITE ${XNB_CMAKE_PACKAGE} "set(onika_DIR ${onika_DIR})\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "find_package(onika REQUIRED)\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "set(XNB_ROOT_DIR ${XNB_ROOT_DIR})\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "set(XNB_APP_SUFFIX -\$\{CMAKE_PROJECT_NAME\})\n")

function(xnb_add_plugin PluginName dirname)

  # Backward compatibility trick : we used to automaticaly add exanbCore to link libraries
  if(NOT "${PluginName}" STREQUAL "exanbCore")
    list(APPEND ${PluginName}_LINK_LIBRARIES exanbCore)
  endif()

  # Tell Onika to duplicate type name definitions to exanb namespace, so that we don't have to rewite all the code by now
  list(APPEND ${PluginName}_COMPILE_DEFINITIONS ONIKA_LOG_EXPORT_NAMESPACE=exanb ONIKA_MATH_EXPORT_NAMESPACE=exanb ONIKA_SCG_EXPORT_NAMESPACE=exanb)

  onika_add_plugin(${PluginName} ${dirname})
  
  unset(GVFILES)
  foreach(SRC ${${PluginName}_SCRS})
    file(READ ${SRC} SRC_CONTENT)
    string(REGEX MATCH "OperatorNodeFactory::instance.*register_factory.*make_grid_variant_operator" SRCGV "${SRC_CONTENT}")
    if(SRCGV)
      list(APPEND GVFILES ${SRC})
    endif()
  endforeach()

  if(GVFILES OR ${PluginName}_SHARED_LIB)
    file(APPEND ${XNB_CMAKE_PACKAGE} "\n# Configuration of exaNBody's ${PluginName} component\n")
    if(${PluginName}_INCLUDE_DIRS)
      file(APPEND ${XNB_CMAKE_PACKAGE} "set(${PluginName}_INCLUDE_DIRS ${${PluginName}_INCLUDE_DIRS})\n")
    else()
      file(APPEND ${XNB_CMAKE_PACKAGE} "unset(${PluginName}_INCLUDE_DIRS)\n")
    endif()
    
    if(${PluginName}_COMPILE_DEFINITIONS)
      file(APPEND ${XNB_CMAKE_PACKAGE} "set(${PluginName}_COMPILE_DEFINITIONS ${${PluginName}_COMPILE_DEFINITIONS})\n")
    else()
      file(APPEND ${XNB_CMAKE_PACKAGE} "unset(${PluginName}_COMPILE_DEFINITIONS)\n")
    endif()
 
    if(${PluginName}_COMPILE_OPTIONS)
       file(APPEND ${XNB_CMAKE_PACKAGE} "set(${PluginName}_COMPILE_OPTIONS ${${PluginName}_COMPILE_OPTIONS})\n")
    else()
      file(APPEND ${XNB_CMAKE_PACKAGE} "unset(${PluginName}_COMPILE_OPTIONS)\n")
    endif()
 
    if(${PluginName}_COMPILE_FEATURES)
      file(APPEND ${XNB_CMAKE_PACKAGE} "set(${PluginName}_COMPILE_FEATURES ${${PluginName}_COMPILE_FEATURES})\n")
    else()
      file(APPEND ${XNB_CMAKE_PACKAGE} "unset(${PluginName}_COMPILE_FEATURES)\n")
    endif()
 
    if(${PluginName}_LINK_DIRECTORIES)
      file(APPEND ${XNB_CMAKE_PACKAGE} "set(${PluginName}_LINK_DIRECTORIES ${${PluginName}_LINK_DIRECTORIES})\n")
    else()
      file(APPEND ${XNB_CMAKE_PACKAGE} "unset(${PluginName}_LINK_DIRECTORIES)\n")
    endif()
 
    if(${PluginName}_LINK_LIBRARIES)
      file(APPEND ${XNB_CMAKE_PACKAGE} "set(${PluginName}_LINK_LIBRARIES ${${PluginName}_LINK_LIBRARIES})\n")
    else()
      file(APPEND ${XNB_CMAKE_PACKAGE} "unset(${PluginName}_LINK_LIBRARIES)\n")
    endif()
    
    if(${PluginName}_EXTERNAL_SRCS)
      file(APPEND ${XNB_CMAKE_PACKAGE} "set(${PluginName}_EXTERNAL_SRCS ${${PluginName}_EXTERNAL_SRCS})\n")
    else()
      file(APPEND ${XNB_CMAKE_PACKAGE} "unset(${PluginName}_EXTERNAL_SRCS)\n")
    endif()
    
    if(${PluginName}_CUSTOM_CMAKE)
      install(FILES ${${PluginName}_CUSTOM_CMAKE} DESTINATION ${XNB_ROOT_DIR}/redist/${PluginName})
      file(APPEND ${XNB_CMAKE_PACKAGE} "set(${PluginName}_CUSTOM_CMAKE ${XNB_ROOT_DIR}/redist/${PluginName}/${PluginName}.cmake)\n")
    else()
      file(APPEND ${XNB_CMAKE_PACKAGE} "unset(${PluginName}_CUSTOM_CMAKE)\n")
    endif()
  endif()

  if(GVFILES)
    message(STATUS "${PluginName} grid variant sources : ${GVFILES}")
    install(FILES ${GVFILES} DESTINATION ${XNB_ROOT_DIR}/redist/${PluginName})
    file(GLOB ${PluginName}_HDRS ${dirname}/*.h)
    if(${PluginName}_HDRS)
      install(FILES ${${PluginName}_HDRS} DESTINATION ${XNB_ROOT_DIR}/redist/${PluginName})
    endif()

    file(APPEND ${XNB_CMAKE_PACKAGE} "\n# Grid templated part of exaNBody's ${PluginName} component\n")    
    file(APPEND ${XNB_CMAKE_PACKAGE} "onika_add_plugin(${PluginName}\$\{XNB_APP_SUFFIX\} ${XNB_ROOT_DIR}/redist/${PluginName})\n")
  endif()

  if(${PluginName}_SHARED_LIB)
    file(APPEND ${XNB_CMAKE_PACKAGE} "\n# Import of exaNBody's ${PluginName} shared library\n")
    if(${PluginName}_SHARED_LIB_INTERFACE)
      file(APPEND ${XNB_CMAKE_PACKAGE} "add_library(${${PluginName}_SHARED_LIB} INTERFACE)\n")
      file(APPEND ${XNB_CMAKE_PACKAGE} "target_include_directories(${${PluginName}_SHARED_LIB} INTERFACE \$\{${PluginName}_INCLUDE_DIRS\} ${XNB_ROOT_DIR}/include)\n")
      file(APPEND ${XNB_CMAKE_PACKAGE} "target_compile_definitions(${${PluginName}_SHARED_LIB} INTERFACE \$\{${PluginName}_COMPILE_DEFINITIONS\})\n")
      file(APPEND ${XNB_CMAKE_PACKAGE} "target_compile_features(${${PluginName}_SHARED_LIB} INTERFACE \$\{${PluginName}_COMPILE_FEATURES\})\n")
    else()
      file(APPEND ${XNB_CMAKE_PACKAGE} "add_library(${${PluginName}_SHARED_LIB} IMPORTED GLOBAL)\n")
      file(APPEND ${XNB_CMAKE_PACKAGE} "target_include_directories(${${PluginName}_SHARED_LIB} PRIVATE \$\{ONIKA_INCLUDE_DIRS\} PUBLIC \$\{${PluginName}_INCLUDE_DIRS\} ${XNB_ROOT_DIR}/include)\n")
      file(APPEND ${XNB_CMAKE_PACKAGE} "target_compile_definitions(${${PluginName}_SHARED_LIB} PRIVATE \$\{ONIKA_COMPILE_DEFINITIONS\} PUBLIC \$\{${PluginName}_COMPILE_DEFINITIONS\})\n")
      file(APPEND ${XNB_CMAKE_PACKAGE} "target_compile_options(${${PluginName}_SHARED_LIB} PRIVATE \$\{ONIKA_COMPILE_OPTIONS\} \$<\$<COMPILE_LANGUAGE:CUDA>:\$\{ONIKA_CUDA_COMPILE_FLAGS\}> \$<\$<COMPILE_LANGUAGE:HIP>:\$\{ONIKA_HIP_COMPILE_FLAGS\}> \$\{${PluginName}_COMPILE_OPTIONS\})\n")
      file(APPEND ${XNB_CMAKE_PACKAGE} "target_compile_features(${${PluginName}_SHARED_LIB} PRIVATE \$\{ONIKA_COMPILE_FEATURES\} \$\{${PluginName}_COMPILE_FEATURES\})\n")
      file(APPEND ${XNB_CMAKE_PACKAGE} "target_link_directories(${${PluginName}_SHARED_LIB} PUBLIC \$\{ONIKA_LIBRARY_DIRS\} \$\{${PluginName}_LINK_DIRECTORIES\})\n")
      file(APPEND ${XNB_CMAKE_PACKAGE} "target_link_libraries(${${PluginName}_SHARED_LIB} \$\{ONIKA_LIBRARIES\} \$\{${PluginName}_LINK_LIBRARIES\})\n")
    endif()
  endif()


endfunction()

add_subdirectory(src)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/exanbody-config.cmake DESTINATION ${XNB_ROOT_DIR})
#install(DIRECTORY exanb cmake thirdparty scripts DESTINATION ${CMAKE_INSTALL_PREFIX})

