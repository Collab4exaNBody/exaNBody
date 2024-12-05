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
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Release RelWithDebInfo Debug)

# main dependency and configuation variables
set(XNB_APP_NAME ${CMAKE_PROJECT_NAME})
get_filename_component(XNB_ROOT_DIR ${CMAKE_INSTALL_PREFIX} ABSOLUTE)
set(XNB_CMAKE_PACKAGE ${CMAKE_CURRENT_BINARY_DIR}/exanbody-config.cmake)
find_package(onika REQUIRED)

string(TIMESTAMP XNB_BUILD_DATE "%Y-%m-%d %Hh%M:%S")
file(WRITE ${XNB_CMAKE_PACKAGE} "# exaNBody CMake package (generated on ${XNB_BUILD_DATE})\n\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "# Main package configuration\n")
if(onika_DIR)
  file(APPEND ${XNB_CMAKE_PACKAGE} "set(onika_DIR ${onika_DIR})\n")
endif()
file(APPEND ${XNB_CMAKE_PACKAGE} "find_package(onika REQUIRED)\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "set(XNB_ROOT_DIR ${XNB_ROOT_DIR})\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "set(XNB_APP_NAME \${CMAKE_PROJECT_NAME})\n")

# create a plugin function, wrapping onika_add_plugin so that we can add application specific global parameters
file(APPEND ${XNB_CMAKE_PACKAGE} "list(APPEND ONIKA_COMPILE_DEFINITIONS \${XNB_COMPILE_DEFINITIONS})\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "list(APPEND ONIKA_INCLUDE_DIRS \${XNB_INCLUDE_DIRECTORIES})\n")

# setup thirdparty base-n encoding lib and export it to cmake package
add_library(basen INTERFACE)
target_include_directories(basen INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/base-n DESTINATION include)
file(APPEND ${XNB_CMAKE_PACKAGE} "\n# base-n encoding library, needed by VTK binary output tools\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "add_library(basen INTERFACE)\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "target_include_directories(basen INTERFACE \${XNB_ROOT_DIR}/include)\n")

function(xnb_add_plugin PluginName dirname)

  # Backward compatibility trick : we used to automaticaly add exanbCore to link libraries
  if("${PluginName}" STREQUAL "exanbCore")
    # Tell Onika to duplicate type name definitions to exanb namespace, so that we don't have to rewite all the code by now
    list(APPEND ${PluginName}_COMPILE_DEFINITIONS ONIKA_PACKAGE_NAME=${XNB_APP_NAME} ONIKA_LOG_EXPORT_NAMESPACE=exanb ONIKA_MATH_EXPORT_NAMESPACE=exanb ONIKA_SCG_EXPORT_NAMESPACE=exanb)
  else()
    list(APPEND ${PluginName}_LINK_LIBRARIES exanbCore)
  endif()

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
    endif()    
    if(${PluginName}_COMPILE_DEFINITIONS)
      file(APPEND ${XNB_CMAKE_PACKAGE} "set(${PluginName}_COMPILE_DEFINITIONS ONIKA_PACKAGE_OVERRIDE=\${XNB_APP_NAME} ${${PluginName}_COMPILE_DEFINITIONS})\n")
    endif() 
    if(${PluginName}_COMPILE_OPTIONS)
       file(APPEND ${XNB_CMAKE_PACKAGE} "set(${PluginName}_COMPILE_OPTIONS ${${PluginName}_COMPILE_OPTIONS})\n")
    endif() 
    if(${PluginName}_COMPILE_FEATURES)
      file(APPEND ${XNB_CMAKE_PACKAGE} "set(${PluginName}_COMPILE_FEATURES ${${PluginName}_COMPILE_FEATURES})\n")
    endif() 
    if(${PluginName}_LINK_DIRECTORIES)
      file(APPEND ${XNB_CMAKE_PACKAGE} "set(${PluginName}_LINK_DIRECTORIES ${${PluginName}_LINK_DIRECTORIES})\n")
    endif() 
    if(${PluginName}_LINK_LIBRARIES)
      file(APPEND ${XNB_CMAKE_PACKAGE} "set(${PluginName}_LINK_LIBRARIES ${${PluginName}_LINK_LIBRARIES})\n")
    endif()    
    if(${PluginName}_EXTERNAL_SRCS)
      file(APPEND ${XNB_CMAKE_PACKAGE} "set(${PluginName}_EXTERNAL_SRCS ${${PluginName}_EXTERNAL_SRCS})\n")
    endif()    
    if(${PluginName}_CUSTOM_CMAKE)
      install(FILES ${${PluginName}_CUSTOM_CMAKE} DESTINATION ${XNB_ROOT_DIR}/redist/${PluginName})
      file(APPEND ${XNB_CMAKE_PACKAGE} "set(${PluginName}_CUSTOM_CMAKE ${XNB_ROOT_DIR}/redist/${PluginName}/${PluginName}.cmake)\n")
    endif()
  endif()

  if(${PluginName}_SHARED_LIB)
    file(APPEND ${XNB_CMAKE_PACKAGE} "\n# Import of exaNBody's ${PluginName} shared library\n")
    if(${PluginName}_SHARED_LIB_INTERFACE)
      #file(APPEND ${XNB_CMAKE_PACKAGE} "message(STATUS \"import interface library ${${PluginName}_SHARED_LIB}\")\n")
      file(APPEND ${XNB_CMAKE_PACKAGE} "add_library(${${PluginName}_SHARED_LIB} INTERFACE)\n")
      file(APPEND ${XNB_CMAKE_PACKAGE} "target_include_directories(${${PluginName}_SHARED_LIB} INTERFACE \${${PluginName}_INCLUDE_DIRS} ${XNB_ROOT_DIR}/include)\n")
      file(APPEND ${XNB_CMAKE_PACKAGE} "target_compile_definitions(${${PluginName}_SHARED_LIB} INTERFACE \${${PluginName}_COMPILE_DEFINITIONS})\n")
      file(APPEND ${XNB_CMAKE_PACKAGE} "target_compile_features(${${PluginName}_SHARED_LIB} INTERFACE \${${PluginName}_COMPILE_FEATURES})\n")
      file(APPEND ${XNB_CMAKE_PACKAGE} "target_link_directories(${${PluginName}_SHARED_LIB} INTERFACE \${ONIKA_LIBRARY_DIRS} \${${PluginName}_LINK_DIRECTORIES})\n")
      file(APPEND ${XNB_CMAKE_PACKAGE} "target_link_libraries(${${PluginName}_SHARED_LIB} INTERFACE \${ONIKA_LIBRARIES} \${${PluginName}_LINK_LIBRARIES})\n")
    else()
      #file(APPEND ${XNB_CMAKE_PACKAGE} "message(STATUS \"import shared library ${${PluginName}_SHARED_LIB}\")\n")
      file(APPEND ${XNB_CMAKE_PACKAGE} "add_library(${${PluginName}_SHARED_LIB} SHARED IMPORTED GLOBAL)\n")
      file(APPEND ${XNB_CMAKE_PACKAGE} "set_target_properties(${${PluginName}_SHARED_LIB} PROPERTIES IMPORTED_CONFIGURATIONS Release)\n")
      file(APPEND ${XNB_CMAKE_PACKAGE} "set_target_properties(${${PluginName}_SHARED_LIB} PROPERTIES IMPORTED_LOCATION_RELEASE ${CMAKE_INSTALL_PREFIX}/lib/lib${${PluginName}_SHARED_LIB}.so)\n")
      file(APPEND ${XNB_CMAKE_PACKAGE} "target_include_directories(${${PluginName}_SHARED_LIB} INTERFACE \${ONIKA_INCLUDE_DIRS} \${${PluginName}_INCLUDE_DIRS} ${XNB_ROOT_DIR}/include)\n")
      file(APPEND ${XNB_CMAKE_PACKAGE} "target_compile_definitions(${${PluginName}_SHARED_LIB} INTERFACE \${ONIKA_COMPILE_DEFINITIONS} \${${PluginName}_COMPILE_DEFINITIONS})\n")
      file(APPEND ${XNB_CMAKE_PACKAGE} "target_compile_options(${${PluginName}_SHARED_LIB} INTERFACE \${ONIKA_COMPILE_OPTIONS} \$<\$<COMPILE_LANGUAGE:CUDA>:\${ONIKA_CUDA_COMPILE_FLAGS}> \$<\$<COMPILE_LANGUAGE:HIP>:\${ONIKA_HIP_COMPILE_FLAGS}> \${${PluginName}_COMPILE_OPTIONS})\n")
      file(APPEND ${XNB_CMAKE_PACKAGE} "target_compile_features(${${PluginName}_SHARED_LIB} INTERFACE \${ONIKA_COMPILE_FEATURES} \${${PluginName}_COMPILE_FEATURES})\n")
      file(APPEND ${XNB_CMAKE_PACKAGE} "target_link_directories(${${PluginName}_SHARED_LIB} INTERFACE \${ONIKA_LIBRARY_DIRS} \${${PluginName}_LINK_DIRECTORIES})\n")
      file(APPEND ${XNB_CMAKE_PACKAGE} "target_link_libraries(${${PluginName}_SHARED_LIB} INTERFACE \${ONIKA_LIBRARIES} \${${PluginName}_LINK_LIBRARIES})\n")
    endif()
    file(APPEND ${XNB_CMAKE_PACKAGE} "set(${PluginName}_SHARED_LIB ${${PluginName}_SHARED_LIB})\n")
  endif()

  if(GVFILES)
    install(FILES ${GVFILES} DESTINATION ${XNB_ROOT_DIR}/redist/${PluginName})
    file(GLOB ${PluginName}_HDRS ${dirname}/*.h)
    if(${PluginName}_HDRS)
      install(FILES ${${PluginName}_HDRS} DESTINATION ${XNB_ROOT_DIR}/redist/${PluginName})
    endif()
    file(APPEND ${XNB_CMAKE_PACKAGE} "\n# Grid templated part of exaNBody's ${PluginName} plugin\n")    
    file(APPEND ${XNB_CMAKE_PACKAGE} "onika_add_plugin(${PluginName} ${XNB_ROOT_DIR}/redist/${PluginName} -\${XNB_APP_NAME}-plugin)\n")
  endif()

endfunction()

add_subdirectory(src)

file(APPEND ${XNB_CMAKE_PACKAGE} "\n# Optionall generate an executable wrapper named after end user's application name\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "macro(GenerateEndUserExecEnv)\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "  file(MAKE_DIRECTORY \${CMAKE_CURRENT_BINARY_DIR}/bin)\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "  file(WRITE  \${CMAKE_CURRENT_BINARY_DIR}/bin/setup-env.sh \"source \${exaNBody_DIR}/bin/setup-env.sh\\n\")\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "  file(APPEND \${CMAKE_CURRENT_BINARY_DIR}/bin/setup-env.sh \"[ \\\$LD_LIBRARY_PATH ] && LD_LIBRARY_PATH=\${CMAKE_INSTALL_PREFIX}/lib:\\\$LD_LIBRARY_PATH\\n\")\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "  file(APPEND \${CMAKE_CURRENT_BINARY_DIR}/bin/setup-env.sh \"[ \\\$LD_LIBRARY_PATH ] || LD_LIBRARY_PATH=\${CMAKE_INSTALL_PREFIX}/lib\\n\")\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "  file(APPEND \${CMAKE_CURRENT_BINARY_DIR}/bin/setup-env.sh \"export LD_LIBRARY_PATH\\n\")\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "  install(PROGRAMS \${CMAKE_CURRENT_BINARY_DIR}/bin/setup-env.sh DESTINATION bin)\n")

file(APPEND ${XNB_CMAKE_PACKAGE} "  file(WRITE  \${CMAKE_CURRENT_BINARY_DIR}/bin/\${XNB_APP_NAME} \"source \${CMAKE_INSTALL_PREFIX}/bin/setup-env.sh\\n\")\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "  file(APPEND \${CMAKE_CURRENT_BINARY_DIR}/bin/\${XNB_APP_NAME} \"[ \\\$ONIKA_CONFIG_PATH ] || ONIKA_CONFIG_PATH=\${CMAKE_INSTALL_PREFIX}/data/config\\n\")\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "  file(APPEND \${CMAKE_CURRENT_BINARY_DIR}/bin/\${XNB_APP_NAME} \"[ \\\$ONIKA_PLUGIN_PATH ] || ONIKA_PLUGIN_PATH=\${onika_DIR}/plugins:\${exaNBody_DIR}/plugins:\${CMAKE_INSTALL_PREFIX}/plugins\\n\")\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "  file(APPEND \${CMAKE_CURRENT_BINARY_DIR}/bin/\${XNB_APP_NAME} \"export ONIKA_CONFIG_PATH\\n\")\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "  file(APPEND \${CMAKE_CURRENT_BINARY_DIR}/bin/\${XNB_APP_NAME} \"export ONIKA_PLUGIN_PATH\\n\")\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "  file(APPEND \${CMAKE_CURRENT_BINARY_DIR}/bin/\${XNB_APP_NAME} \"\${ONIKA_RUN} \\\$*\\n\")\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "  install(PROGRAMS \${CMAKE_CURRENT_BINARY_DIR}/bin/\${XNB_APP_NAME} DESTINATION bin)\n")

file(APPEND ${XNB_CMAKE_PACKAGE} "  # local execution wrapper for execution from build directory without installation\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "  file(WRITE  \${CMAKE_CURRENT_BINARY_DIR}/\${XNB_APP_NAME} \"source \${exaNBody_DIR}/bin/setup-env.sh\\n\")\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "  file(APPEND \${CMAKE_CURRENT_BINARY_DIR}/\${XNB_APP_NAME} \"[ \\\$LD_LIBRARY_PATH ] && LD_LIBRARY_PATH=\${PROJECT_BINARY_DIR}/lib:\\\$LD_LIBRARY_PATH\\n\")\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "  file(APPEND \${CMAKE_CURRENT_BINARY_DIR}/\${XNB_APP_NAME} \"[ \\\$LD_LIBRARY_PATH ] || LD_LIBRARY_PATH=\${PROJECT_BINARY_DIR}/lib\\n\")\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "  file(APPEND \${CMAKE_CURRENT_BINARY_DIR}/\${XNB_APP_NAME} \"[ \\\$ONIKA_CONFIG_PATH ] || ONIKA_CONFIG_PATH=\${CMAKE_CURRENT_SOURCE_DIR}/data/config\\n\")\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "  file(APPEND \${CMAKE_CURRENT_BINARY_DIR}/\${XNB_APP_NAME} \"[ \\\$ONIKA_PLUGIN_PATH ] || ONIKA_PLUGIN_PATH=\${onika_DIR}/plugins:\${exaNBody_DIR}/plugins:\${PROJECT_BINARY_DIR}/plugins\\n\")\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "  file(APPEND \${CMAKE_CURRENT_BINARY_DIR}/\${XNB_APP_NAME} \"export LD_LIBRARY_PATH\\n\")\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "  file(APPEND \${CMAKE_CURRENT_BINARY_DIR}/\${XNB_APP_NAME} \"export ONIKA_CONFIG_PATH\\n\")\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "  file(APPEND \${CMAKE_CURRENT_BINARY_DIR}/\${XNB_APP_NAME} \"export ONIKA_PLUGIN_PATH\\n\")\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "  file(APPEND \${CMAKE_CURRENT_BINARY_DIR}/\${XNB_APP_NAME} \"\${ONIKA_RUN} \\\$*\\n\")\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "  file(CHMOD  \${CMAKE_CURRENT_BINARY_DIR}/\${XNB_APP_NAME} FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ)\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "endmacro()\n")

# install cmake package file to allow external project to find_package exaNBody
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/exanbody-config.cmake DESTINATION ${XNB_ROOT_DIR})

# generate exaNBody environment setup, based on onika onverionment setup script
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/bin/setup-env.sh
  "source ${onika_DIR}/bin/setup-env.sh\n"
  "[ \$LD_LIBRARY_PATH ] && LD_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:\$LD_LIBRARY_PATH\n"
  "[ \$LD_LIBRARY_PATH ] || LD_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib\n"
  "export LD_LIBRARY_PATH\n")
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/bin/setup-env.sh DESTINATION bin)

# also an executable wrapper for exaNBody itself
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/bin/exaNBody
  "source ${CMAKE_INSTALL_PREFIX}/bin/setup-env.sh\n"
  "[ \$ONIKA_CONFIG_PATH ] || ONIKA_CONFIG_PATH=${CMAKE_INSTALL_PREFIX}/data/config\n"
  "[ \$ONIKA_PLUGIN_PATH ] || ONIKA_PLUGIN_PATH=${onika_DIR}/plugins:${CMAKE_INSTALL_PREFIX}/plugins\n"
  "export ONIKA_CONFIG_PATH\n"
  "export ONIKA_PLUGIN_PATH\n"
  "${ONIKA_RUN} \$*\n")
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/bin/exaNBody DESTINATION bin)

# generate configuration files and scripts to run from build directory without install
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/exaNBody
  "source ${onika_DIR}/bin/setup-env.sh\n"
  "[ \$LD_LIBRARY_PATH ] && LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/lib:\$LD_LIBRARY_PATH\n"
  "[ \$LD_LIBRARY_PATH ] || LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/lib\n"
  "[ \$ONIKA_CONFIG_PATH ] || ONIKA_CONFIG_PATH=${CMAKE_CURRENT_SOURCE_DIR}/data/config\n"
  "[ \$ONIKA_PLUGIN_PATH ] || ONIKA_PLUGIN_PATH=${onika_DIR}/plugins:${PROJECT_BINARY_DIR}/plugins\n"
  "export ONIKA_CONFIG_PATH\n"
  "export ONIKA_PLUGIN_PATH\n"
  "export LD_LIBRARY_PATH\n"
  "${ONIKA_RUN} \$*\n")
file(CHMOD ${CMAKE_CURRENT_BINARY_DIR}/exaNBody FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ)


