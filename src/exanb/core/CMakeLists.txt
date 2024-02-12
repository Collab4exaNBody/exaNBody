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
# =================================
# === micro-stamp core library  ===
# =================================

# file(GLOB USTAMP_CORE_SRCS *.cpp)
set(exanbCore_NO_CORE_LIBS ON)
set(exanbCore_LINK_LIBRARIES ${ONIKA_LIBRARIES} ${YAML_CPP_LIBRARIES} ${OpenMP_CXX_LIBRARIES} ${MPI_CXX_LIBRARIES} ${EXASTAMP_HOST_HW_LIBRARIES} tinyexpr dl)
set(exanbCore_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} ${ONIKA_INCLUDE_DIRS})
set(exanbCore_COMPILE_DEFINITIONS ${XNB_CUDA_COMPILE_DEFINITIONS} ${ONIKA_DEFINITIONS})

xstamp_add_plugin(exanbCore ${CMAKE_CURRENT_SOURCE_DIR})

#GenerateCurrentDirectorySourceFiles(USTAMP_CORE_SRCS USTAMP_CORE_DEPS)
#add_library(exanbCore SHARED ${USTAMP_CORE_SRCS})
#target_compile_options(exanbCore PRIVATE ${USTAMP_CXX_FLAGS})
#target_compile_definitions(exanbCore PRIVATE ${USTAMP_COMPILE_DEFINITIONS} ${XNB_CUDA_COMPILE_DEFINITIONS} PUBLIC ${ONIKA_DEFINITIONS})
#target_include_directories(exanbCore PRIVATE ${USTAMP_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} PUBLIC ${ONIKA_INCLUDE_DIRS})
#target_link_directories(exanbCore PRIVATE ${CUDA_LIBRARY_DIR})
#target_link_libraries(exanbCore PUBLIC ${YAML_CPP_LIBRARIES} PRIVATE ${CUDA_LIBRARIES} PUBLIC ${ONIKA_LIBRARIES})
#install(TARGETS exanbCore DESTINATION lib)

