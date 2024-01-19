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
# ==========================================
# === exaStamp/exaDEM/exaSPH application ===
# ==========================================

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${USTAMP_APPS_DIR})

file(GLOB USTAMP_MAIN_SRCS *.cpp)

#message(STATUS "USTAMP_CORE_LIBRARIES=${USTAMP_CORE_LIBRARIES}")

add_executable(${XNB_APP_NAME} ${USTAMP_MAIN_SRCS})
target_compile_options(${XNB_APP_NAME} PUBLIC ${USTAMP_CXX_FLAGS})
target_compile_definitions(${XNB_APP_NAME} PUBLIC ${USTAMP_COMPILE_DEFINITIONS} ${XSTAMP_CUDA_COMPILE_DEFINITIONS})
target_include_directories(${XNB_APP_NAME} PUBLIC ${USTAMP_INCLUDE_DIRS})
target_link_libraries(${XNB_APP_NAME} exanbMPI ${USTAMP_CORE_LIBRARIES})

install(TARGETS ${XNB_APP_NAME} DESTINATION bin)

