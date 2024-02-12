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
cmake_minimum_required(VERSION 3.18)
project(exaNBody LANGUAGES C CXX)

get_filename_component(XNB_ROOT_DIR ${CMAKE_INSTALL_PREFIX} ABSOLUTE)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/exanbody-config.cmake "set(exaNBody_DIR ${XNB_ROOT_DIR})\nlist(APPEND CMAKE_MODULE_PATH \"${XNB_ROOT_DIR}/cmake\")\ninclude(exaNBody)\n")

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/exanbody-config.cmake DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY src cmake thirdparty scripts DESTINATION ${CMAKE_INSTALL_PREFIX})

