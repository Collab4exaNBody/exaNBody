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


option(XNB_ENABLE_CCC_USER "Use  CCC supercomputing facility's user library" OFF)
if(XNB_ENABLE_CCC_USER)
  add_library(exanb_ccc_user INTERFACE)
  target_compile_definitions(exanb_ccc_user INTERFACE __use_lib_ccc_user=1)
  target_link_libraries(exanb_ccc_user INTERFACE ccc_user)
  set(exanbLogic_LINK_LIBRARIES exanb_ccc_user)
  file(APPEND ${XNB_CMAKE_PACKAGE} "\n# CCC supercomputing facility's user library\n")
  file(APPEND ${XNB_CMAKE_PACKAGE} "add_library(exanb_ccc_user INTERFACE)\n")
  file(APPEND ${XNB_CMAKE_PACKAGE} "target_compile_definitions(exanb_ccc_user INTERFACE __use_lib_ccc_user=1)\n")
  file(APPEND ${XNB_CMAKE_PACKAGE} "target_link_libraries(exanb_ccc_user INTERFACE ccc_user)\n")
endif()

xnb_add_plugin(exanbLogic ${CMAKE_CURRENT_SOURCE_DIR})

