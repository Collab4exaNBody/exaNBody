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
# ======================================
# === exaStamp extension sub-modules ===
# ======================================

file(GLOB EXASTAMP_SUBMODULES RELATIVE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/exaStamp*)
message(STATUS "EXASTAMP_SUBMODULES = ${EXASTAMP_SUBMODULES}")

function(ExaStampFileGlob var pattern)
  file(GLOB filelist ${pattern})
  if(pattern MATCHES ${CMAKE_SOURCE_DIR})
    foreach(esm ${EXASTAMP_SUBMODULES})
      string(REPLACE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/${esm} addon_pattern ${pattern})
      file(GLOB addon_pattern ${pattern})
      list(APPEND filelist ${addon_pattern})
    endforeach()
  endif()
  set(${var} ${filelist} PARENT_SCOPE)
endfunction()


