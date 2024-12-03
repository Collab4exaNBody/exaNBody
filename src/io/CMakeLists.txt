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
# === base IO library  ===
# =================================
find_package(ZLIB REQUIRED)

set(XNB_MPIIO_PREOPEN_FILE_PARTS "4" CACHE STRING "Number of pre-opened MPI-IO file parts. Increase if crashes happen when writing very large files during parallel runs")
set(exanbIO_COMPILE_DEFINITIONS "XNB_MPIIO_PREOPEN_FILE_PARTS=${XNB_MPIIO_PREOPEN_FILE_PARTS}")
set(exanbIO_LINK_LIBRARIES exanbGridCellParticles exanbCompute exanbMPI ${ZLIB_LIBRARIES} basen stdc++fs)
#set(exanbIO_INCLUDE_DIRS ${BASEN_INCLUDE_DIRS}) 
xnb_add_plugin(exanbIO ${CMAKE_CURRENT_SOURCE_DIR})

