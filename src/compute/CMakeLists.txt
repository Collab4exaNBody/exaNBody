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
# ====================================================================
# === micro-stamp pair potential compute operator library library  ===
# ====================================================================

set(XNB_CHUNK_NBH_DELAYED_COMPUTE_BUFER_SIZE "4" CACHE STRING "delay computations in neighbor traversal loop to enforce synchronized calls to user functor. size of delay buffer. set to 0 to disable")
if(${XNB_CHUNK_NBH_DELAYED_COMPUTE_BUFER_SIZE} LESS_EQUAL 1)
  set(XNB_CHUNK_NBH_DELAYED_COMPUTE_BUFER_SIZE 1)
endif()
set(XNB_CHUNK_NBH_DELAYED_COMPUTE_MAX_BLOCK_SIZE "64" CACHE STRING "Delay buffer uses shared memory. when enabled, gpu block size must be limited to this size.")
option(XNB_USE_CS1_SPECIALIZATION "Use specialized implementation of chunk_neighbors based computations for chunk_size=1" ON)
if(${XNB_USE_CS1_SPECIALIZATION})
  set(XNB_USE_CS1_SPECIALIZATION_CPP XNB_USE_CS1_SPECIALIZATION=1)
endif()
set(exanbCompute_COMPILE_DEFINITIONS
	"XNB_CHUNK_NBH_DELAYED_COMPUTE_BUFER_SIZE=${XNB_CHUNK_NBH_DELAYED_COMPUTE_BUFER_SIZE}"
	"XNB_CHUNK_NBH_DELAYED_COMPUTE_MAX_BLOCK_SIZE=${XNB_CHUNK_NBH_DELAYED_COMPUTE_MAX_BLOCK_SIZE}"
	"${XNB_USE_CS1_SPECIALIZATION_CPP}")

set(exanbCompute_LINK_LIBRARIES exanbGridCellParticles exanbParticleNeighbors)
onika_add_plugin(exanbCompute ${CMAKE_CURRENT_SOURCE_DIR})

