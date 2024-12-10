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

set(XNB_CHUNK_NEIGHBORS_CS_LIST "4;1" CACHE STRING "Chunk neighbors specialization list. Use VARIMPL insead of a number for runtime variable chunk_size.")

unset(XNB_CHUNK_NEIGHBORS_CS_LIST_DEF)
foreach(SPE ${XNB_CHUNK_NEIGHBORS_CS_LIST})
  if(XNB_CHUNK_NEIGHBORS_CS_LIST_DEF)
    set(XNB_CHUNK_NEIGHBORS_CS_LIST_DEF "${XNB_CHUNK_NEIGHBORS_CS_LIST_DEF} XNB_COMMA ${SPE}")
  else()
    set(XNB_CHUNK_NEIGHBORS_CS_LIST_DEF "${SPE}")
  endif()
endforeach()

if(XNB_CHUNK_NEIGHBORS_CS_LIST_DEF)
  set(exanbParticleNeighbors_COMPILE_DEFINITIONS XNB_CHUNK_NEIGHBORS_CS_LIST=${XNB_CHUNK_NEIGHBORS_CS_LIST_DEF})
endif()
set(exanbParticleNeighbors_LINK_LIBRARIES exanbAMR)
xnb_add_plugin(exanbParticleNeighbors ${CMAKE_CURRENT_SOURCE_DIR})

