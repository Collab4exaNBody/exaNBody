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

# =============================
# === micro-stamp mini-app  ===
# =============================
# does not specialize anything, simmply adds a plugin with MD intramolecular potential

# ==========================================
# === build application plugins and libs ===
# ==========================================
set(ONIKA_BUILD_microStamp_FORCED ON)
set(microStamp_LINK_LIBRARIES exanbGridCellParticles exanbCompute exanbParticleNeighbors exanbIO exanbMPI exanbCore)
onika_add_plugin(microStamp ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ============================
# === Add some basic tests ===
# ============================
AddRegressionTestDir(${CMAKE_CURRENT_SOURCE_DIR}/regression)


