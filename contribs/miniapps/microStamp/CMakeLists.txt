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


cmake_minimum_required(VERSION 3.26)

# ========================================
# === Compiler toolchain configuration ===
# ========================================
# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS NO)

# ===============
# === Project ===
# ===============
project(microStamp VERSION 1.0 LANGUAGES C CXX)

# ==============================
# === Application version    ===
# ==============================
set(MICROSTAMP_VERSION "microStamp-${CMAKE_PROJECT_VERSION}") 
message(STATUS "Project version : ${MICROSTAMP_VERSION}")

# ===========================
# === CMake customization ===
# ===========================
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Release RelWithDebInfo Debug)

# ======================================
# === exaNBody application framework ===
# ======================================
find_package(exaNBody)

# ==========================================
# === build application plugins and libs ===
# ==========================================
set(microStamp_LINK_LIBRARIES exanbGridCellParticles exanbCompute exanbMPI exanbCore)
onika_add_plugin(microStamp ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ============================
# === Add some basic tests ===
# ============================
AddRegressionTestDir(${CMAKE_CURRENT_SOURCE_DIR}/regression)

# ===========================================================
# === generate execution wrappers and environment scripts ===
# ===========================================================
GenerateEndUserExecEnv()

