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

# Copy example folder for microStamp
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/samples DESTINATION share/microStamp_examples)

# Add regression tests for microStamp
if(ONIKA_REGRESSION_TEST_ENABLE)
  file(GLOB_RECURSE POTENTIAL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/potentials/*")
  file(COPY ${POTENTIAL_FILES} DESTINATION "${CMAKE_BINARY_DIR}/contribs/microStamp")
  AddRegressionTestDir(${CMAKE_CURRENT_SOURCE_DIR}/samples)
endif()

