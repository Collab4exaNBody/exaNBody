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
# =======================================
# === Outils et biblioth�ques tierces ===
# =======================================
# MPI est necessaire pour ce projet
if(NOT MPI_CXX_LIBRARIES)
  find_package(MPI REQUIRED COMPONENTS C)
  set(MPI_CXX_LIBRARIES ${MPI_CXX_LIBRARIES} ${MPI_C_EXTRA_LIBRARIES})
endif()

# Test avec des intervalles stables � +/-1 pret, avec des identifiants tournant
add_executable(XsDataMoveSimpleTest SimpleTest.cc)
target_include_directories(XsDataMoveSimpleTest PUBLIC ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(XsDataMoveSimpleTest exanbMPI ${MPI_CXX_LIBRARIES})

set(NUMPROCS_LIST 1 7 8)
set(IDROTATION_LIST 0 1 77)
set(IDSTART_LIST 0 1 1023)
set(IDCOUNT_LIST 10 110 100000)
set(NPERMS_LIST 10 1000)
set(SEED_LIST 0 1976)

foreach(NUMPROCS ${NUMPROCS_LIST})
    foreach(IDROTATION ${IDROTATION_LIST})
      foreach(IDSTART ${IDSTART_LIST})
        foreach(IDCOUNT ${IDCOUNT_LIST})
          math(EXPR IDEND ${IDSTART}+${IDCOUNT})
          xstamp_add_test_par(xsdatamove_simple_np${NUMPROCS}_rot${IDROTATION}_st${IDSTART}_ic${IDCOUNT} ${NUMPROCS} 1 ${CMAKE_CURRENT_BINARY_DIR}/XsDataMoveSimpleTest ${MPIEXEC_POSTFLAGS} ${IDSTART} ${IDEND} ${IDROTATION})
        endforeach()
      endforeach()
    endforeach()
endforeach()

xstamp_add_test_par(xsdatamove_simple_adhoc1 3 1 ${CMAKE_CURRENT_BINARY_DIR}/XsDataMoveSimpleTest ${MPIEXEC_POSTFLAGS} 0 20 3)
xstamp_add_test_par(xsdatamove_simple_adhoc2 3 1 ${CMAKE_CURRENT_BINARY_DIR}/XsDataMoveSimpleTest ${MPIEXEC_POSTFLAGS} 0 107 3)

# test avec des intervalles et listes d'identifiants al�atoires
add_executable(XsDataMoveRandomTest RandomTest.cc)
target_include_directories(XsDataMoveRandomTest PUBLIC ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(XsDataMoveRandomTest exanbMPI ${MPI_CXX_LIBRARIES})

foreach(NUMPROCS ${NUMPROCS_LIST})
  foreach(SEED ${SEED_LIST})
    foreach(IDCOUNT ${IDCOUNT_LIST})
      foreach(NPERMS ${NPERMS_LIST})
        xstamp_add_test_par(xsdatamove_random_s${SEED}_ic${IDCOUNT}_permm${NPERMS}_np${NUMPROCS} ${NUMPROCS} 1 ${CMAKE_CURRENT_BINARY_DIR}/XsDataMoveRandomTest ${MPIEXEC_POSTFLAGS} ${SEED} ${IDCOUNT} ${NPERMS})
      endforeach()
    endforeach()
  endforeach()
endforeach()

