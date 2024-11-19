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

# Test avec des intervalles stables à +/-1 pret, avec des identifiants tournant
add_executable(XsDataMoveSimpleTest SimpleTest.cc)
target_include_directories(XsDataMoveSimpleTest PUBLIC ${XNB_INCLUDE_DIRS})
target_link_libraries(XsDataMoveSimpleTest exanbMPI ${XNB_CORE_LIBS})

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

# test avec des intervalles et listes d'identifiants aléatoires
add_executable(XsDataMoveRandomTest RandomTest.cc)
target_include_directories(XsDataMoveRandomTest PUBLIC ${XNB_INCLUDE_DIRS})
target_link_libraries(XsDataMoveRandomTest exanbMPI ${XNB_CORE_LIBS})

foreach(NUMPROCS ${NUMPROCS_LIST})
  foreach(SEED ${SEED_LIST})
    foreach(IDCOUNT ${IDCOUNT_LIST})
      foreach(NPERMS ${NPERMS_LIST})
        xstamp_add_test_par(xsdatamove_random_s${SEED}_ic${IDCOUNT}_permm${NPERMS}_np${NUMPROCS} ${NUMPROCS} 1 ${CMAKE_CURRENT_BINARY_DIR}/XsDataMoveRandomTest ${MPIEXEC_POSTFLAGS} ${SEED} ${IDCOUNT} ${NPERMS})
      endforeach()
    endforeach()
  endforeach()
endforeach()

