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
message("Analysing ${BINARY_FILE} ...")

execute_process(COMMAND ${SOATL_OBJDUMP} -D ${BINARY_FILE} OUTPUT_FILE ${BINARY_FILE}.asm OUTPUT_QUIET ERROR_QUIET)

file(STRINGS ${BINARY_FILE}.asm BINARRY_ASSEMBLY)

set(INSLIST vmov vmovapd vmovaps vmovupd vmovups sqrt vsqrtpd vrsqrtpd vsqrtps vrsqrtps vsqrtsd vrsqrtsd vsqrtss vrsqrtss vfmadd132sd vfmadd132pd vfmadd132ss vfmadd132ps)

set(mova vmovapd vmovaps)
set(movu vmovupd vmovups)
set(sqrtp vsqrtpd vrsqrtpd vsqrtps vrsqrtps)
set(sqrts vsqrtsd vrsqrtsd vsqrtss vrsqrtss)
set(fmas vfmadd132sd vfmadd132ss)
set(fmap vfmadd132pd vfmadd132ps)
set(fma vfmadd132sd vfmadd132ss vfmadd132pd vfmadd132ps)
set(INSSUM mova movu sqrtp sqrts fmas fmap)

set(INSREPORT mova movu sqrtp sqrts fmap fmas)

foreach(ins ${INSLIST})
  set(${ins} 0)
endforeach()

foreach(line ${BINARRY_ASSEMBLY})
  foreach(ins ${INSLIST})
    if(${line} MATCHES ".*${ins}.*")
      math(EXPR ${ins} ${${ins}}+1)
    endif()
  endforeach()
endforeach()

foreach(tosum ${INSSUM})
  set(tmplist ${${tosum}})
  set(${tosum} 0)
  foreach(ins ${tmplist})
    math(EXPR ${tosum} ${${tosum}}+${${ins}})
  endforeach()
endforeach()

foreach(ins ${INSREPORT})
  message("${ins} ${${ins}}")
endforeach()


