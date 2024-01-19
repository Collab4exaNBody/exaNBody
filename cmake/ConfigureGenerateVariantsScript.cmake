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
#message(STATUS "GRID_VARIANT_FILE=${GRID_VARIANT_FILE}")
file(READ ${GRID_VARIANT_FILE} GRID_VARIANT_LIST)
#message(STATUS "GRID_VARIANT_LIST=${GRID_VARIANT_LIST}")

function(ParseSourceVariants src VariantDB)
  file(READ ${src} srcdata)
  string(REGEX REPLACE "#[ \t]*pragma[ \t]+xstamp_cuda_enable" "" srcdata "${srcdata}")
  string(REGEX REPLACE "#[ \t]*pragma[ \t]+xstamp_grid_variant" "\${GRID_VARIANT:}" srcdata "${srcdata}")
  string(REGEX MATCHALL "\\\${[^}]+}" varlist "${srcdata}")
  set(varcounter 0)
  foreach(var ${varlist})
    if(${var} MATCHES "\\\${GRID_VARIANT:")
      string(REPLACE "${var}" "\${${VariantDB}Item${varcounter}}\n" srcdata "${srcdata}")
      math(EXPR varcounter "${varcounter}+1")
      list(APPEND ${VariantDB}List "${GRID_VARIANT_LIST}")
    elseif(${var} MATCHES "\\\${VARIANT:")
      string(REPLACE "${var}" "\${${VariantDB}Item${varcounter}}\n" srcdata "${srcdata}")
      math(EXPR varcounter "${varcounter}+1")
      string(REGEX REPLACE "\\\${VARIANT:" "" var "${var}")
      string(REGEX REPLACE "}\\\$" "" var "${var}")
      string(REGEX REPLACE "^[ \t\n]+" "" var "${var}")
      #message("[SCR] found variant : ${var}")
      list(APPEND ${VariantDB}List "${var}")
    endif()
  endforeach()  
  set(${VariantDB}SourceData ${srcdata} PARENT_SCOPE)
  set(${VariantDB}List ${${VariantDB}List} PARENT_SCOPE)
endfunction()

function(WriteFileIfDifferent filename buffer)
  set(writefile ON)
  if(EXISTS ${filename})
    file(READ ${filename} data)
    if("${data}" STREQUAL "${buffer}")
      # message(STATUS "[SCR] content of file '${filename}' unchanged, write skipped")
      set(writefile OFF)
    endif()
  endif()
#  message(STATUS "WriteFileIfDifferent('${filename}','...') writefile=${writefile}")
  if(writefile)
    # message(STATUS "Generating file '${filename}' ...")
    file(WRITE ${filename} "${buffer}")
  endif()
endfunction()

function(GenVariantSourceFile VariantIndex VariantDB)
  list(LENGTH ${VariantDB}List NVariants)
  if(${VariantIndex} EQUAL NVariants)
    list(LENGTH ${VariantDB}SourceFiles FileCounter)
    string(CONFIGURE "${${VariantDB}SourceData}" GenSourceContent)
    set(srcfilename ${${VariantDB}SourceFilePrefix}${FileCounter}${${VariantDB}SourceFileSuffix})
    #message(STATUS "[SCR] configure ${srcfilename}")
    WriteFileIfDifferent(${srcfilename} "${GenSourceContent}")
    list(APPEND ${VariantDB}SourceFiles ${srcfilename})
  else()
    math(EXPR NextVariantIndex "${VariantIndex}+1")
    list(GET ${VariantDB}List ${VariantIndex} variant)
    string(REGEX MATCHALL "[^\n]*\n" values "${variant}")
    list(LENGTH values nvalues)
    foreach(val ${values})
      set(${VariantDB}Item${VariantIndex} ${val})
      GenVariantSourceFile(${NextVariantIndex} ${VariantDB})
      unset(${VariantDB}Item${VariantIndex})
    endforeach()
  endif()
  set(${VariantDB}SourceFiles ${${VariantDB}SourceFiles} PARENT_SCOPE)
endfunction()

function(GenerateSourcesFromInput src srcdirlistdd dstdir VariantDB outsrclist)
    ParseSourceVariants(${src} ${VariantDB})
    string(REGEX REPLACE ".in\$" "" ${VariantDB}SourceFilePrefix "${src}")
    string(REGEX MATCH ".[^.]+\$" ${VariantDB}SourceFileSuffix "${${VariantDB}SourceFilePrefix}")
    string(REGEX REPLACE "${${VariantDB}SourceFileSuffix}\$" "" ${VariantDB}SourceFilePrefix "${${VariantDB}SourceFilePrefix}")
    
    # Avoid generating files in source tree in case of script error
    string(REPLACE ":" ";" srcdirlist "${srcdirlistdd}")
#    message(STATUS "srcdirlist = ${srcdirlist}")
    foreach(srcdir ${srcdirlist})
      string(REPLACE "${srcdir}" "${dstdir}" ${VariantDB}SourceFilePrefix "${${VariantDB}SourceFilePrefix}")
#      message(STATUS "replace ${srcdir} -> ${dstdir} : ${${VariantDB}SourceFilePrefix}")
    endforeach()
    
    if(NOT "${${VariantDB}SourceFilePrefix}" MATCHES "${dstdir}")
      message(FATAL_ERROR "substitution of ${srcdirlistdd} by ${dstdir} failed for ${${VariantDB}SourceFilePrefix}")
    endif()
#    message(STATUS "prefix=${${VariantDB}SourceFilePrefix}, suffix=${${VariantDB}SourceFileSuffix}")
    list(LENGTH ${VariantDB}List nvariants)
#    message(STATUS "${src} has ${nvariants} variations")
    if(${nvariants} EQUAL 0)
      string(CONFIGURE "${${VariantDB}SourceData}" GenSourceContent)
      set(srcfilename ${${VariantDB}SourceFilePrefix}${${VariantDB}SourceFileSuffix})
      WriteFileIfDifferent(${srcfilename} "${GenSourceContent}")
      set(${VariantDB}SourceFiles ${srcfilename})
    else()
      GenVariantSourceFile(0 ${VariantDB})
    endif()
    list(APPEND ${outsrclist} ${${VariantDB}SourceFiles})
        
    unset(${VariantDB}List)
    unset(${VariantDB}SourceFilePrefix)
    unset(${VariantDB}SourceFileSuffix)
    unset(${VariantDB}SourceFiles)
    unset(${VariantDB}SourceData)
    unset(VariantDB)
    set(${outsrclist} ${${outsrclist}} PARENT_SCOPE)
endfunction()


set(VariantDB SCRDB)
#message(STATUS "[SCR] process input file ${SourceFile}")
GenerateSourcesFromInput(${SourceFile} ${InputBaseDir} ${OutputBaseDir} ${VariantDB} GeneratedSources)
#message(STATUS "[SCR] generated files ${GeneratedSources}")


