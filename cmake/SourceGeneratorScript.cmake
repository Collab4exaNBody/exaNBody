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
cmake_minimum_required(VERSION 3.10)

# populate CMake variables from input config description file
include(InputConfigFileReadFunctions)
include(InputConfigGeneratorFunctions)
include(PotentialGeneratorFunctions)

ReadInputConfigDB(${EXASTAMP_INPUT_CONFIG_FILE})

GenInputConfigStringParserCppDecl(Input)
GenInputConfigStringParserCppBody(Input)
GenInputConfigFieldCppDecl()
GenInputConfigTypeDecl()
set(InputConfigCppReadFieldStatusVar autogen_config_read_ok)
set(InputConfigCppReadFieldVar autogen_field_var)
set(InputConfigCppReadValueVar autogen_val_var)
set(InputConfigCppDataBaseVar m_configKeywords)
GenInputConfigDataBaseCppInitDecl(${InputConfigCppDataBaseVar})
GenInputConfigDataBaseCppInitBody(${InputConfigCppDataBaseVar})
GenInputConfigCppReadField(${InputConfigCppDataBaseVar} ${InputConfigCppReadFieldVar} ${InputConfigCppReadValueVar} ${InputConfigCppReadFieldStatusVar})
GenInputConfigFieldCppAccessorInlineBody()
GenInputConfigGroupCppDecl()
GenInputConfigFieldConvertCpp()

GenIPotentialConfigurationMemberDecl()
GenIPotentialConfigurationConstructorBody()
GenPotentialParametersStructsDecl()
GenPotentialReferenceMapConfigureBody(__potentialConfiguration)
GenIPotentialEnumSymbols()
GenEamPotentialImplementations()
GenPairPotentialImplementations()
GenEamPotentialSingleSpecGridComputeForceBody()
GenEamPotentialSingleSpecGridVerletComputeForceBody()
GenPairPotentialSingleSpecGridComputeForceBody()
GenPairPotentialSingleSpecGridVerletComputeForceBody()

set(InputConfigPrintHelpString __helpString)
GenPrintHelpBody(${InputConfigPrintHelpString})

# generate fconfigured sources
string(REPLACE " " ";" EXASTAMP_CONFIGURED_HDRS ${EXASTAMP_CONFIGURED_HDRS})
string(REPLACE " " ";" EXASTAMP_CONFIGURED_SRCS ${EXASTAMP_CONFIGURED_SRCS})
list(APPEND FilesToConfigure ${EXASTAMP_CONFIGURED_HDRS})
list(APPEND FilesToConfigure ${EXASTAMP_CONFIGURED_SRCS})

foreach(ifile ${FilesToConfigure})
  string(REGEX REPLACE ".in$" "" ofile ${ifile})
  configure_file(${SOURCE_DIR}/${ifile} ${OUTPUT_DIR}/${ofile})
endforeach()

