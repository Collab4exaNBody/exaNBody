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

