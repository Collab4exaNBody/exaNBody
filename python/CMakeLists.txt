# pyexanbody: Python package for exaNBody, on top of pyonika.
# _exanb_data registers extractors for exaNBody types (SimulationStatistics,
# GridCellValues, ParticleFieldBundle) into pyonika's slot_as_array registry.
xnb_add_python_package(
  NAME pyexanbody
  PACKAGE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pyexanbody
  EXTENSION _exanb_data
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/bind_exanb_data.cpp
  LINK_LIBRARIES exanbCore exanbGridCellParticles exanbIO
  DEPENDENCIES pyonika
  DESCRIPTION "Python interface to the exaNBody particle simulation framework")

# README and example scripts
file(GLOB PYEXANBODY_EXAMPLES ${CMAKE_CURRENT_SOURCE_DIR}/examples/*.py)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README.md ${CMAKE_CURRENT_SOURCE_DIR}/PYTHON_DATA_ACCESS.md DESTINATION python)
install(FILES ${PYEXANBODY_EXAMPLES} DESTINATION python/examples)

if(ONIKA_REGRESSION_TEST_ENABLE)
  add_test(NAME pyexanbody_tests
           COMMAND ${Python3_EXECUTABLE} -m pytest -q -p no:cacheprovider ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  set_tests_properties(pyexanbody_tests PROPERTIES ENVIRONMENT
    "PYTHONPATH=${PROJECT_BINARY_DIR}/python:${onika_DIR}/python:$ENV{PYTHONPATH};ONIKA_PLUGIN_PATH=${onika_DIR}/plugins:${PROJECT_BINARY_DIR}/plugins;ONIKA_CONFIG_PATH=${PROJECT_SOURCE_DIR}/data/config")
endif()
