# ---------------------------------------------------------------------------
# C++ extension: _exanb_data
# Registers exaNBody-specific slot extractors into pyonika's slot_as_array
# registry (SimulationStatistics → dict, and future Tier 2 types).
# ---------------------------------------------------------------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.13.6)
  FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(_exanb_data ${CMAKE_CURRENT_SOURCE_DIR}/bind_exanb_data.cpp)
target_include_directories(_exanb_data PRIVATE
  ${ONIKA_INCLUDE_DIRS}
  ${CMAKE_INSTALL_PREFIX}/include
)
# These three definitions are what exanbody-config.cmake propagates via
# exanbCore's INTERFACE properties.  They make onika::math types (Vec3d, IJK,
# AABB, …) available unqualified inside namespace exanb.
target_compile_definitions(_exanb_data PRIVATE
  ONIKA_MATH_EXPORT_NAMESPACE=exanb
  ONIKA_SCG_EXPORT_NAMESPACE=exanb
  ONIKA_LOG_EXPORT_NAMESPACE=exanb
)
target_link_libraries(_exanb_data PRIVATE exanbCore exanbGridCellParticles exanbIO)
#   ${CMAKE_INSTALL_PREFIX}/lib/libexanbIO.so
#   ${CMAKE_INSTALL_PREFIX}/lib/libexanbCore.so
#   ${CMAKE_INSTALL_PREFIX}/lib/libexanbGridCellParticles.so
# )
# register_slot_extractor lives in pyonika.so which Python loads with
# RTLD_LOCAL by default.  pyexanbody/__init__.py re-opens pyonika with
# RTLD_GLOBAL before importing _exanb_data, making the symbol visible at
# runtime without a link-time dependency on pyonika.
target_link_options(_exanb_data PRIVATE
  $<$<PLATFORM_ID:Linux>:-Wl,--allow-shlib-undefined>
  $<$<PLATFORM_ID:Darwin>:-Wl,-undefined,dynamic_lookup>
)
set_target_properties(_exanb_data PROPERTIES
  CXX_STANDARD 20
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/pyexanbody
)
install(TARGETS _exanb_data DESTINATION lib/pyexanbody)

# ---------------------------------------------------------------------------
# Generate _paths.py from template, substituting onika_DIR and CMAKE_INSTALL_PREFIX.
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/pyexanbody/_paths.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/pyexanbody/_paths.py
    @ONLY
)

# Install the package source files (everything under pyexanbody/ except the .in template).
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/pyexanbody
    DESTINATION lib
    FILES_MATCHING PATTERN "*.py"
    PATTERN "_paths.py.in" EXCLUDE
)

# Install the CMake-generated _paths.py (contains baked-in install paths).
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/pyexanbody/_paths.py
    DESTINATION lib/pyexanbody
)

# Install README and example scripts with the solar-system sample .msp.
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README.md DESTINATION python)
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/exemples/pyexanbody_dryrun_test_import_pyexanbody.py
    ${CMAKE_CURRENT_SOURCE_DIR}/exemples/pyexanbody_reproduce_solar_system_case.py
    ${CMAKE_CURRENT_SOURCE_DIR}/exemples/pyexanbody_run_example.py
    ${CMAKE_CURRENT_SOURCE_DIR}/exemples/pyexanbody_run_main_config.py
    ${CMAKE_CURRENT_SOURCE_DIR}/exemples/pyexanbody_run_solar_system.py
    ${CMAKE_CURRENT_SOURCE_DIR}/exemples/pyexanbody_read_sim_stats.py
    ${CMAKE_CURRENT_SOURCE_DIR}/exemples/pyexanbody_read_cell_values.py
    DESTINATION python/exemples
)
