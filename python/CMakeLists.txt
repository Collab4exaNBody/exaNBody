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
    DESTINATION python/exemples
)
