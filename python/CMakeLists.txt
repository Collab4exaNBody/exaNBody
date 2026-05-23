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
    ${CMAKE_CURRENT_SOURCE_DIR}/exemples/test.py
    ${CMAKE_CURRENT_SOURCE_DIR}/exemples/run_solar_system.py
    ${CMAKE_CURRENT_SOURCE_DIR}/exemples/solar_system_graph.py
    ${CMAKE_CURRENT_SOURCE_DIR}/../contribs/microCosmos/samples/simple_example/solar_system.msp
    DESTINATION python/exemples
)
