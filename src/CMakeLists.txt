# just forward to subdirectories, in the right order

# Simulation code execution support library
option(SOATL_SIZE_32BITS_FORCED "ExaStamp Uses 32 bits size type for SOATL arrays" ON)
add_subdirectory(onika)

# N-Body common library
add_subdirectory(exanb)

# shared main
add_subdirectory(appmain)

