# just forward to subdirectories, in the right order

# Simulation code execution support library
option(SOATL_SIZE_32BITS_FORCED "ExaStamp Uses 32 bits size type for SOATL arrays" ON)
add_subdirectory(onika)

# N-Body common library
add_subdirectory(exanb)

# exaStamp (DM)
xstamp_optional_package(exaStamp exaStamp)

# DEM app
xstamp_optional_package(exaDEM exaDEM)

# SPH plugins
xstamp_optional_package(exaSPH exaSPH)

# shared main
add_subdirectory(appmain)

