# =================================
# === base IO library  ===
# =================================
find_package(ZLIB REQUIRED)

set(exanbIO_LINK_LIBRARIES exanbGridCellParticles exanbMPI ${ZLIB_LIBRARIES} stdc++fs)
set(exanbIO_INCLUDE_DIRS ${BASEN_INCLUDE_DIRS}) 
xstamp_add_plugin(exanbIO ${CMAKE_CURRENT_SOURCE_DIR})

