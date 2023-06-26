# =================================
# === micro-stamp mpi library  ===
# =================================

set(exanbMPI_COMPILE_DEFINITIONS ${EXASTAMP_ZOLTAN_DEFINITIONS})
set(exanbMPI_INCLUDE_DIRS ${EXASTAMP_ZOLTAN_INCLUDE_DIRS})
set(exanbMPI_LINK_LIBRARIES exanbCompute exanbGridCellParticles ${EXASTAMP_ZOLTAN_LIBRARIES})
xstamp_add_plugin(exanbMPI ${CMAKE_CURRENT_SOURCE_DIR})

