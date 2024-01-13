
add_subdirectory(core)
add_subdirectory(compute)
add_subdirectory(mpi)
add_subdirectory(defbox)
add_subdirectory(logic)
add_subdirectory(amr)
add_subdirectory(particle_neighbors)
add_subdirectory(grid_cell_particles)
add_subdirectory(analytics)
add_subdirectory(io)
add_subdirectory(debug)

AddRegressionTestDir(${CMAKE_CURRENT_SOURCE_DIR}/regression)

