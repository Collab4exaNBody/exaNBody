
# Copy example folder for microStamp
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/samples DESTINATION share/microStamp_examples)

# Add regression tests for microStamp
if(ONIKA_REGRESSION_TEST_ENABLE)
  file(GLOB_RECURSE POTENTIAL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/potentials/*")
  file(COPY ${POTENTIAL_FILES} DESTINATION "${CMAKE_BINARY_DIR}/contribs/microStamp")
  AddRegressionTestDir(${CMAKE_CURRENT_SOURCE_DIR}/samples)
endif()

