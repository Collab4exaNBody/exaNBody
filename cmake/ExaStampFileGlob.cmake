# ======================================
# === exaStamp extension sub-modules ===
# ======================================

file(GLOB EXASTAMP_SUBMODULES RELATIVE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/exaStamp*)
message(STATUS "EXASTAMP_SUBMODULES = ${EXASTAMP_SUBMODULES}")

function(ExaStampFileGlob var pattern)
  file(GLOB filelist ${pattern})
  if(pattern MATCHES ${CMAKE_SOURCE_DIR})
    foreach(esm ${EXASTAMP_SUBMODULES})
      string(REPLACE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/${esm} addon_pattern ${pattern})
      file(GLOB addon_pattern ${pattern})
      list(APPEND filelist ${addon_pattern})
    endforeach()
  endif()
  set(${var} ${filelist} PARENT_SCOPE)
endfunction()


