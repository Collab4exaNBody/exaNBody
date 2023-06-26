# Important variables :
# EXASTAMP_TEST_DATA_DIR : directory to check
# EXASTAMP_TEST_DATA_MD5_FILE : md5 signature file
# EXASTAMP_CHECK_DATA_DIR_MD5 : enable verification

function(CheckDirectoryMD5 dirpath md5file resultvar)
  set(${resultvar} ON PARENT_SCOPE)
  file(READ "${md5file}" md5db)
  string(REGEX MATCHALL "[^\n]+\n" md5db "${md5db}")
  foreach(md5line ${md5db})
    string(REGEX REPLACE "[ ]+" ";" md5entry ${md5line})
    string(REGEX REPLACE "\n" "" md5entry "${md5entry}")
    list(GET md5entry 0 filename)
    list(GET md5entry 1 refdigest)
    message(STATUS "Check integrity of ${dirpath}/${filename}")
    if(NOT EXISTS ${dirpath}/${filename})
      message(WARNING "Failed to read ${dirpath}/${filename}")
      set(${resultvar} OFF PARENT_SCOPE)
    endif()
    file(MD5 ${dirpath}/${filename} digest)
    if(NOT "${digest}" STREQUAL "${refdigest}")
      message(WARNING "${filename}: MD5 digest ${digest} differs from ${refdigest}")
      set(${resultvar} OFF PARENT_SCOPE)
    endif()
  endforeach()
endfunction()

