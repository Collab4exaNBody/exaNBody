add_subdirectory(miniapps)

option(EXANB_BUILD_CONTRIB_MD "Build Molecular Dynamics contribs " OFF)
if(EXANB_BUILD_CONTRIB_MD)
  add_subdirectory(md)
endif()


