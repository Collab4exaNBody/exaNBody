
option(EXANB_BUILD_MICROSTAMP "Build microStamp (MD) miniapp" OFF)
if(EXANB_BUILD_MICROSTAMP)
  add_subdirectory(microStamp)
endif()

