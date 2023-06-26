# ==========================================
# === exaStamp/exaDEM/exaSPH application ===
# ==========================================

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${USTAMP_APPS_DIR})

file(GLOB USTAMP_MAIN_SRCS *.cpp)

#message(STATUS "USTAMP_CORE_LIBRARIES=${USTAMP_CORE_LIBRARIES}")

add_executable(${XNB_APP_NAME} ${USTAMP_MAIN_SRCS})
target_compile_options(${XNB_APP_NAME} PUBLIC ${USTAMP_CXX_FLAGS})
target_compile_definitions(${XNB_APP_NAME} PUBLIC ${USTAMP_COMPILE_DEFINITIONS} ${XSTAMP_CUDA_COMPILE_DEFINITIONS})
target_include_directories(${XNB_APP_NAME} PUBLIC ${USTAMP_INCLUDE_DIRS})
target_link_libraries(${XNB_APP_NAME} exanbMPI ${USTAMP_CORE_LIBRARIES})

install(TARGETS ${XNB_APP_NAME} DESTINATION bin)

