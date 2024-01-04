# ==========================================
# === exaStamp/exaDEM/exaSPH application ===
# ==========================================

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${USTAMP_APPS_DIR})

file(GLOB USTAMP_MAIN_SRCS *.cpp)

#message(STATUS "USTAMP_CORE_LIBRARIES=${USTAMP_CORE_LIBRARIES}")

add_executable(${XNB_APP_NAME} ${USTAMP_MAIN_SRCS})
target_compile_options(${XNB_APP_NAME} PUBLIC ${USTAMP_CXX_FLAGS}) # $<$<COMPILE_LANGUAGE:CUDA>:${ONIKA_CUDA_COMPILE_FLAGS}>)
target_compile_definitions(${XNB_APP_NAME} PUBLIC ${USTAMP_COMPILE_DEFINITIONS} ${XSTAMP_CUDA_COMPILE_DEFINITIONS})
#target_compile_features(${XNB_APP_NAME} PUBLIC cuda_std_17)
target_include_directories(${XNB_APP_NAME} PUBLIC ${USTAMP_INCLUDE_DIRS})
target_link_libraries(${XNB_APP_NAME} exanbMPI ${USTAMP_CORE_LIBRARIES})
#set_target_properties(${XNB_APP_NAME} PROPERTIES CUDA_RUNTIME_LIBRARY Shared CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS ${XNB_APP_NAME} DESTINATION bin)

