# ===============
# === Project ===
# ===============
cmake_minimum_required(VERSION 3.16)
project(exaNBody LANGUAGES C CXX)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/exanbody-config.cmake "set(exaNBody_DIR ${CMAKE_INSTALL_PREFIX})\nlist(APPEND CMAKE_MODULE_PATH \"${CMAKE_INSTALL_PREFIX}/cmake\")\ninclude(exaNBody)\n")

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/exanbody-config.cmake DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY src cmake thirdparty DESTINATION ${CMAKE_INSTALL_PREFIX})

