# ===============
# === Project ===
# ===============
cmake_minimum_required(VERSION 3.16)
project(exaNBody LANGUAGES C CXX)

get_filename_component(XNB_ROOT_DIR ${CMAKE_INSTALL_PREFIX} ABSOLUTE)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/exanbody-config.cmake "set(exaNBody_DIR ${XNB_ROOT_DIR})\nlist(APPEND CMAKE_MODULE_PATH \"${XNB_ROOT_DIR}/cmake\")\ninclude(exaNBody)\n")

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/exanbody-config.cmake DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY src cmake thirdparty scripts DESTINATION ${CMAKE_INSTALL_PREFIX})

