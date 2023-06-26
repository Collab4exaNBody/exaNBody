# ===============
# === Project ===
# ===============
cmake_minimum_required(VERSION 3.16)
project(exaNBody LANGUAGES CXX)

set(XNB_ROOT_DIR ${CMAKE_SOURCE_DIR})
list(APPEND CMAKE_MODULE_PATH "${XNB_ROOT_DIR}/cmake")
include(exaNBody)

# install ...


