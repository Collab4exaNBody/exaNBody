# direct in source inclusion of exaNBody, for developpers only

set(exaNBody_DIR ${CMAKE_CURRENT_LIST_DIR})
list(APPEND CMAKE_MODULE_PATH "${exaNBody_DIR}/cmake")
include(exaNBody)

