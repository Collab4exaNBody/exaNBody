# ===================================
# === micro-stamp Snap potential  ===
# ===================================

add_executable(MakeSnapAutoGen snap_gen.cxx)

set(mdSnap_GENERATED_SOURCES_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(mdSnap_GENERATED_SOURCES_OUTPUT_DIR ${mdSnap_GENERATED_SOURCES_INCLUDE_DIR}/md/snap)
file(MAKE_DIRECTORY ${mdSnap_GENERATED_SOURCES_OUTPUT_DIR})

set(mdSnap_GENERATED_SOURCES
  ${mdSnap_GENERATED_SOURCES_OUTPUT_DIR}/compute_ui_jmax2.hxx
  ${mdSnap_GENERATED_SOURCES_OUTPUT_DIR}/compute_ui_jmax3.hxx
  ${mdSnap_GENERATED_SOURCES_OUTPUT_DIR}/compute_ui_jmax4.hxx
  ${mdSnap_GENERATED_SOURCES_OUTPUT_DIR}/ymap_jmax2.hxx
  ${mdSnap_GENERATED_SOURCES_OUTPUT_DIR}/ymap_jmax3.hxx
  ${mdSnap_GENERATED_SOURCES_OUTPUT_DIR}/ymap_jmax4.hxx)

add_custom_target(SnapAutoGen
  COMMAND MakeSnapAutoGen ${mdSnap_GENERATED_SOURCES_OUTPUT_DIR}
  DEPENDS MakeSnapAutoGen
  BYPRODUCTS ${mdSnap_GENERATED_SOURCES})

set(mdSnap_LINK_LIBRARIES exanbCompute exanbParticleNeighbors exanbCore)
set(mdSnap_INCLUDE_DIRS ${mdSnap_GENERATED_SOURCES_INCLUDE_DIR})
set(ONIKA_BUILD_mdSnap_DEFAULT ON)
set(ONIKA_BUILD_mdSnap_FORCED ON)
xnb_add_plugin(mdSnap ${CMAKE_CURRENT_SOURCE_DIR})
add_dependencies(${mdSnap_PLUGIN_LIB} SnapAutoGen)
install(DIRECTORY ${mdSnap_GENERATED_SOURCES_INCLUDE_DIR}/md TYPE INCLUDE)

