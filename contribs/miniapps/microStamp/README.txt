#
# Ubuntu 22.04 g++-11.4.0 X Cuda 12
# Ubuntu 22.04 g++-11.4.0 (no Cuda)
#
XNB_INSTALL_DIR=${HOME}/local/exaNBody
MICROSTAMP_SRC=${HOME}/dev/exaNBody/contribs/miniapps/microStamp
MICROSTAMP_INSTALL_DIR=${HOME}/local/microStamp
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=${MICROSTAMP_INSTALL_DIR} \
      -DexaNBody_DIR=${XNB_INSTALL_DIR} \
	    ${MICROSTAMP_SRC}


