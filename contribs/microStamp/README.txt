# Mini-app build process from scratch

# 1. first, download and build onika, our HPC building blocks platform
git clone -b main https://github.com/Collab4exaNBody/onika.git

# follow instructions in onika/doc/build-examples.txt
# if needed, build a compatible version of yaml-cpp with instructions in onika/doc/yaml-build.txt
# when everything is ready, compile and install
make -j4 install


# 2. download and build exaNBody (using branch release-2.0)
git clone -b release-2.0 https://github.com/Collab4exaNBody/exaNBody.git
# to configure exaNBody follow instructions in exaNBody/doc/build-examples.txt
# when configuring with CMake, it is important to enable EXANB_BUILD_MICROSTAMP
make -j4 install


# once built, from the build directory of exaNBody, you can run some tests
# dry run without user input file (just test the platform is ok)
./exaNBody /dev/null

# test a simple simulation using the microStamp mini-app
./exaNBody path-to-exaNBody-source/contribs/miniapps/microStamp/samples/lattice_sphere.msp

