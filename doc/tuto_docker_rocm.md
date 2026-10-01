# Building onika and exaNBody with HIP in a Docker/Podman container

This documents how to compile `onika` and `exaNBody` for AMD GPUs (HIP,
`gfx90a`) using the `rocm/dev-ubuntu-22.04` image, on a machine without an
AMD GPU. Compilation only needs the ROCm/HIP toolchain — not the actual
hardware. The same steps are used by the `cmake_hip.yml` (onika) and
`hip.yml` (exaNBody) CI workflows.

Every command below works with either `docker` or `podman`; podman-specific
details are called out where they differ.

## 1. Pull the image

```bash
docker pull rocm/dev-ubuntu-22.04:latest
# or
podman pull docker.io/rocm/dev-ubuntu-22.04:latest
```

With podman, use the fully qualified name (`docker.io/...`), otherwise podman
may prompt for, or refuse, short-name resolution.

If you're behind a proxy, Docker needs the daemon configured via a systemd
drop-in at `/etc/systemd/system/docker.service.d/http-proxy.conf`:

```ini
[Service]
Environment="HTTP_PROXY=http://proxy.example.com:8080"
Environment="HTTPS_PROXY=http://proxy.example.com:8080"
Environment="NO_PROXY=localhost,127.0.0.1"
```

then `sudo systemctl daemon-reload && sudo systemctl restart docker`.
Podman is daemonless: it simply uses the `http_proxy`/`https_proxy`
variables of the shell running `podman pull`.

## 2. Launch the container

Source trees and the build directory are mounted at the **same paths** as on
the host, so absolute paths in build scripts work unchanged:

```bash
ONIKA_SRC_DIR=/path/to/onika
XNB_SRC_DIR=/path/to/exaNBody
HIP_WORK_DIR=${HOME}/hipdir          # build scripts, build and install trees
mkdir -p ${HIP_WORK_DIR}

docker run -it --rm \
  -e HOME=${HOME} \
  -v ${HIP_WORK_DIR}:${HIP_WORK_DIR} \
  -v ${ONIKA_SRC_DIR}:${ONIKA_SRC_DIR}:ro \
  -v ${XNB_SRC_DIR}:${XNB_SRC_DIR}:ro \
  -w ${HIP_WORK_DIR} \
  rocm/dev-ubuntu-22.04:latest bash
```

With podman, replace `docker run` by `podman run`, use the fully qualified
image name, and on SELinux hosts (RHEL, Fedora, Rocky, ...) add
`--security-opt label=disable`, otherwise the container gets
`Permission denied` on the mounted directories:

```bash
podman run -it --rm --security-opt label=disable \
  -e HOME=${HOME} \
  -v ${HIP_WORK_DIR}:${HIP_WORK_DIR} \
  -v ${ONIKA_SRC_DIR}:${ONIKA_SRC_DIR}:ro \
  -v ${XNB_SRC_DIR}:${XNB_SRC_DIR}:ro \
  -w ${HIP_WORK_DIR} \
  docker.io/rocm/dev-ubuntu-22.04:latest bash
```

Rootless podman maps the container's `root` to your own user, so files
written to `${HIP_WORK_DIR}` belong to you on the host.

## 3. Install CMake, yaml-cpp and MPI inside the container

The `rocm/dev-ubuntu-22.04` image ships **none of CMake, yaml-cpp or MPI**,
and all three are required. The container runs as root, so no `sudo` is
needed. Behind a proxy, export `http_proxy`/`https_proxy` inside the
container first (the Docker daemon proxy settings do not propagate to it).

### CMake (Kitware apt repository)

Ubuntu 22.04's CMake (3.22) is older than `onika`'s
`cmake_minimum_required` (3.26). Kitware's apt repository provides a recent
`cmake` together with `ccmake`:

```bash
apt-get update && apt-get install -y wget gpg
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc \
  | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main" \
  | tee /etc/apt/sources.list.d/kitware.list
apt-get update
apt-get install -y cmake cmake-curses-gui
cmake --version
```

(`pip install cmake` also gives a recent `cmake`, but its `ccmake` is a
non-working stub.)

### yaml-cpp and MPI

Use the distribution packages; `find_package(yaml-cpp)` and
`find_package(MPI)` then find them without any extra CMake flag:

```bash
apt-get install -y libyaml-cpp-dev libopenmpi-dev openmpi-bin
```

## 4. Build

### 4.1 Pin HIP clang to the C++ compiler's GCC

`.cu` files are compiled by ROCm's clang, which has no C++ standard library
of its own: it uses the `libstdc++` of whatever GCC installation it detects,
which is not necessarily the `g++` used for C++ sources. If it picks a GCC
older than 10, C++20 headers are missing and the build fails with:

```
onika/oarray.h:25:10: fatal error: 'compare' file not found
1 error generated when compiling for gfx90a.
```

Always point clang at the GCC installation of `g++` explicitly:

```bash
GCC_INSTALL_DIR=$(dirname $(g++ -print-libgcc-file-name))
echo ${GCC_INSTALL_DIR}   # e.g. /usr/lib/gcc/x86_64-linux-gnu/11
```

and pass `-DCMAKE_HIP_FLAGS="--gcc-install-dir=${GCC_INSTALL_DIR}"` to
**every** configure step below: `onika` does not export `CMAKE_HIP_FLAGS` to
projects built on top of it.

### 4.2 Build onika

```bash
#!/bin/bash
ONIKA_SRC_DIR=/path/to/onika
ONIKA_BUILD_DIR=${HOME}/hipdir/build_onika_hip
ONIKA_INSTALL_DIR=${HOME}/hipdir/onika_hip
GCC_INSTALL_DIR=$(dirname $(g++ -print-libgcc-file-name))

mkdir -p ${ONIKA_BUILD_DIR}
cd ${ONIKA_BUILD_DIR}
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=${ONIKA_INSTALL_DIR} \
      -DONIKA_BUILD_CUDA=ON \
      -DONIKA_ENABLE_HIP=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DCMAKE_HIP_FLAGS="--gcc-install-dir=${GCC_INSTALL_DIR}" \
      -DONIKA_HAS_GPU_ATOMIC_MIN_MAX_DOUBLE=ON \
      ${ONIKA_SRC_DIR}
make -j8 install
```

### 4.3 Build exaNBody

```bash
#!/bin/bash
XNB_SRC_DIR=/path/to/exaNBody
XNB_BUILD_DIR=${HOME}/hipdir/build_exanbody_hip
XNB_INSTALL_DIR=${HOME}/hipdir/exaNBody_hip
ONIKA_INSTALL_DIR=${HOME}/hipdir/onika_hip
GCC_INSTALL_DIR=$(dirname $(g++ -print-libgcc-file-name))

mkdir -p ${XNB_BUILD_DIR}
cd ${XNB_BUILD_DIR}
cmake -DCMAKE_BUILD_TYPE=Release \
      -Donika_DIR=${ONIKA_INSTALL_DIR} \
      -DCMAKE_INSTALL_PREFIX=${XNB_INSTALL_DIR} \
      -DCMAKE_HIP_FLAGS="--gcc-install-dir=${GCC_INSTALL_DIR}" \
      -DONIKA_USE_HIP=ON \
      -DEXANB_BUILD_CONTRIB_MD=ON \
      -DEXANB_BUILD_MICROSTAMP=ON \
      -DEXANB_BUILD_CONTRIB_PI=ON \
      -DEXANB_BUILD_MICROCOSMOS=ON \
      -DSNAP_CPU_USE_LOCKS=ON \
      -DSNAP_FP32_MATH=OFF \
      ${XNB_SRC_DIR}
make -j8 install
```

`-DONIKA_USE_HIP=ON` is needed as long as the installed `onika-config.cmake`
does not export it (check with
`grep ONIKA_USE_HIP ${ONIKA_INSTALL_DIR}/onika-config.cmake`). Without it,
`.cu` files are treated as CUDA sources, and since only the HIP language is
enabled CMake **silently drops them**: the build succeeds, but every GPU
kernel operator (`ghost_update_all`, snap, ...) is missing at runtime.

### 4.4 Check that the `.cu` files were actually compiled

A successful build does not prove it, so check for the object files:

```bash
find ${HOME}/hipdir/build_onika_hip    -name "*.cu.o"   # gpu_reduce_min_fp64, gpu_uvm_benchmark, parallel_queue_lane
find ${HOME}/hipdir/build_exanbody_hip -name "*.cu.o"   # update_ghosts, snap_force_generic, gravitational_force, ...
```

Both lists must be non-empty.

If a configure step fails (e.g. a missing dependency), remove the build
directory before trying again so CMake does not reuse a stale cache.

## Notes

Compiling for `gfx90a` does not require an AMD GPU on the build machine —
only the HIP/ROCm compiler toolchain (provided by the `rocm/dev-*` image)
and an explicit target architecture. Running the result does require one.
