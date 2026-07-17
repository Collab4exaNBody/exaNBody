# Building ONIKA with HIP in a Docker container

This documents how to compile `onika` for AMD GPUs (HIP, `gfx90a`) using the
`rocm/dev-ubuntu-22.04` image, on a machine without an AMD GPU. Compilation
only needs the ROCm/HIP toolchain — not the actual hardware.

## 1. Pull the image

```bash
docker pull rocm/dev-ubuntu-22.04:latest
```

If you're behind a proxy, configure the Docker daemon first via a systemd
drop-in at `/etc/systemd/system/docker.service.d/http-proxy.conf`:

```ini
[Service]
Environment="HTTP_PROXY=http://proxy.example.com:8080"
Environment="HTTPS_PROXY=http://proxy.example.com:8080"
Environment="NO_PROXY=localhost,127.0.0.1"
```

then `sudo systemctl daemon-reload && sudo systemctl restart docker`.

## 2. Launch the container

```bash
docker run -it --rm \
  -e HOME=/home/lafourcadep \
  -v ${HOME}$/hipdir:${HOME}hipdir \
  -v /path/to/onika:/path/to/onika/onika:ro \
  -v /path/to/exaNBody:/path/to/exaNBody:ro \  
  -v ${HOME}/local:${HOME}/local \
  -w ${HOME}/hipdir \
  rocm/dev-ubuntu-22.04:latest bash
```

This mounts, at matching host paths so the build script's absolute paths work
unchanged:
- `hipdir` (read-write, holds the build script and build directory)
- `onika` source tree (read-only)
- `~/local` (read-write, receives the `onikaGPU_HIP` install output —
  yaml-cpp is installed via apt inside the container, see step 3, so no
  custom yaml-cpp install dir needs to be mounted)

## 3. Install CMake, yaml-cpp and MPI inside the container

The `rocm/dev-ubuntu-22.04` base image ships **none of CMake, yaml-cpp, or
MPI**, and `onika`'s `CMakeLists.txt` requires all three (the build script
also calls `/usr/local/bin/cmake` specifically — CMake 3.26 on the host).

Since you're behind a proxy, the container also needs proxy env vars for
`apt`/`pip` to reach the internet — the daemon-level proxy config from step 1
does not propagate into the container's shell:

```bash
export http_proxy=http://proxy.example.com:8080
export https_proxy=http://proxy.example.com:8080
```

### CMake and ccmake (Kitware apt repo — recommended)

Ubuntu 22.04's repo CMake is 3.22 (too old for `onika`'s
`cmake_minimum_required` ≥ 3.26), and the `cmake` PyPI wheel doesn't ship a
working `ccmake` at all (it needs `ncurses`, which doesn't fit a portable
wheel — installing it produces a broken stub that errors with `'ccmake' is
not yet included in the package for this platform`).

The clean fix is **Kitware's official apt repository**, which provides a
recent, matching `cmake` + `ccmake` (+ `cpack`/`ctest`) built together with
ncurses support — the same approach used in the project's `cmake.yml` CI
workflow:

```bash
apt-get update && apt-get install -y wget gpg
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc \
  | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main" \
  | tee /etc/apt/sources.list.d/kitware.list
apt-get update
apt-get install -y cmake cmake-curses-gui
cmake --version
ccmake --version
```

This installs both binaries at `/usr/bin/{cmake,ccmake}` at the same recent
version (≥ 3.26, matching/exceeding the host's 3.26). Since the build script
hardcodes `/usr/local/bin/cmake`, either edit that path in
`build_install_onika_gpu.sh` to plain `cmake` (resolved via `PATH`), or
symlink it:

```bash
ln -sf $(command -v cmake) /usr/local/bin/cmake
```

<details>
<summary>Alternative: pip (gets a recent <code>cmake</code>, but no working
<code>ccmake</code> — expand if you only need <code>cmake</code>)</summary>

```bash
apt-get update && apt-get install -y python3-pip   # if pip isn't present
pip3 install --upgrade cmake
cmake --version
```

This lands at `/usr/local/bin/cmake` (matching the build script's hardcoded
path) but you'd still need `ccmake` from the Kitware repo or `cmake-curses-gui`
— and in that case, remove the broken pip stub first so it doesn't shadow the
working binary:

```bash
rm -f /usr/local/bin/ccmake
hash -r
```
</details>

### yaml-cpp

The build script originally pointed `yaml-cpp_DIR` at a custom
`yaml-cpp-0.6.3` install (whose CMake config actually lives in a
`lib/cmake/yaml-cpp` subdirectory — a path mismatch that makes
`find_package(yaml-cpp)` fail with `Could not find a package configuration
file provided by "yaml-cpp"`). Simplest fix: skip the custom install
entirely and use the distro package, exactly as the project's `cmake.yml` CI
workflow does:

```bash
apt-get update && apt-get install -y libyaml-cpp-dev
```

This installs yaml-cpp's headers, libs, and CMake config files to standard
system paths, where `find_package(yaml-cpp)` locates them automatically — no
`-Dyaml-cpp_DIR=...` needed. **Remove (or leave empty) the
`YAML_CPP_INSTALL_DIR`/`-Dyaml-cpp_DIR=...` lines in
`build_install_onika_gpu.sh`** so CMake falls back to this system install.

### MPI

`CMakeLists.txt` calls `find_package(MPI)`, which fails
(`Could NOT find MPI (missing: MPI_C_FOUND MPI_CXX_FOUND)`) unless an MPI
implementation is installed. Install OpenMPI via apt:

```bash
apt-get update && apt-get install -y libopenmpi-dev openmpi-bin
```

This provides `mpicc`/`mpicxx` and the headers/libs that CMake's `FindMPI`
module looks for — it's auto-detected on the next `cmake` run, no extra flags
needed.

## 4. Run the build script

### 4.1 Build onika

```bash
#!/bin/bash

mkdir build_onika_gpu
cd build_onika_gpu

ONIKA_SRC_DIR=/path/to/onika
ONIKA_INSTALL_DIR=${HOME}/local/onikaGPU_HIP
ONIKA_SETUP_ENV_COMMANDS=""
eval ${ONIKA_SETUP_ENV_COMMANDS}
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=${ONIKA_INSTALL_DIR} \
      -DONIKA_BUILD_CUDA=ON \
      -DONIKA_ENABLE_HIP=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DONIKA_HAS_GPU_ATOMIC_MIN_MAX_DOUBLE=ON \
      -DONIKA_SETUP_ENV_COMMANDS="${ONIKA_SETUP_ENV_COMMANDS}" \
      ${ONIKA_SRC_DIR}
make -j8 install
cd ../
```

### 4.2 Build exaNBody

mkdir build_exanbody_gpu
cd build_exanbody_gpu

```bash
XNB_SRC_DIR=/path/to/exanobdy
XNB_INSTALL_DIR=${HOME}/local/exaNBodyGPU_HIP
XNB_SETUP_ENV_COMMANDS=""
eval ${ONIKA_SETUP_ENV_COMMANDS}

cmake -DCMAKE_BUILD_TYPE=Release \
      -Donika_DIR=${ONIKA_INSTALL_DIR} \
      -DCMAKE_INSTALL_PREFIX=${XNB_INSTALL_DIR} \ 
      -DEXANB_BUILD_CONTRIB_MD=ON \ 
      -DEXANB_BUILD_MICROSTAMP=ON \ 
      -DEXANB_BUILD_CONTRIB_PI=ON \ 
      -DEXANB_BUILD_MICROCOSMOS=ON \ 
      -DSNAP_CPU_USE_LOCKS=ON \ 
      -DSNAP_FP32_MATH=OFF \ 
      ${XNB_SRC_DIR}
```

## Notes

Compiling for `gfx90a` does not require an AMD GPU on the build machine —
only the HIP/ROCm compiler toolchain (provided by the `rocm/dev-*` image)
and an explicit target architecture.

