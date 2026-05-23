# pyexanbody — Python interface to the exaNBody N-body simulation framework

`pyexanbody` is a pure-Python package that lets you run exaNBody simulations from Python. It wraps `pyonika` (the onika C++ bindings) and ensures that all exaNBody operators are loaded before the first `init()` call. The full `pyonika` API is re-exported, so you only ever import `pyexanbody`.

---

## Table of contents

1. [How it works](#how-it-works)
2. [Prerequisites](#prerequisites)
3. [Build and install](#build-and-install)
4. [Quick start](#quick-start)
5. [Running an installed example](#running-an-installed-example)
6. [Running an arbitrary .msp file](#running-an-arbitrary-msp-file)
7. [Building a simulation graph from Python](#building-a-simulation-graph-from-python)
8. [Example scripts](#example-scripts)
9. [API reference](#api-reference)
   - [pyexanbody-specific helpers](#pyexanbody-specific-helpers)
   - [Inherited pyonika API](#inherited-pyonika-api)
10. [Implementation notes](#implementation-notes)
11. [Source files](#source-files)

---

## How it works

exaNBody is **not** a separate executable — it is a set of shared-library plugins that extend the onika runtime. The same `onika-exec` binary runs both onika-only and exaNBody simulations; the only difference is which directories are on `ONIKA_PLUGIN_PATH` and where `ONIKA_CONFIG_PATH` points.

`import pyexanbody` does four things before any user code runs:

| Step | What happens |
|---|---|
| `_ensure_env()` | Sets `ONIKA_PLUGIN_PATH`, `ONIKA_CONFIG_PATH`, `LD_LIBRARY_PATH`, `ONIKA_DATA_PATH`, `sys.path` — mirrors `setup-env.sh` |
| `import pyonika` | Loads the C++ extension; `libonika.so` and `libmpi.so` enter the process |
| `_ensure_mpi_external()` | Pre-initialises MPI via ctypes so onika treats it as externally owned — `end()` never calls `MPI_Finalize()`, enabling multiple `init/run/end` cycles |
| re-exports | All `pyonika` symbols are re-exported at the `pyexanbody` namespace level |

Once these are in place, `pyonika.init()` loads all 14 exaNBody plugins and registers their operators (domain management, AMR grid, particle neighbours, I/O, MD potentials, gravitational force, …) alongside the onika built-ins.

---

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| onika | installed | `install-onika/` must exist with `pyonika.<platform>.so` in `lib/` |
| exaNBody | installed | `install-exanbody/` must exist |
| Python 3 | ≥ 3.8 | |
| numpy | ≥ 1.20 | Required by pyonika at import time |

---

## Build and install

Enable `EXANB_BUILD_PYTHON` when configuring exaNBody. No C++ compilation is performed — CMake only generates `_paths.py` (which bakes in the install paths) and installs the Python files.

Also enable `EXANB_BUILD_MICROSTAMP` to install the potential files (`share/potentials/`) which are required by the MD and SNAP examples.

```bash
cmake -B build-exanbody \
      -DCMAKE_INSTALL_PREFIX=/path/to/install-exanbody \
      -Donika_DIR=/path/to/install-onika \
      -DEXANB_BUILD_PYTHON=ON \
      -DEXANB_BUILD_MICROSTAMP=ON \
      exaNBody/
cmake --build build-exanbody
cmake --install build-exanbody
```

The installed layout is:

```
install-exanbody/
├── bin/
│   └── setup-env.sh              ← source this; sets PYTHONPATH in addition to other vars
├── lib/
│   ├── libexanbCore.so           ← needed at dlopen time by exaNBody plugins
│   └── pyexanbody/
│       ├── __init__.py
│       └── _paths.py             ← generated: ONIKA_INSTALL_DIR, XNB_INSTALL_DIR
├── plugins/                      ← 14 exaNBody plugin .so files
├── data/config/
│   └── main-config.msp           ← exaNBody default config (domain, compute-loop, …)
├── share/
│   ├── examples/                 ← installed .msp examples (microStamp, microCosmos, …)
│   ├── microCosmos_examples/
│   ├── microStamp_examples/
│   └── potentials/               ← SNAP/LJ potential files, found automatically via ONIKA_DATA_PATH
└── python/
    ├── README.md                 ← this file
    └── exemples/
        ├── test.py
        ├── run_solar_system.py
        └── solar_system_graph.py
```

After install, source the environment script:

```bash
source /path/to/install-exanbody/bin/setup-env.sh
python install-exanbody/python/exemples/test.py
```

---

## Quick start

```python
import os, sys
import pyexanbody as xnb

main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")

ctx = xnb.init([sys.argv[0], main_config])
if ctx.error_code >= 0:
    sys.exit(ctx.error_code)

# The full exaNBody operator catalogue is available
print(xnb.available_operators())

xnb.run(ctx)
xnb.end(ctx)

# init/run/end can be called again — MPI stays alive across calls
```

---

## Running an installed example

Use `run_example()` to run any file from `install-exanbody/share/` by bare
filename — no path needed:

```python
import pyexanbody as xnb

xnb.run_example("solar_system.msp")
xnb.run_example("deterministic_lj.msp")
xnb.run_example("snap_from_lattice.msp")   # potential files found automatically
```

`list_examples()` shows all available examples:

```python
xnb.list_examples()
# ['benchmark_lj_snap/input_lj_Ni.msp',
#  'deterministic/deterministic_lj.msp',
#  ...
#  'simple_example/solar_system.msp', ...]
```

`find_example()` resolves a name to its full path without running it:

```python
path = xnb.find_example("solar_system.msp")
# → '.../install-exanbody/share/examples/simple_example/solar_system.msp'
```

---

## Running an arbitrary .msp file

Use `run_file()` to run any `.msp` file by path:

```python
xnb.run_file("my_sim.msp")
```

Optional onika overrides can be passed as keyword arguments using `__` as the dotted-key separator:

```python
xnb.run_file("my_sim.msp",
             configuration__omp_num_threads=4,
             global__simulation_end_iteration=50)
```

---

## Building a simulation graph from Python

The full `pyonika` graph-building API is available. This lets you reproduce any `.msp` file in pure Python without writing configuration files. See `exemples/solar_system_graph.py` for a complete working example.

**Pattern A — run a graph from an existing `.msp` file:**

```python
import sys
import pyexanbody as xnb

ctx = xnb.init([sys.argv[0], "my_sim.msp"])
if ctx.error_code >= 0:
    sys.exit(ctx.error_code)
xnb.run(ctx)
xnb.end(ctx)
```

**Pattern B — build the graph entirely from Python:**

```python
import os, sys
import pyexanbody as xnb

# 1. Bootstrap MPI, OpenMP, plugins and operator definitions.
main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")
ctx = xnb.init([sys.argv[0], main_config])
if ctx.error_code >= 0:
    sys.exit(ctx.error_code)

# 2. Override named operator defaults — equivalent to top-level .msp keys.
xnb.set_operator_defaults({
    "global": {
        "dt": "1 s",
        "rcut_max": "280 m",
        "simulation_end_iteration": 100,
    },
    "input_data": [
        {"domain": {"cell_size": "150 m", "grid_dims": [13, 13, 13], ...}},
        "init_rcb_grid",
        {"lattice": {"structure": "BCC", "types": ["Atom"], "size": [...]}},
    ],
    "compute_force": [
        {"gravitational_force": {"config": {"G": "6.67e-11 m^3/kg/s^2"}, "rcut": "280 m"}}
    ],
})

# 3. Build and run the graph — equivalent to: simulation: default_simulation
graph = xnb.build_simulation_graph(ctx, ["default_simulation"])
xnb.run_node(ctx, graph)
xnb.end(ctx)
```

**`.msp` → Python mapping:**

| `.msp` top-level key | Python equivalent |
|---|---|
| `global: { dt: 1 s, … }` | `set_operator_defaults({"global": {"dt": "1 s", …}})` |
| `input_data: [ … ]` | `set_operator_defaults({"input_data": [ … ]})` |
| `compute_force: [ … ]` | `set_operator_defaults({"compute_force": [ … ]})` |
| `dump_data: dump_data_paraview` | `set_operator_defaults({"dump_data": "dump_data_paraview"})` |
| `simulation: default_simulation` | `build_simulation_graph(ctx, ["default_simulation"])` |

> **`run` vs `run_node`:** use `run(ctx)` when `init()` was given a full `.msp` file and you want to execute its graph. Use `run_node(ctx, graph)` when you built the graph yourself with `build_simulation_graph()`, because that graph is not stored inside `ctx`.

---

## Example scripts

All scripts assume `setup-env.sh` has been sourced.

### `test.py`

Minimal smoke test. Imports `pyexanbody`, initialises onika+exaNBody from `main-config.msp`, checks that exaNBody operators are present in the operator catalogue, runs the default `test_simulation` graph, and exits.

```bash
python exemples/test.py
# pyexanbody: 213 operators loaded
# pyexanbody: exaNBody operators visible (e.g. ['domain', 'gravitational_force', 'lattice'])
# pyexanbody: OK
```

### `run_solar_system.py`

Runs the gravitational N-body solar-system example using `run_example()`. Accepts an optional `.msp` path as the first argument.

```bash
python exemples/run_solar_system.py
python exemples/run_solar_system.py /path/to/my_sim.msp
```

### `solar_system_graph.py`

Full Python reproduction of `solar_system.msp` without reading the `.msp` file. Demonstrates the `set_operator_defaults()` + `build_simulation_graph()` pattern with a real exaNBody simulation:

- calls `init()` with `main-config.msp` only — to bootstrap MPI, plugins and default operator definitions
- overrides `global`, `input_data`, `compute_force` and `dump_data` via `set_operator_defaults()`
- builds the full simulation graph from Python with `build_simulation_graph(ctx, ["default_simulation"])`
- runs and finalises

Use this as the starting point when you want to drive an exaNBody simulation entirely from Python.

```bash
python exemples/solar_system_graph.py
```

---

## API reference

### pyexanbody-specific helpers

#### `pyexanbody.run_example(name, argv0="pyexanbody", **extra_args) -> int`

Resolve an installed example by bare filename and run it end-to-end. Searches all subdirectories of `XNB_SHARE_DIR` (`install-exanbody/share/`).

```python
xnb.run_example("solar_system.msp")
xnb.run_example("deterministic_lj.msp", global__simulation_end_iteration=10)
```

#### `pyexanbody.find_example(name) -> str`

Resolve an installed example name to its absolute path without running it. Raises `FileNotFoundError` if not found.

```python
path = xnb.find_example("solar_system.msp")
```

#### `pyexanbody.list_examples() -> list[str]`

Return a sorted list of all installed example `.msp` files as paths relative to their share subdirectory. Each entry can be passed directly to `find_example()` or `run_example()`.

```python
xnb.list_examples()
# ['benchmark_lj_snap/input_lj_Ni.msp', 'deterministic/deterministic_lj.msp', ...]
```

#### `pyexanbody.XNB_SHARE_DIR`

Absolute path to `install-exanbody/share/`. Useful for building data file paths manually.

#### `pyexanbody.run_file(msp_path, argv0="pyexanbody", **extra_args) -> int`

Run a complete simulation from an `.msp` file by path and return 0 on normal completion, or the early-exit code from onika.

```python
rc = xnb.run_file("my_sim.msp")
rc = xnb.run_file("my_sim.msp", configuration__omp_num_threads=4)
```

### Inherited pyonika API

`pyexanbody` re-exports the complete `pyonika` public API. See `onika/python/README.md` for the full reference. The key entry points are:

| Function / type | Description |
|---|---|
| `init(argv)` → `ApplicationContext` | Bootstrap onika+exaNBody from an `.msp` file |
| `run(ctx)` | Execute the graph built by `init()` |
| `run_node(ctx, node)` | Execute a Python-built graph |
| `end(ctx)` | Finalise and free resources (MPI is never finalised — safe to call again) |
| `make_operator(name, config={})` | Instantiate a registered operator by name |
| `available_operators()` | List all registered operators (onika + exaNBody) |
| `set_operator_defaults(dict)` | Register named operator defaults (top-level `.msp` keys) |
| `build_simulation_graph(ctx, list)` | Build a simulation graph from a Python list |
| `slot_as_array(slot)` | Zero-copy numpy view of a slot value |
| `ApplicationContext` | Holds full simulation state; provides `node()`, `mpi_rank`, … |
| `OperatorNode` | Node in the simulation graph; provides `in_slots()`, `slot_as_array()`, … |
| `OperatorSlotBase` | One input/output slot; provides `value_as_string()`, `yaml_initialize()`, … |
| `OnikaError` | Raised for unknown operator names or slot type mismatches |

---

## Implementation notes

### Why no C++ is needed

exaNBody operators are registered in the onika `OperatorNodeFactory` at plugin load time (inside each `lib*-plugin.so`). The factory is the same object that `pyonika.available_operators()` and `pyonika.make_operator()` query. There is therefore nothing to bind — once the plugins are loaded, all exaNBody operators are visible through the existing pyonika API.

### MPI lifecycle and multiple `init/run/end` cycles

`onika::app::initialize_mpi()` calls `MPI_Initialized()` at startup. If MPI is already up, it sets `external_mpi_init = true` and onika's `end()` skips `MPI_Finalize()`. If onika initialised MPI itself, `end()` finalises it — making a second `init()` call fatal.

`_ensure_mpi_external()` is called at `import pyexanbody` time, after `import pyonika` has loaded `libmpi.so` into the process. It locates the MPI library via `/proc/self/maps`, then calls `MPI_Init()` if MPI is not yet initialised. From that point on, every subsequent `pyonika.init()` sees `MPI_Initialized() == 1` and sets `external_mpi_init = true`, so `end()` never finalises MPI. Multiple `init/run/end` cycles work safely in the same Python session.

### `_ensure_env()` and idempotency

`_ensure_env()` is called once at `import pyexanbody` time. It only modifies env vars and `sys.path` when the required entries are missing, so it is safe to call multiple times and does not clobber entries already set by `setup-env.sh`. If `setup-env.sh` was sourced before `import pyexanbody`, `_ensure_env()` is effectively a no-op.

### `LD_LIBRARY_PATH` and `dlopen`

Setting `os.environ["LD_LIBRARY_PATH"]` in Python before calling `pyonika.init()` works because the Linux dynamic linker reads `LD_LIBRARY_PATH` from the process environment at each `dlopen()` call. onika's plugin loader calls `dlopen()` during `init()`, so prepending `install-exanbody/lib` ensures `libexanbCore.so` (a link-time dependency of the exaNBody plugins) is found.

### Example discovery, potential files and `ONIKA_DATA_PATH`

`ONIKA_CONFIG_PATH` is a **single directory** — onika's `config_file_path()` searches only that one dir plus the working directory. It cannot be a colon-separated list.

`ONIKA_DATA_PATH` is a **colon-separated list** searched by `data_file_path()`, which operators call when opening data files (particle dumps, potential files, external masks, etc.). `_ensure_env()` appends all first-level subdirectories of `XNB_SHARE_DIR` to `ONIKA_DATA_PATH`. This automatically covers:

- `share/examples/` — data files referenced inside example `.msp` files
- `share/potentials/` — SNAP `.snapparam`/`.snapcoeff` and LJ potential files (installed from `contribs/microStamp/potentials/` via `EXANB_BUILD_MICROSTAMP`)

Example `.msp` files themselves are resolved at the Python level: `find_example()` walks `XNB_SHARE_DIR` searching by filename or relative path, then returns the absolute path passed to `pyonika.init()`.

### `_paths.py` and portability

CMake generates `_paths.py` at install time with absolute paths. If the installation is moved, regenerate `_paths.py` by re-running `cmake --install`, or simply source `setup-env.sh` before importing — when the env vars are already set, `_ensure_env()` is a no-op.

---

## Source files

```
exaNBody/
├── CMakeLists.txt                      ← option(EXANB_BUILD_PYTHON); add_subdirectory(python);
│                                          PYTHONPATH added to setup-env.sh template
├── contribs/microStamp/CMakeLists.txt  ← install(DIRECTORY potentials/ DESTINATION share/potentials)
└── python/
    ├── CMakeLists.txt                  ← configure_file(_paths.py.in); install package + examples
    ├── README.md                       ← this file
    ├── pyexanbody/
    │   ├── __init__.py                 ← _ensure_env(), _ensure_mpi_external(),
    │   │                                  full pyonika re-export, find/list/run_example(), run_file()
    │   └── _paths.py.in               ← CMake template: @onika_DIR@ and @CMAKE_INSTALL_PREFIX@
    └── exemples/
        ├── test.py                     ← smoke test + operator catalogue check
        ├── run_solar_system.py         ← run solar_system.msp via run_example()
        └── solar_system_graph.py       ← full Python reproduction of solar_system.msp
                                           using set_operator_defaults() + build_simulation_graph()
```
