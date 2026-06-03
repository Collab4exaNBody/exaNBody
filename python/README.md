# pyexanbody — Python interface to the exaNBody N-body simulation framework

`pyexanbody` is a Python package that wraps `pyonika` (the onika C++ bindings)
and makes the full exaNBody operator catalogue available from Python.  It
handles environment setup, MPI lifecycle, and exaNBody-specific data extraction,
and re-exports the entire `pyonika` API so user code only ever imports this one
module.

---

## Table of contents

1. [Source layout](#source-layout)
2. [Build](#build)
3. [Quick start](#quick-start) — Pattern A (run file), B (full Python graph), C (patch .msp values)
4. [What happens at import time](#what-happens-at-import-time)
5. [pyexanbody API reference](#pyexanbody-api-reference)
   - [Inherited pyonika API](#inherited-pyonika-api)
   - [Example discovery and execution](#example-discovery-and-execution)
   - [Data access helpers](#data-access-helpers)
   - [Constants](#constants)
6. [.msp → Python mapping](#msp--python-mapping)
7. [Running multiple simulations](#running-multiple-simulations)
8. [Memory model and lifetime](#memory-model-and-lifetime)
9. [Implementation — `pyexanbody/__init__.py`](#implementation--pyexanbodyinitpy)
   - [Step 1 — `_ensure_env()`](#step-1--_ensure_env)
   - [Step 2 — `import pyonika`](#step-2--import-pyonika)
   - [Step 3 — `_ensure_mpi_external()`](#step-3--_ensure_mpi_external)
   - [Step 4 — loading `_exanb_data`](#step-4--loading-_exanb_data)
   - [Step 5 — re-exporting pyonika](#step-5--re-exporting-pyonika)
10. [Implementation — `bind_exanb_data.cpp`](#implementation--bind_exanb_datacpp)
    - [Retrieving `register_slot_extractor` via PyCapsule](#retrieving-register_slot_extractor-via-pycapsule)
    - [Tier 1 — `SimulationStatistics` extractor](#tier-1--simulationstatistics-extractor)
    - [Tier 2 — `GridCellValues` extractor and `GridCellValuesView`](#tier-2--gridcellvalues-extractor-and-gridcellvaluesview)
    - [Module initialisation](#module-initialisation)
11. [CMake integration](#cmake-integration)

---

## Source layout

```
exaNBody/
├── python/
│   ├── CMakeLists.txt              ← _exanb_data target, _paths.py generation, installs
│   ├── bind_exanb_data.cpp         ← pybind11 extension: SimulationStatistics + GridCellValues
│   ├── README.md                   ← this file
│   ├── PYTHON_DATA_ACCESS.md       ← design document for data-access tiers
│   ├── pyexanbody/
│   │   ├── __init__.py             ← env setup, MPI init, re-export, helpers
│   │   └── _paths.py.in            ← CMake template → _paths.py (baked install paths)
│   └── exemples/
│       ├── pyexanbody_dryrun_test_import_pyexanbody.py   ← import smoke test
│       ├── pyexanbody_run_main_config.py                 ← Pattern A + full graph/slot inspection
│       ├── pyexanbody_run_example.py                     ← list / find / run installed examples
│       ├── pyexanbody_run_solar_system.py                ← Pattern A: run solar_system.msp
│       ├── pyexanbody_reproduce_solar_system_case.py     ← Pattern B: full Python graph, no .msp
│       ├── pyexanbody_patch_defaults.py                  ← Pattern C: patch .msp values before run
│       ├── pyexanbody_parameter_sweep.py                 ← multiple init/run/end cycles
│       ├── pyexanbody_read_sim_stats.py                  ← Tier 1 data access
│       ├── pyexanbody_read_cell_values.py                ← Tier 2 data access
│       └── pyexanbody_extract_particle_data.py           ← Tier 3a data access
```

---

## Build

`EXANB_BUILD_PYTHON` compiles `_exanb_data.so` (the pybind11 extension for
exaNBody-specific slot extractors) and installs the `pyexanbody` Python package.
`EXANB_BUILD_MICROSTAMP` installs potential files under `share/potentials/`,
required by the MD and SNAP examples.

```bash
cmake -B build-exanbody \
      -DEXANB_BUILD_PYTHON=ON \
      -DEXANB_BUILD_MICROSTAMP=ON \
      -Donika_DIR=/path/to/install-onika \
      -DCMAKE_INSTALL_PREFIX=/path/to/install-exanbody \
      exaNBody/
cmake --build  build-exanbody
cmake --install build-exanbody
```

No extra C++ files need to be written — the only compiled output is
`_exanb_data.<platform>.so`.  The `pyexanbody` package itself is pure Python.

**Install layout:**

```
install-exanbody/
├── bin/
│   └── setup-env.sh              ← sets PYTHONPATH, ONIKA_PLUGIN_PATH, …
├── lib/
│   ├── libexanbCore.so           ← needed at dlopen time by plugin .so files
│   └── pyexanbody/
│       ├── __init__.py
│       ├── _paths.py             ← generated: ONIKA_INSTALL_DIR, XNB_INSTALL_DIR
│       └── _exanb_data.<plat>.so
├── plugins/                      ← 14 exaNBody plugin .so files
├── data/config/
│   └── main-config.msp           ← exaNBody default config
├── share/
│   ├── examples/                 ← installed .msp example files
│   ├── potentials/               ← SNAP / LJ potential files (EXANB_BUILD_MICROSTAMP)
│   └── …
└── python/
    └── exemples/                 ← installed example scripts
```

After install, source the environment script before running any Python script:

```bash
source /path/to/install-exanbody/bin/setup-env.sh
python3 my_script.py
```

---

## Quick start

```python
import pyexanbody as xnb

# Pattern A — run an installed example by name (simplest)
xnb.run_example("solar_system.msp")

# Pattern A — run any .msp file by path
xnb.run_file("my_sim.msp")
```

> **Examples:** `exemples/pyexanbody_dryrun_test_import_pyexanbody.py` (smoke test),
> `exemples/pyexanbody_run_main_config.py` (Pattern A + full graph/slot inspection),
> `exemples/pyexanbody_run_solar_system.py` (run solar_system.msp by name or path).

```python
import pyexanbody as xnb, sys, os

main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")

# Pattern B — build the simulation graph entirely from Python, without any .msp file
# beyond main-config.msp for bootstrapping.

# Step 1: bootstrap MPI, plugins and default operator definitions.
ctx = xnb.init([sys.argv[0], main_config])
if ctx.error_code >= 0:
    sys.exit(ctx.error_code)

# Step 2: define the simulation from Python (equivalent to top-level .msp keys).
xnb.set_operator_defaults({
    "global": {
        "dt": "1 s",  "rcut_max": "280 m",
        "simulation_end_iteration": 10,
    },
    "input_data": [
        {"domain":  {"cell_size": "150 m", "grid_dims": [13, 13, 13],
                     "bounds": [[0,0,0], ["1950 m","1950 m","1950 m"]],
                     "periodic": [True,True,True]}},
        "init_rcb_grid",
        {"lattice":  {"structure": "BCC", "types": ["A"],
                      "size": ["150 m","150 m","150 m"]}},
    ],
    "compute_force": [
        {"gravitational_force": {"config": {"G": "6.67e-11 m^3/kg/s^2"}, "rcut": "280 m"}}
    ],
    "dump_data": "nop",
})

# Step 3: build and run.
graph = xnb.build_simulation_graph(ctx, ["default_simulation"])
xnb.run_node(ctx, graph)
xnb.end(ctx)
```

> **Example:** `exemples/pyexanbody_reproduce_solar_system_case.py` — full Python
> reproduction of `solar_system.msp` without reading the `.msp` file.

```python
# Pattern C — load main-config.msp but patch a few values before running.
#
# init() builds the simulation graph internally from the .msp file but does NOT
# run it.  set_operator_defaults() merges new values into the factory defaults
# (only the supplied keys are overridden).  You must then call
# build_simulation_graph() to produce a new graph using those updated defaults,
# and run it with run_node().
#
# IMPORTANT: run(ctx) would execute the graph built during init(), before
# set_operator_defaults() was called — your changes would have no effect.
# Always use run_node(ctx, graph) after build_simulation_graph().

ctx = xnb.init([sys.argv[0], main_config])
if ctx.error_code >= 0:
    sys.exit(ctx.error_code)

# Patch only the keys you want to change — the rest come from main-config.msp.
xnb.set_operator_defaults({
    "global": {"simulation_end_iteration": 5},
})

# build_simulation_graph(ctx) with no list reuses the simulation: node stored
# by init() — same structure, updated parameter values.
graph = xnb.build_simulation_graph(ctx)
xnb.run_node(ctx, graph)
xnb.end(ctx)
```

> **Example:** `exemples/pyexanbody_patch_defaults.py`.

---

## What happens at import time

```python
import pyexanbody as xnb
```

Four steps execute automatically, in order, before any user code runs:

| Step | Function | What it does |
|---|---|---|
| 1 | `_ensure_env()` | Sets `ONIKA_PLUGIN_PATH`, `ONIKA_CONFIG_PATH`, `LD_LIBRARY_PATH`, `ONIKA_DATA_PATH`, `sys.path` from the baked-in install paths |
| 2 | `import pyonika` | Loads the pybind11 C++ extension; `libonika.so` and `libmpi.so` enter the process |
| 3 | `_ensure_mpi_external()` | Pre-initialises MPI via ctypes so `end()` never calls `MPI_Finalize()` — enabling multiple `init`/`run`/`end` cycles |
| 4 | `from pyexanbody import _exanb_data` | Imports the exaNBody-specific slot extractor extension, registering `SimulationStatistics` and `GridCellValues` in pyonika's `slot_as_array` registry |

After import, `xnb.init()` loads all 14 exaNBody plugin `.so` files and
registers their operators (domain management, AMR grid, particle neighbours,
MD potentials, gravity, I/O, …) alongside the onika built-ins, giving a
catalogue of ~213 operators.

---

## pyexanbody API reference

### Inherited pyonika API

`pyexanbody` re-exports the complete `pyonika` public API.  See
`onika/python/README.md` for the full reference.  Key entry points:

| Symbol | Description |
|---|---|
| `init(argv)` → `ApplicationContext` | Bootstrap onika + exaNBody from an `.msp` file |
| `run(ctx)` | Execute the graph built by `init()` |
| `run_node(ctx, node)` | Execute a Python-built graph |
| `end(ctx)` | Finalise (MPI never finalised — safe to call `init` again) |
| `make_operator(name, config={})` | Instantiate a registered operator by name |
| `available_operators()` | Sorted list of all registered operators |
| `set_operator_defaults(dict)` | Merge named operator defaults |
| `get_operator_defaults()` | Current operator defaults as a dict |
| `build_simulation_graph(ctx, list)` | Build a simulation graph from a Python list |
| `slot_as_array(slot)` | Zero-copy numpy view of a slot value |
| `ApplicationContext` | Simulation state: MPI, GPU/CPU counts, graph, profiling |
| `OperatorNode` | Graph node: slots, traversal, numpy access |
| `OperatorSlotBase` | One input/output slot: type, value, direction |
| `OnikaError` | Unknown operator or slot type incompatibility |

---

### Example discovery and execution

#### `xnb.list_examples() → list[str]`

Return a sorted list of all installed `.msp` example files.  Each entry is a
path relative to its share subdirectory root, suitable for passing directly to
`find_example()` or `run_example()`.

```python
xnb.list_examples()
# ['benchmark_lj_snap/input_lj_Ni.msp',
#  'deterministic/deterministic_lj.msp',
#  'simple_example/solar_system.msp', ...]
```

#### `xnb.find_example(name: str) → str`

Resolve an example name to its absolute path without running it.  `name` can
be a bare filename (`"solar_system.msp"`) or a relative path within a share
subdirectory (`"deterministic/deterministic_lj.msp"`).  The search walks all
first-level subdirectories of `XNB_SHARE_DIR`, then one level deeper for bare
filenames.

```python
path = xnb.find_example("solar_system.msp")
# → '.../install-exanbody/share/examples/simple_example/solar_system.msp'
```

Raises `FileNotFoundError` if no match is found, including the list of
available examples in the message.

#### `xnb.run_example(name: str, argv0="pyexanbody", **extra_args) → int`

Resolve an installed example by name and run it end-to-end.  Delegates to
`run_file(find_example(name), …)`.

```python
xnb.run_example("solar_system.msp")
```

Returns 0 on normal completion or the early-exit code from onika.

#### `xnb.run_file(msp_path: str, argv0="pyexanbody", **extra_args) → int`

Run a complete simulation from an `.msp` file by path.  Internally does:
`init → (early-exit check) → run → end`.

```python
xnb.run_file("my_sim.msp")
```

> **Note on `**extra_args`:** the underlying onika command-line parser expects
> space-separated `--key value` pairs (not `--key=value`) and only reaches
> `configuration:` YAML keys (not simulation parameters like `global`).  Use
> `set_operator_defaults()` after `init()` to override simulation parameters.

> **Examples:** `exemples/pyexanbody_run_example.py` (list / find / run any installed example),
> `exemples/pyexanbody_run_solar_system.py` (run solar_system.msp by name or path).

---

### Data access helpers

#### `xnb.read_sim_stats(graph, _debug=False) → dict`

Return global simulation statistics from the most recent execution as a
Python dict.  Walks the graph depth-first looking for an operator named
`simulation_stats` and returns the dict produced by the `SimulationStatistics`
extractor (Tier 1).

```python
stats = xnb.read_sim_stats(graph)
# {
#   "kinetic_energy":  float,
#   "particle_count":  int,
#   "min_vel":         float,
#   "max_vel":         float,
#   "min_acc":         float,
#   "max_acc":         float,
# }
print(f"N  = {stats['particle_count']}")
print(f"KE = {stats['kinetic_energy']:.4g}")
```

Returns an empty dict if no `simulation_stats` node is found or if
`_exanb_data` failed to load.  Pass `_debug=True` to print diagnostic
information about which nodes were visited and what `slot_as_array` returned.

`simulation_stats` is always present in `default_simulation`; it runs inside
`first_iteration` and `print_log_if_triggered`.  The returned dict is a copy
by value — safe to use after `end()`.

> **Example:** `exemples/pyexanbody_read_sim_stats.py`.

---

#### `xnb.read_cell_values(graph, field_name: str) → numpy.ndarray | None`

Return a zero-copy numpy array for a named grid cell field, with ghost cells
stripped.  Walks the graph looking for the first operator with a
`grid_cell_values` slot that contains `field_name`, and returns
`GridCellValuesView.field_inner(field_name)`.

```python
density = xnb.read_cell_values(graph, "density")
# subdiv=1 → shape (inx, iny, inz),  dtype float64
# subdiv=S → shape (inx, iny, inz, S, S, S)
```

Returns `None` if no matching node or field is found.  The returned array is a
zero-copy view — keep the `ApplicationContext` alive while it is in use.

For full control, access the `GridCellValuesView` directly via
`node.slot_as_array("grid_cell_values")`:

```python
gcv = node.slot_as_array("grid_cell_values")
print(gcv.field_names)              # list of registered field names
print(gcv.shape, gcv.ghost_layers)  # stored dims (incl. ghosts), ghost count

raw   = gcv.field("density")        # (n_cells,) — all cells + ghost border
inner = gcv.field_inner("density")  # (inx, iny, inz) — domain only, zero-copy
```

> **Example:** `exemples/pyexanbody_read_cell_values.py`.

---

#### Per-particle fields — `extract_particle_field` operator

Per-particle data (positions, velocities, forces, …) is extracted by adding
one `extract_particle_field` operator per field to the simulation graph.  Each
operator copies its field from the cell-of-SoA grid into a contiguous
`std::vector<double>` output slot, readable as a zero-copy numpy array via
`slot_as_array("data")`.

```python
import numpy as np

xnb.set_operator_defaults({
    "final_dump": [
        {"extract_particle_field": {"field_name": f}}
        for f in ["rx", "ry", "rz", "vx", "vy", "vz"]
    ],
    "dump_data": "nop",
})
graph = xnb.build_simulation_graph(ctx, ["default_simulation"])
xnb.run_node(ctx, graph)

particle_data = {}
def collect(node):
    if node.name().startswith("extract_particle_field"):
        fname = node.slot_values().get("field_name", "").strip("'\"")
        arr   = node.slot_as_array("data")
        if arr is not None and fname:
            particle_data[fname] = np.array(arr, copy=True)
graph.apply_graph(collect)

pos = np.stack([particle_data[f] for f in ["rx","ry","rz"]], axis=1)  # (N, 3)
```

> **Example:** `exemples/pyexanbody_extract_particle_data.py`.

---

### Constants

| Name | Value | Description |
|---|---|---|
| `XNB_SHARE_DIR` | `<XNB_INSTALL_DIR>/share` | Root of the installed share tree |
| `XNB_INSTALL_DIR` | (from `_paths.py`) | exaNBody install prefix |
| `ONIKA_INSTALL_DIR` | (from `_paths.py`) | onika install prefix |

---

## .msp → Python mapping

The mapping between `.msp` top-level keys and Python calls is the same as for
pyonika (see `onika/python/README.md — .msp → Python mapping`).  The
exaNBody-specific keys follow the same pattern:

| `.msp` top-level key | Python equivalent |
|---|---|
| `global: { dt: 1 s, … }` | `set_operator_defaults({"global": {"dt": "1 s", …}})` |
| `input_data: [ … ]` | `set_operator_defaults({"input_data": [ … ]})` |
| `compute_force: [ … ]` | `set_operator_defaults({"compute_force": [ … ]})` |
| `dump_data: dump_data_paraview` | `set_operator_defaults({"dump_data": "dump_data_paraview"})` |
| `simulation: default_simulation` | `build_simulation_graph(ctx, ["default_simulation"])` |
| `run exaNBody my_sim.msp` | `xnb.run_file("my_sim.msp")` |

> **Example:** `exemples/pyexanbody_reproduce_solar_system_case.py` — full Python
> reproduction of `solar_system.msp` using `set_operator_defaults()` +
> `build_simulation_graph()`, without reading the `.msp` file.

---

## Running multiple simulations

Because `pyexanbody` pre-initialises MPI at import time (Step 3 of
[What happens at import time](#what-happens-at-import-time)), `end()` never
calls `MPI_Finalize()`.  Multiple `init`/`run`/`end` cycles are therefore safe
in a single Python session.  Each `init()` reloads the `.msp` defaults;
`set_operator_defaults()` applies per-run overrides on top.

```python
import pyexanbody as xnb, sys, os

main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")

for niter in [2, 5, 10]:
    ctx = xnb.init([sys.argv[0], main_config])
    xnb.set_operator_defaults({
        "global": {"simulation_end_iteration": niter},
    })
    graph = xnb.build_simulation_graph(ctx)
    xnb.run_node(ctx, graph)
    stats = xnb.read_sim_stats(graph)
    print(f"niter={niter:3d}  KE={stats.get('kinetic_energy', float('nan')):.6g}")
    xnb.end(ctx)
```

> **Example:** `exemples/pyexanbody_parameter_sweep.py`.

---

## Memory model and lifetime

| API | Memory model | Safe after `xnb.end(ctx)`? |
|---|---|---|
| `read_sim_stats(graph)` | copy — `py::dict` by value | **yes** |
| `read_cell_values(graph, field)` | zero-copy — pointer into `CudaMMVector<double>` | **no** |
| `gcv.field(name)` | zero-copy — same pointer | **no** |
| `gcv.field_inner(name)` | zero-copy — same pointer, offset into inner cells | **no** |
| `node.slot_as_array("data")` (Tier 3a) | zero-copy of `std::vector<double>` | **no** |

For zero-copy arrays, copy before calling `end()`:

```python
density = xnb.read_cell_values(graph, "density")
owned   = np.array(density, copy=True)   # owns its memory
xnb.end(ctx)
print(owned.mean())   # ✓ safe
```

`GCVView` holds a raw `const GridCellValues*` pointer.  The pointed-to object
is owned by an `OperatorSlot` inside the simulation graph, which is in turn
owned by the `ApplicationContext`.  As long as `ctx` is alive (and `end()` has
not been called), the pointer is valid.

> **Example:** `exemples/pyexanbody_read_cell_values.py` (zero-copy access and
> safe-copy pattern demonstrated side by side).

---

## Implementation — `pyexanbody/__init__.py`

The `__init__.py` runs five sequential steps at import time.

### Step 1 — `_ensure_env()`

Sets up the process environment from the paths baked into `_paths.py` by CMake.
All modifications are **idempotent**: variables are only extended or set when
the required entry is not already present, so sourcing `setup-env.sh` before
importing is a no-op.

| Variable | Action |
|---|---|
| `sys.path` | Prepend `<onika_lib>` so `import pyonika` works |
| `LD_LIBRARY_PATH` | Prepend `<onika_lib>` and `<xnb_lib>` so `dlopen` finds `libexanbCore.so` when onika's plugin loader calls it during `init()` |
| `ONIKA_CONFIG_PATH` | Set to `<xnb_install>/data/config` if not already set (single directory, not colon-separated) |
| `ONIKA_PLUGIN_PATH` | Append onika and exaNBody plugin directories if not present (colon-separated) |
| `ONIKA_DATA_PATH` | Append all first-level subdirectories of `XNB_SHARE_DIR` (examples, potentials, …) — colon-separated, allows operators to find data files by basename |

`ONIKA_CONFIG_PATH` accepts only a single directory (not a list), while
`ONIKA_DATA_PATH` is a colon-separated search path used by operators that open
data files (potentials, dump masks, etc.).  All first-level subdirectories of
`share/` are appended so that `share/potentials/` (SNAP and LJ files installed
by `EXANB_BUILD_MICROSTAMP`) is automatically on the path.

### Step 2 — `import pyonika`

Loads the pybind11 C++ extension.  Because `sys.path` now includes the onika
lib directory (from Step 1), the import succeeds without `setup-env.sh` having
been sourced externally.  Importing `pyonika` also loads `libonika.so` and its
transitive dependencies (`libmpi.so`, CUDA runtime if available) into the
process.

### Step 3 — `_ensure_mpi_external()`

Pre-initialises MPI so that onika treats it as externally owned, enabling
multiple `init`/`run`/`end` cycles in one Python session.

**Why this is needed:** `onika::app::initialize_mpi()` calls `MPI_Initialized()`
at startup.  If MPI is already up, it sets `external_mpi_init = true` and
`onika::app::end()` skips `MPI_Finalize()`.  If onika itself initialises MPI,
`end()` finalises it — making any subsequent `init()` call fatal (the MPI
standard forbids all MPI calls after `MPI_Finalize()`).

**Implementation:** After `import pyonika` has loaded `libmpi.so` as a
transitive dependency, `_ensure_mpi_external()` finds the library in
`/proc/self/maps` (Linux), loads it with `ctypes.CDLL`, and calls
`MPI_Init_thread(..., MPI_THREAD_MULTIPLE=3, &provided)` via ctypes if
`MPI_Initialized()` reports false.  `MPI_Init_thread` (not plain `MPI_Init`)
is used deliberately: onika's `initialize_mpi()` would itself call
`MPI_Init_thread(..., MPI_THREAD_MULTIPLE, ...)` if it owned MPI init, and
when it takes the "external MPI" branch it calls `MPI_Query_thread()` to check
the thread level — using plain `MPI_Init` would leave the level at
`MPI_THREAD_SINGLE` and trigger the warning *"no MPI_THREAD_MULTIPLE support"*.

Because `_ensure_mpi_external()` owns the `MPI_Init_thread` call, it also
registers a `MPI_Finalize` `atexit` handler.  This is necessary when running
under `mpirun`/`srun`: onika's `end()` always skips `MPI_Finalize` when it
sees `external_mpi_init=true`, so without this handler the process would exit
without finalising MPI and mpirun would report an improper termination.  The
`atexit` handler runs even on `sys.exit()`, and guards against double-finalise
with a `MPI_Finalized()` check.

A fallback searches by standard library name (`"mpi"`, `"openmpi"`, `"mpich"`,
etc.) for portability.  The entire function is a no-op after the first call or
if MPI is already up.

### Step 4 — loading `_exanb_data`

```python
from pyexanbody import _exanb_data
```

`_exanb_data` is a pybind11 extension module that registers exaNBody-specific
slot extractors into pyonika's `slot_as_array` registry.  Importing it is
sufficient — it runs its registration code as a side effect.

If loading fails (e.g. `_exanb_data.so` was not built), a `UserWarning` is
issued and pyexanbody continues without data access support.
`read_sim_stats()` and `read_cell_values()` will return empty results.

See [Implementation — `bind_exanb_data.cpp`](#implementation--bind_exanb_datacpp)
for what this extension registers.

### Step 5 — re-exporting pyonika

```python
from pyonika import (init, run, run_node, end, make_operator, available_operators,
                     set_operator_defaults, build_simulation_graph, slot_as_array,
                     ApplicationContext, OperatorNode, OperatorSlotBase, OnikaError)
```

All public pyonika names are imported into the `pyexanbody` namespace.  User
code needs only `import pyexanbody as xnb`.

---

## Implementation — `bind_exanb_data.cpp`

`_exanb_data` is a pybind11 module whose sole job is to call
`register_slot_extractor()` for two exaNBody C++ types.  It does no independent
operator binding — it only extends pyonika's existing `slot_as_array` dispatch
table.

### Retrieving `register_slot_extractor` via PyCapsule

`register_slot_extractor` lives in `pyonika.so`, but pybind11 extensions are
loaded by Python's import machinery with `RTLD_LOCAL` by default.  `RTLD_LOCAL`
hides a shared library's symbols from other `dlopen`'d libraries, so
`_exanb_data.so` cannot call `register_slot_extractor` via a normal function
call or a `dlsym` lookup — the symbol is not exported into the global namespace.

The solution (set up in `bind_soatl.cpp`) is to expose the raw function pointer
as a `PyCapsule` attribute on the `pyonika` module object:

```cpp
// in bind_soatl.cpp (pyonika side)
m.attr("_register_slot_extractor_fn") = py::capsule(
    reinterpret_cast<void*>(&register_slot_extractor),
    "_register_slot_extractor_fn");
```

`bind_exanb_data.cpp` retrieves it at module load time via the Python C API:

```cpp
static RegisterFn get_register_fn()
{
    py::module_ pyonika = py::module_::import("pyonika");
    py::capsule cap = pyonika.attr("_register_slot_extractor_fn").cast<py::capsule>();
    return reinterpret_cast<RegisterFn>(cap.get_pointer());
}
```

This is entirely `RTLD_LOCAL`-safe: both the capsule lookup and the function
call go through the Python object model, not the dynamic linker.

### Tier 1 — `SimulationStatistics` extractor

`SimulationStatistics` is a plain struct with 6 fields:

```cpp
struct SimulationStatistics {
    double             m_kinetic_energy = 0.0;
    double             m_min_vel, m_max_vel;
    double             m_min_acc, m_max_acc;
    unsigned long long m_particle_count = 0;
};
```

The extractor is a lambda registered under `typeid(SimulationStatistics)`:

```cpp
reg(std::type_index(typeid(SS)), [](OSB& slot) -> py::object {
    auto* typed = static_cast<onika::scg::OperatorSlot<SS>*>(&slot);
    if (!typed->has_value()) return py::none();
    const SS& s = **typed;
    py::dict d;
    d["kinetic_energy"]  = s.m_kinetic_energy;
    d["particle_count"]  = static_cast<unsigned long long>(s.m_particle_count);
    d["min_vel"]  = s.m_min_vel;   d["max_vel"]  = s.m_max_vel;
    d["min_acc"]  = s.m_min_acc;   d["max_acc"]  = s.m_max_acc;
    return d;
});
```

Key points:
- The `static_cast` is safe because the type string already matched in `slot_as_array`.
- The return value is a fresh `py::dict` — a **copy** of the C++ struct
  fields.  The dict is safe to use after `end()`.
- `m_particle_count` is explicitly cast to `unsigned long long` to ensure
  Python receives an `int`, not a floating-point value.

### Tier 2 — `GridCellValues` extractor and `GridCellValuesView`

`GridCellValuesT<double>` (typedef'd as `GridCellValues`) stores a flat
`CudaMMVector<double>` buffer in **cell-major AoS layout**:

```
buffer = [cell0: f0_s000 f0_s001 … f0_sSSS f1_s000 … | cell1: … | …]
```

Each cell occupies `total_comps` doubles, where `total_comps` is the sum of
`m_components` across all registered fields.  For field `f`:

- `f.m_offset` — byte offset (in doubles) from the start of a cell's block
- `f.m_subdiv` — sub-cell subdivision factor S (1 = no subdivision)
- `f.m_components` = S³ × `ncomps_per_subcell`

#### `GCVView` — the Python wrapper struct

```cpp
struct GCVView { const exanb::GridCellValues* ptr; };
```

A minimal struct holding a raw (non-owning) pointer to the `GridCellValues`
object inside the operator slot.  The caller must keep the `ApplicationContext`
alive while any numpy array derived from `GCVView` is in use.

The extractor lambda returns a `py::cast(GCVView{&gcv})`:

```cpp
reg(std::type_index(typeid(GCV)), [](OSB& slot) -> py::object {
    auto* typed = static_cast<onika::scg::OperatorSlot<GCV>*>(&slot);
    if (!typed->has_value()) return py::none();
    const GCV& gcv = **typed;
    if (gcv.empty()) return py::none();
    return py::cast(GCVView{&gcv});
});
```

#### `GridCellValuesView` Python class

`GCVView` is exposed to Python as `GridCellValuesView` with the following
members:

| Attribute / method | Description |
|---|---|
| `field_names` | List of all registered field names (from `gcv.fields()`) |
| `shape` | `(nx, ny, nz)` — stored grid dims **including ghost layers** |
| `n_cells` | Total cell count including ghost layers |
| `ghost_layers` | Number of ghost layers on each side |
| `has_field(name)` | Field existence check |
| `field(name)` | Zero-copy view of ALL cells including ghost layers |
| `field_inner(name)` | Zero-copy ghost-stripped 3-D (or up to 7-D) view |

#### `field(name)` — raw view with ghost cells

`field()` builds a `py::buffer_info` starting at
`data().data() + f.m_offset` — the first component of field `f` in cell 0.

Shape and strides depend on `S = f.m_subdiv` and `ncomps = f.m_components / S³`:

| S | ncomps | ndim | shape | strides |
|---|---|---|---|---|
| 1 | 1 | 1 | `(n_cells,)` | `(tot·8,)` |
| 1 | >1 | 2 | `(n_cells, ncomps)` | `(tot·8, 8)` |
| >1 | 1 | 4 | `(n_cells, S, S, S)` | `(tot·8, S²ncomps·8, S·ncomps·8, ncomps·8)` |
| >1 | >1 | 5 | `(n_cells, S, S, S, ncomps)` | `(tot·8, S²ncomps·8, S·ncomps·8, ncomps·8, 8)` |

#### `field_inner(name)` — ghost-stripped view

`field_inner()` skips the ghost border using a **pointer offset** and bridges
over the ghost cells using **strides that span the full stored row**.  No data
is copied.

The base pointer is offset to the first inner cell:

```cpp
size_t inner_offset = (gl*ny*nz + gl*nz + gl) * tot + f.m_offset;
const double* base  = v.ptr->data().data() + inner_offset;
```

where `gl` is the number of ghost layers and `(gl, gl, gl)` is the index of
the first inner cell in (i, j, k) order.

The strides along the three cell axes (`si`, `sj`, `sk`) are computed from the
**full stored dimensions** `(nx, ny, nz)`, not the inner dimensions
`(inx, iny, inz)`:

```
si = ny · nz · tot · 8    (stepping in i skips a full ny*nz slice of the stored grid)
sj =      nz · tot · 8
sk =           tot · 8
```

Because `si`, `sj`, `sk` use the full stored dimensions, advancing from one
inner cell to the next automatically steps over the ghost cells in the stored
buffer.  numpy reads only the `(inx, iny, inz)` cells that the shape declares;
the ghost cells are never touched.

The sub-cell strides (`s_ci`, `s_cj`, `s_ck`) are field-local and do not span
ghost cells:

```
s_ci = S² · ncomps · 8
s_cj = S  · ncomps · 8
s_ck =      ncomps · 8
```

Full shape / stride table for `field_inner`:

| S | ncomps | ndim | shape | strides |
|---|---|---|---|---|
| 1 | 1 | 3 | `(inx, iny, inz)` | `(si, sj, sk)` |
| 1 | >1 | 4 | `(inx, iny, inz, ncomps)` | `(si, sj, sk, 8)` |
| >1 | 1 | 6 | `(inx, iny, inz, S, S, S)` | `(si, sj, sk, s_ci, s_cj, s_ck)` |
| >1 | >1 | 7 | `(inx, iny, inz, S, S, S, ncomps)` | `(si, sj, sk, s_ci, s_cj, s_ck, 8)` |

### Module initialisation

```cpp
PYBIND11_MODULE(_exanb_data, m)
{
    py::class_<GCVView>(m, "GridCellValuesView")
        .def_property_readonly("field_names", …)
        // … all GCVView members …
        .def("field_inner", …);

    RegisterFn reg = get_register_fn();
    register_simulation_statistics(reg);
    register_grid_cell_values(reg);
}
```

The two `register_*` calls happen at module load time — as soon as
`from pyexanbody import _exanb_data` executes in `__init__.py`.  From that
point on, `slot_as_array()` in pyonika can handle `SimulationStatistics` and
`GridCellValues` slots correctly.

---

## CMake integration

`python/CMakeLists.txt` handles four responsibilities:

**1. Build `_exanb_data.so`**

```cmake
pybind11_add_module(_exanb_data bind_exanb_data.cpp)
target_link_libraries(_exanb_data PRIVATE exanbCore exanbGridCellParticles exanbIO)
target_compile_definitions(_exanb_data PRIVATE
    ONIKA_MATH_EXPORT_NAMESPACE=exanb
    ONIKA_SCG_EXPORT_NAMESPACE=exanb
    ONIKA_LOG_EXPORT_NAMESPACE=exanb)
```

The three `ONIKA_*_EXPORT_NAMESPACE=exanb` definitions propagate what
`exanbody-config.cmake` sets via `exanbCore`'s `INTERFACE` properties.  They
make onika math types (`Vec3d`, `IJK`, `AABB`, …) available unqualified inside
namespace `exanb`, which the exaNBody headers expect.

**2. Undefined-symbol link options**

```cmake
target_link_options(_exanb_data PRIVATE
    $<$<PLATFORM_ID:Linux>:-Wl,--allow-shlib-undefined>
    $<$<PLATFORM_ID:Darwin>:-Wl,-undefined,dynamic_lookup>)
```

`register_slot_extractor` lives in `pyonika.so` but `_exanb_data.so` does not
link against it at build time (to avoid a runtime link-time dependency).  The
symbol is resolved at import time via the PyCapsule mechanism instead.
`--allow-shlib-undefined` suppresses the linker error that would otherwise
occur on Linux.

**3. Generate `_paths.py`**

```cmake
configure_file(pyexanbody/_paths.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/pyexanbody/_paths.py @ONLY)
```

`_paths.py.in` contains two CMake variables:

```python
ONIKA_INSTALL_DIR = "@onika_DIR@"
XNB_INSTALL_DIR   = "@CMAKE_INSTALL_PREFIX@"
```

`configure_file` substitutes them with the actual install paths at build time,
producing `_paths.py` with literal strings.  This makes `import pyexanbody`
work without `setup-env.sh` as long as the install directories have not moved.

**4. Install**

- `_exanb_data.<platform>.so` → `lib/pyexanbody/`
- `pyexanbody/__init__.py` + generated `_paths.py` → `lib/pyexanbody/`
- `README.md` → `python/`
- All example scripts → `python/exemples/`
