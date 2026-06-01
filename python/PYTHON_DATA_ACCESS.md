# Python data access — design document

This document describes the plan for exposing simulation data from exaNBody
directly into Python objects (numpy arrays, dicts).  Three categories of data
are targeted:

1. **Per-particle fields** — positions, velocities, forces, IDs, …
2. **Per-cell / grid-cell fields** — `GridCellValues` named scalar/vector fields
3. **Global thermodynamic variables** — kinetic energy, particle count, …

---

## Background: the `slot_as_array` mechanism

`pyonika`'s `bind_soatl.cpp` maintains a static registry of
`type_index → extractor` functions.  The module-level `slot_as_array(slot)`
function and `OperatorNode.slot_as_array(name)` iterate this registry and
return a zero-copy numpy view when a matching extractor is found.

Currently registered types (onika side):

| C++ type | numpy shape |
|---|---|
| `std::vector<T>` (scalar arithmetic T) | `(N,)` |
| `std::vector<std::array<T,3>>` | `(N,3)` |
| `std::vector<std::array<T,4>>` | `(N,4)` |

All three target data categories involve types **not** in this registry
(`Grid`, `GridCellValuesT<double>`, `SimulationStatistics`).

---

## Tier 1 — Global thermodynamic variables (`SimulationStatistics`)

### Current state

`SimulationStatistics` is a plain 6-field struct computed by the
`simulation_stats` operator:

```cpp
struct SimulationStatistics {
    double             m_kinetic_energy = 0.0;
    double             m_min_vel, m_max_vel;
    double             m_min_acc, m_max_acc;
    unsigned long long m_particle_count = 0;
};
```

It is stored in an output slot but not registered in `slot_as_array`, so
`slot_as_array` returns `None` for it today.

### Plan

**Step 1 (onika)** — expose a registration API in `bind_soatl`:

```cpp
// onika/python/bind_soatl.h — new public function
using SlotExtractorFn =
    std::function<pybind11::object(onika::scg::OperatorSlotBase&)>;
void register_slot_extractor(std::type_index ti, SlotExtractorFn fn);
```

**Step 2 (exaNBody)** — add `python/bind_exanb_data.cpp`, compiled as part of
the Python build.  At plugin load time it calls `register_slot_extractor` for
`SimulationStatistics` and returns a Python `dict`:

```python
stats = xnb.slot_as_array(ctx.node("simulation_stats").out_slot("simulation_stats"))
# → {"kinetic_energy": 1.23e9, "particle_count": 500,
#    "min_vel": 0.0, "max_vel": 42.3, "min_acc": …, "max_acc": …}
```

**Python helper (pyexanbody)**:

```python
xnb.read_sim_stats(ctx)  # runs simulation_stats node then returns the dict
```

---

## Tier 2 — Per-cell data (`GridCellValuesT<double>`)

### Current state

`GridCellValuesT<double>` holds a flat `CudaMMVector<double>` buffer and a
`std::unordered_map<std::string, GridCellField>` name→(offset, stride, subdiv)
map.  It is not registered in `slot_as_array`.

### Plan

Register an extractor (same `bind_exanb_data.cpp`) that returns a lightweight
Python wrapper object exposing:

```python
gcv = xnb.slot_as_array(node.out_slot("grid_cell_values"))
gcv.field_names        # list[str]
gcv.shape              # (nx, ny, nz)
gcv.field("density")   # zero-copy numpy array, shape (nx*ny*nz, components)
```

The raw pointer comes directly from `m_data.data()` and the stride/offset from
`GridCellField` metadata — no copy is needed.

**Python helper**:

```python
xnb.read_cell_values(ctx, "density")   # → numpy array
```

---

## Tier 3 — Per-particle fields (Grid)

The `Grid<FieldSet<...>>` stores particles in a **cell-of-SoA** layout: each
cell holds a separate allocation for each field.  Data for a given field (e.g.
all `rx` values) is therefore non-contiguous across cells.

Two implementation options are described below.

---

### Tier 3a — Copy-on-read via `extract_particle_field` operator (IMPLEMENTED)

A new exaNBody operator copies a named field from all non-ghost cells into a
contiguous `std::vector<double>` output slot.  Because `std::vector<double>` is
already registered in `slot_as_array`, no new C++ bindings are needed.

**Operator**: `extract_particle_field`

| Slot | Direction | Type | Default | Description |
|---|---|---|---|---|
| `grid` | IN | `GridT` (REQUIRED) | — | Particle grid |
| `field_name` | IN | `std::string` | `"rx"` | Field to extract (`rx`, `ry`, `rz`, `vx`, `vy`, `vz`, `ax`, `ay`, `az`, `id`, `type`, …) |
| `include_ghosts` | IN | `bool` | `false` | Include ghost-layer particles |
| `data` | OUT | `std::vector<double>` | — | Flattened per-particle values, cast to `double` |

Particle order follows the cell-major iteration order of the grid (k → j → i,
inner cells first).

**Known limitation**: dynamic/generic fields (`generic_real<N>`,
`generic_vec3<N>`) whose names are set at runtime (e.g. `"density"`) are not
matched by compile-time `short_name()`.  Support for them requires either a
separate lookup mechanism or Tier 3b.

**Usage from Python**:

Include one `extract_particle_field` per field in the `dump_data` sequence via
`set_operator_defaults`.  All operators in the same graph share the `grid` slot
automatically; each has its own independent `data` output slot.

```python
import os, sys
import numpy as np
import pyexanbody as xnb

main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")
ctx = xnb.init([sys.argv[0], main_config])

xnb.set_operator_defaults({
    "global": {"simulation_end_iteration": 5, "simulation_dump_frequency": 1, ...},
    "input_data": [...],
    "compute_force": [...],
    # Replace Paraview with per-field extractors
    "dump_data": [
        {"extract_particle_field": {"field_name": "rx"}},
        {"extract_particle_field": {"field_name": "ry"}},
        {"extract_particle_field": {"field_name": "rz"}},
    ],
})

graph = xnb.build_simulation_graph(ctx)
xnb.run_node(ctx, graph)

# Collect extractor nodes and read their output slots
particle_data = {}
def collect(node):
    if node.name() != "extract_particle_field":
        return
    fname = node.slot_values().get("field_name", "").strip('"\'')
    arr = node.slot_as_array("data")   # zero-copy numpy view
    if arr is not None and fname:
        particle_data[fname] = np.array(arr, copy=True)  # own the data

graph.apply_graph(collect)

pos = np.stack([particle_data["rx"], particle_data["ry"], particle_data["rz"]], axis=1)
# pos.shape == (N_particles, 3)
```

**Complete working example**: `python/exemples/pyexanbody_extract_particle_data.py`

**Implementation**: `src/compute/extract_particle_field.cpp`

---

### Tier 3b — Zero-copy non-contiguous view (FUTURE WORK)

For performance-critical workflows where copying is unacceptable, a C++
extractor registered via the `register_slot_extractor` API (Tier 1 / Step 1)
could expose the Grid's cell-of-SoA layout directly as a Python object.

The object would provide:

```python
grid_view = xnb.slot_as_array(node.in_slot("grid"))
grid_view.field("rx")         # list of per-cell numpy arrays (non-contiguous)
grid_view.flat_field("rx")    # contiguous copy (same as Tier 3a)
grid_view.n_particles         # total non-ghost particle count
grid_view.cell_offsets        # numpy array of per-cell offsets (for manual indexing)
```

Challenges:
- Per-cell arrays have variable length → no single stride → cannot use
  pybind11's standard buffer protocol for a flat view without a copy.
- The `CellParticles` SoA type is a complex template; the C++ extractor must be
  registered for every concrete `Grid<FieldSet<...>>` instantiation.
- Dynamic/generic fields are not enumerable at compile time; a separate
  runtime-name registry would be needed.

This option is worth revisiting once the copy overhead of Tier 3a is measured
to be a real bottleneck.

---

## Summary table

| Category | Status | How | Python API |
|---|---|---|---|
| Global stats (`SimulationStatistics`) | Planned | `register_slot_extractor` API in onika + extractor in `bind_exanb_data.cpp` | `xnb.read_sim_stats(ctx)` |
| Per-cell (`GridCellValues`) | Planned | Same `bind_exanb_data.cpp`, Python wrapper object | `xnb.read_cell_values(ctx, field)` |
| Per-particle — copy | **Done** | `extract_particle_field` operator + `apply_graph` pattern | `node.slot_as_array("data")` |
| Per-particle — zero-copy | Future (Tier 3b) | C++ Grid extractor registered via same API | `grid_view.field("rx")` |
