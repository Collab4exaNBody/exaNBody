# Python data access — design document

This document describes how simulation data is exposed from exaNBody into
Python objects (numpy arrays, dicts).  Three categories of data are targeted:

1. **Per-particle fields** — positions, velocities, forces, IDs, …
2. **Per-cell / grid-cell fields** — `GridCellValues` named scalar/vector fields
3. **Global thermodynamic variables** — kinetic energy, particle count, …

---

## Zero-copy vs copy-on-read

### What is zero-copy?

A **zero-copy** view means the returned Python/numpy object holds a raw pointer
directly into the C++ memory buffer — no data is duplicated.  When you index
`arr[i, j, k]` in Python, it reads directly from the C++ allocation.

The mechanism is numpy's **strided buffer protocol**: a numpy array is fully
described by `(pointer, shape, strides, dtype)`.  By setting the pointer to
`data.data() + offset` and computing strides that match the existing C++ AoS
cell-major layout, the numpy array describes the C++ memory without moving any
bytes.

```
C++ buffer:  [cell0_f0 | cell0_f1 | cell1_f0 | cell1_f1 | ...]
                  ↑                      ↑
             numpy ptr              numpy ptr + stride
```

### Lifetime requirement

Because the numpy array does not own the memory, it is only valid **while the
`ApplicationContext` (`ctx`) is alive**.  Calling `xnb.end(ctx)` destroys the
C++ object and the numpy array becomes a dangling pointer.

Always copy before `end()` if you need the data beyond the simulation:

```python
owned = np.array(gcv.field_inner("density"), copy=True)  # safe after end()
xnb.end(ctx)
```

### What is NOT zero-copy?

`SimulationStatistics` (Tier 1) is a **copy**: the extractor reads each C++
field (`m_kinetic_energy`, `m_particle_count`, …) and inserts them into a new
Python `dict`.  No pointer into C++ memory is held, so lifetime does not matter.

`extract_particle_field` (Tier 3a) is also a **copy**: the operator writes a
fresh `std::vector<double>` at each dump step.  The returned numpy view is
zero-copy *of that vector*, but the vector itself is a copy of the particle data.

| API | Memory model | Lifetime constraint |
|---|---|---|
| `gcv.field(name)` | **zero-copy** — pointer into `CudaMMVector<double>` | `ctx` must stay alive |
| `gcv.field_inner(name)` | **zero-copy** — same pointer, ghost-skipping strides | `ctx` must stay alive |
| `xnb.read_cell_values(graph, name)` | **zero-copy** (calls `field_inner`) | `ctx` must stay alive |
| `xnb.read_sim_stats(graph)` | **copy** — Python dict built by value | no constraint |
| `node.slot_as_array("data")` (Tier 3a) | zero-copy *of vector* | `ctx` must stay alive |

---

## Background: the `slot_as_array` mechanism

`pyonika`'s `bind_soatl.cpp` maintains a static registry keyed by the mangled
type name (`typeid(T).name()`).  `slot_as_array(slot)` does an O(1) lookup via
`slot.value_type()` and calls the matching extractor — no `dynamic_cast` needed.
This makes cross-DSO registration safe even with `RTLD_LOCAL`.

New extractors are registered via:

```cpp
// onika/python/bind_soatl.h
void register_slot_extractor(std::type_index ti, SlotExtractorFn fn);
```

The function pointer is exposed as a `PyCapsule` at `pyonika._register_slot_extractor_fn`
so that other `RTLD_LOCAL` pybind11 extensions (e.g. `_exanb_data`) can call it
without requiring `RTLD_GLOBAL` or link-time dependencies on `pyonika.so`.

Registered types (onika side, all zero-copy):

| C++ type | numpy shape |
|---|---|
| `std::vector<T>` (scalar arithmetic T) | `(N,)` |
| `std::vector<std::array<T,3>>` | `(N,3)` |
| `std::vector<std::array<T,4>>` | `(N,4)` |

Additional types registered by `_exanb_data` (exaNBody side):

| C++ type | Python object | Memory model |
|---|---|---|
| `SimulationStatistics` | `dict` | copy |
| `GridCellValuesT<double>` | `GridCellValuesView` | zero-copy |

---

## Tier 1 — Global thermodynamic variables (`SimulationStatistics`)

**Status: DONE**

`SimulationStatistics` is a plain 6-field struct:

```cpp
struct SimulationStatistics {
    double             m_kinetic_energy = 0.0;
    double             m_min_vel, m_max_vel;
    double             m_min_acc, m_max_acc;
    unsigned long long m_particle_count = 0;
};
```

The extractor (in `python/bind_exanb_data.cpp`) reads each field and inserts
it into a new `py::dict`.  This is a **copy**: no pointer into C++ memory is
held, so the dict is safe to use after `xnb.end(ctx)`.

```python
stats = xnb.read_sim_stats(graph)
# → {"kinetic_energy": 1.23e9, "particle_count": 500,
#    "min_vel": 0.0, "max_vel": 42.3, "min_acc": …, "max_acc": …}
```

**Complete working example**: `python/exemples/pyexanbody_read_sim_stats.py`

---

## Tier 2 — Per-cell data (`GridCellValuesT<double>`)

**Status: DONE**

### Data layout

`GridCellValuesT<double>` stores a flat `CudaMMVector<double>` buffer in
cell-major AoS layout:

```
buffer = [cell0: f0_sc000 f0_sc001 … f1_sc000 … | cell1: … | …]
```

Each cell holds `total_comps` doubles, where `total_comps` is the sum of
`m_components` across all fields.  For a field `f`:

- `f.m_offset` — start of this field's block within each cell
- `f.m_subdiv` — sub-cell subdivision factor S (1 = no subdivision)
- `f.m_components` = S³ × ncomps_per_subcell

### Python API — `GridCellValuesView`

`node.slot_as_array("grid_cell_values")` returns a `GridCellValuesView` object:

| Attribute / method | Type | Description |
|---|---|---|
| `field_names` | `list[str]` | Names of all registered fields |
| `shape` | `tuple(nx, ny, nz)` | Stored grid dims **including ghost layers** |
| `n_cells` | `int` | Total cells including ghost layers |
| `ghost_layers` | `int` | Number of ghost layers on each side |
| `has_field(name)` | `bool` | Field existence check |
| `field(name)` | numpy array | Zero-copy, includes ghost cells |
| `field_inner(name)` | numpy array | Zero-copy, ghost-stripped |

#### `field(name)` — raw view with ghost cells

Returns a zero-copy numpy view of the flat C++ buffer.  Shape depends on
`m_subdiv` (S) and `ncomps`:

| subdiv | ncomps | shape |
|---|---|---|
| 1 | 1 | `(n_cells,)` |
| 1 | >1 | `(n_cells, ncomps)` |
| >1 | 1 | `(n_cells, S, S, S)` |
| >1 | >1 | `(n_cells, S, S, S, ncomps)` |

Strides: `(total_comps·8, S²·ncomps·8, S·ncomps·8, ncomps·8[, 8])`.

Use this when you need the raw buffer or want to apply your own ghost mask.

#### `field_inner(name)` — ghost-stripped 3-D view

Returns a zero-copy numpy view with ghost cells excluded, using a pointer offset
to skip the ghost border and strides to span over it:

| subdiv | ncomps | shape |
|---|---|---|
| 1 | 1 | `(inx, iny, inz)` |
| 1 | >1 | `(inx, iny, inz, ncomps)` |
| >1 | 1 | `(inx, iny, inz, S, S, S)` |
| >1 | >1 | `(inx, iny, inz, S, S, S, ncomps)` |

where `inx = nx - 2·ghost_layers`, etc.

Strides: `(ny·nz·tot·8, nz·tot·8, tot·8, S²·ncomps·8, S·ncomps·8, ncomps·8[, 8])`.

**No data is copied** — the strides bridge over the ghost border cells.
The base pointer is `data.data() + (gl·ny·nz + gl·nz + gl)·total_comps + f.m_offset`.

#### `xnb.read_cell_values(graph, field_name)`

Convenience wrapper that walks the graph with `apply_graph`, finds the first
node with a matching `grid_cell_values` slot, and returns `field_inner(name)`.

```python
density = xnb.read_cell_values(graph, "density")
# subdiv=1 → shape (inx, iny, inz)
# subdiv=S → shape (inx, iny, inz, S, S, S)
```

**Lifetime**: the returned array is a zero-copy view — keep `ctx` alive.

### Sub-cell subdivision (`grid_subdiv`)

`set_cell_values` accepts a `grid_subdiv: S` parameter that divides each cell
into an S×S×S block of sub-cells, giving higher spatial resolution than the
simulation grid.

With `grid_subdiv: 2` and a 28×28×28 inner grid, `field_inner("density")`
returns shape `(28, 28, 28, 2, 2, 2)` — 6 axes total.  Axes 0–2 are cell
indices, axes 3–5 are sub-cell indices within each cell.  All six dimensions
are navigable via zero-copy strides.

**Complete working example**: `python/exemples/pyexanbody_read_cell_values.py`

---

## Tier 3 — Per-particle fields (Grid)

The `Grid<FieldSet<...>>` stores particles in a **cell-of-SoA** layout: each
cell holds a separate allocation for each field.  Data for a given field (e.g.
all `rx` values) is therefore non-contiguous across cells.

---

### Tier 3a — Copy-on-read via `extract_particle_field` operator (DONE)

A dedicated exaNBody operator copies a named field from all non-ghost cells into
a contiguous `std::vector<double>` output slot.  `std::vector<double>` is
already registered in `slot_as_array`, so no new C++ bindings are needed.

The returned numpy array is a **zero-copy view of the vector**, but the vector
itself is filled by copying from the per-cell SoA.

**Operator**: `extract_particle_field`

| Slot | Direction | Type | Description |
|---|---|---|---|
| `grid` | IN | `GridT` | Particle grid (REQUIRED) |
| `field_name` | IN | `std::string` | Field to extract (`rx`, `ry`, `rz`, `vx`, …) |
| `include_ghosts` | IN | `bool` | Include ghost-layer particles (default false) |
| `data` | OUT | `std::vector<double>` | Flattened per-particle values cast to double |

```python
xnb.set_operator_defaults({
    "final_dump": [
        {"extract_particle_field": {"field_name": "rx"}},
        {"extract_particle_field": {"field_name": "ry"}},
        {"extract_particle_field": {"field_name": "rz"}},
    ],
})
graph = xnb.build_simulation_graph(ctx, ["default_simulation"])
xnb.run_node(ctx, graph)

particle_data = {}
def collect(node):
    if not node.name().startswith("extract_particle_field"):
        return
    fname = node.slot_values().get("field_name", "").strip('"\'')
    arr = node.slot_as_array("data")   # zero-copy view of the vector
    if arr is not None and fname:
        particle_data[fname] = np.array(arr, copy=True)  # own the data

graph.apply_graph(collect)
pos = np.stack([particle_data["rx"], particle_data["ry"], particle_data["rz"]], axis=1)
```

**Complete working example**: `python/exemples/pyexanbody_extract_particle_data.py`

**Implementation**: `src/compute/extract_particle_field.cpp`

---

### Tier 3b — Zero-copy non-contiguous view (FUTURE WORK)

A C++ extractor registered via `register_slot_extractor` could expose the
Grid's cell-of-SoA layout directly.  This is not yet implemented; per-cell
arrays have variable length, making a single-stride buffer protocol impossible
without a copy or a custom Python sequence object.

---

## Summary table

| Tier | Category | Status | Memory model | Python API |
|---|---|---|---|---|
| 1 | Global stats (`SimulationStatistics`) | **Done** | copy (dict by value) | `xnb.read_sim_stats(graph)` |
| 2 | Per-cell (`GridCellValues`) | **Done** | **zero-copy** | `xnb.read_cell_values(graph, field)` |
| 3a | Per-particle — copy | **Done** | copy into vector, zero-copy view | `node.slot_as_array("data")` |
| 3b | Per-particle — zero-copy | Future | zero-copy (planned) | `grid_view.field("rx")` |
