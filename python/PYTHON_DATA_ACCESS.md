# Reading simulation data from Python

This page describes how values computed by onika operators reach Python. It applies to `pyonika`, `pyexanbody` and to applications built on them, such as `pyexastamp`.

## Mechanism: the slot extractor registry

Each value an operator produces lives in a slot of that operator. `node.slot_as_array(name)` (or `pyonika.slot_as_array(slot)`) finds a converter for the slot's C++ type in pyonika's extractor registry and returns the resulting Python object. It returns `None` if the type has no extractor or the slot has no value yet.

- **Registry key:** the mangled type name (`typeid(T).name()`). That name is the same in every shared library, so extractors can be registered from separate extension modules that are loaded `RTLD_LOCAL`.
- **Registering an extractor:** extension modules use the PyCapsule `pyonika._register_slot_extractor_fn`. There is no link-time dependency on pyonika. See `bind_exanb_data.cpp`, and exaStamp's `bind_exastamp_data.cpp`.
- **Copies:** every value returned is a copy. It stays valid after `pyonika.end()`, and writing to it does not change the simulation.
- **Timing:** read values after `run()` / `run_node()` and before `end()`, because `end()` frees the graph's resources.

## Registered types

| C++ type | Registered by | Python result |
|---|---|---|
| `std::vector<T>` (arithmetic `T`) | pyonika | 1-D numpy array |
| `std::vector<std::array<T,3 or 4>>` | pyonika | `(N,3)` / `(N,4)` numpy array |
| `exanb::SimulationStatistics` | pyexanbody | dict: `kinetic_energy`, `particle_count`, `min_vel`, `max_vel`, `min_acc`, `max_acc` |
| `exanb::GridCellValues` | pyexanbody | `GridCellValuesView` (field names, shapes; `field()` / `field_inner()` return copies) |
| `exanb::ParticleFieldBundle` | pyexanbody | dict `{field name: numpy array}` (see below) |
| `exaStamp::ThermodynamicState` | pyexastamp | dict of thermodynamic quantities |
| `exaStamp::ParticleSpecies` | pyexastamp | list of dict |
| `exaStamp::MoleculeSpeciesVector` | pyexastamp | dict of molecule definitions |

## Per-particle fields

Particle data is stored per grid cell. To read it, place an `extract_particle_fields` operator in the graph:

```yaml
simulation_epilog:
  - extract_particle_fields:
      fields: [ rx, ry, rz, vx, vy, vz ]   # empty / omitted: every exportable field
      include_ghosts: false
      gather: false                        # true: all MPI processes' particles on rank 0
```

Then call `pyexanbody.particles(ctx)`. It returns one numpy array per field:
- all arrays share the same cell-major particle order;
- each field keeps its native dtype (`id` is `uint64`, `type` is `uint8`);
- `id` is always included when the grid has it.

| Field value type | Array shape |
|---|---|
| arithmetic scalar | `(N,)` |
| `Vec3d` | `(N,3)` |
| `Mat3d` | `(N,3,3)`, row-major (`m11, m12, ...`) |
| `Quaternion` | `(N,4)`, ordered `w, x, y, z` |
| `oarray_t<T,K>` | `(N,K)` |

Requesting an unknown field, or a field whose type cannot be exported, raises an error. In a single-process run that error is a Python `RuntimeError`.

`extract_particle_field` is the older operator: one scalar field per operator, cast to `double` and read with `node.slot_as_array("data")`. It is kept for compatibility.

## Results computed several times

Some operators appear several times in a graph. For example, simulation statistics and the thermodynamic state are computed both in `first_iteration` and in the compute loop.

The `read_*` helpers and `particles()` return the value of the last instance in graph order, which is the most recently updated one. `particles()` and pyexastamp's `read_*` helpers also accept `node=` with an operator name or pathname to choose another instance.
