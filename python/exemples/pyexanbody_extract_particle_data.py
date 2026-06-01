#!/usr/bin/env python3
"""
Extract per-particle data into numpy arrays using extract_particle_field.

The extract_particle_field operator copies a named scalar field from all
non-ghost grid cells into a contiguous std::vector<double> output slot.
Because std::vector<double> is already registered in pyonika's slot_as_array
registry, the result is immediately readable as a zero-copy numpy array.

This script builds the solar-system gravitational simulation from Python
(same as pyexanbody_reproduce_solar_system_case.py) and replaces Paraview
output with field extractors so all per-particle data lands in Python arrays.

Usage:
    python pyexanbody_extract_particle_data.py
"""

import os
import sys
import numpy as np
import pyexanbody as xnb

# ---------------------------------------------------------------------------
# 1. Bootstrap — load MPI, plugins, and operator definitions
# ---------------------------------------------------------------------------
main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")
ctx = xnb.init([sys.argv[0], main_config])
if ctx.error_code >= 0:
    sys.exit(ctx.error_code)

# ---------------------------------------------------------------------------
# 2. Configure the simulation.
#
#    dump_data is replaced by a list of extract_particle_field operators —
#    one per field.  Each operator reads from the shared grid slot (connected
#    automatically within the graph) and writes to its own data output slot.
# ---------------------------------------------------------------------------
FIELDS = ["rx", "ry", "rz", "vx", "vy", "vz"]

xnb.set_operator_defaults({

    "global": {
        "dt":                        "1 s",
        "rcut_max":                  "280 m",
        "rcut_inc":                  "18 m",
        "simulation_log_frequency":  1,
        "simulation_dump_frequency": 5,  # extract at every iteration
        "simulation_end_iteration":  5,
    },

    "input_data": [
        {"particle_types": {
            "verbose": False,
            "particle_type_map": {"Meteor": 0, "Asteroid": 1, "Moon": 2, "Planet": 3},
            "particle_type_properties": {
                "Meteor":   {"mass": "1.e3 kg", "radius": "15 m"},
                "Asteroid": {"mass": "1.e4 kg", "radius": "30 m"},
                "Moon":     {"mass": "1.e5 kg", "radius": "80 m"},
                "Planet":   {"mass": "1.e6 kg", "radius": "150 m"},
            },
        }},
        {"particle_regions": [
            {"PLANET": {"quadric": {"shape": "sphere", "transform": [
                {"scale":     ["200 m",  "200 m",  "200 m"]},
                {"translate": ["975 m",  "975 m",  "975 m"]},
            ]}}},
            {"MOON": {"quadric": {"shape": "sphere", "transform": [
                {"scale":     ["80 m",     "80 m",     "80 m"]},
                {"translate": ["1072.5 m", "1072.5 m", "1072.5 m"]},
            ]}}},
        ]},
        {"domain": {
            "cell_size":  "150 m",
            "grid_dims":  [13, 13, 13],
            "bounds":     [[0.0, 0.0, 0.0], ["1950 m", "1950 m", "1950 m"]],
            "periodic":   [True, True, True],
            "expandable": False,
            "xform":      [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        }},
        "init_rcb_grid",
        {"lattice": {
            "structure": "CUSTOM",
            "types":     ["Planet", "Moon"],
            "positions": [[0.5, 0.5, 0.5], [0.55, 0.55, 0.55]],
            "size":      ["1950 m", "1950 m", "1950 m"],
        }},
        {"gaussian_noise_r": {"sigma": "1 m"}},
        {"lattice": {
            "structure": "BCC",
            "types":     ["Meteor", "Asteroid"],
            "size":      ["150 m", "150 m", "150 m"],
        }},
        {"gaussian_noise_r": {"sigma": "1 m", "sigma_cut": "2 m",
                              "region": "not ( PLANET or MOON )"}},
        {"set_velocity": {"value": ["1.5 m/s", 0, 0], "region": "MOON"}},
        {"gaussian_noise_v": {"sigma": "2 m/s",
                              "region": "not ( PLANET or MOON )"}},
    ],

    "compute_force": [
        {"gravitational_force": {
            "config": {"G": "6.67e-11 m^3/kg/s^2"},
            "rcut":   "280 m",
        }}
    ],

    # One extract_particle_field operator per field replaces Paraview output.
    # All operators share the same grid input slot (connected by name within
    # the graph); each has its own independent data output slot.
    "final_dump": [
        {"extract_particle_field": {"field_name": fname}}
        for fname in FIELDS
    ],
    "dump_data": "nop",
})

# ---------------------------------------------------------------------------
# 3. Build and run the simulation graph.
#
#    "default_simulation" is the full N-body backbone defined in
#    main-config.msp: input_data → compute_loop (with dump_data) → final_dump.
#    main-config.msp defaults to "test_simulation" (a minimal smoke-test
#    that has no compute_loop), so we must name "default_simulation"
#    explicitly here.
# ---------------------------------------------------------------------------
graph = xnb.build_simulation_graph(ctx, ["default_simulation"])
xnb.run_node(ctx, graph)

# ---------------------------------------------------------------------------
# 4. Collect the output arrays.
#
#    apply_graph() visits every node in the graph depth-first and calls the
#    provided callback.  After run_node(), each extract_particle_field node's
#    data slot holds the values from the last dump step.
#
#    Onika may append "#N" to operator names when several instances of the
#    same operator appear in a sequence, so we use startswith() instead of ==.
#
#    slot_as_array() returns a zero-copy numpy VIEW into the operator's
#    internal std::vector<double>.  Call .copy() (or np.array(..., copy=True))
#    if the array must outlive the graph or the end() call below.
# ---------------------------------------------------------------------------
particle_data = {}
visited_names = []   # kept for diagnostics

def collect_extractor(node):
    visited_names.append(node.name())
    if not node.name().startswith("extract_particle_field"):
        return
    # slot_values() returns {slot_name: value_as_string} for all slots
    fname = node.slot_values().get("field_name", "").strip('"\'')
    if not fname:
        return
    arr = node.slot_as_array("data")
    if arr is None:
        print(f"  warning: no data for field '{fname}' (operator may not have run)")
        return
    particle_data[fname] = np.array(arr, copy=True)  # own the memory

graph.apply_graph(collect_extractor)

# ---------------------------------------------------------------------------
# 5. Print a summary and demonstrate basic numpy usage
# ---------------------------------------------------------------------------
if not particle_data:
    print("No data extracted.")
    print(f"  Graph contained {len(visited_names)} nodes.")
    print(f"  Node names seen: {sorted(set(visited_names))}")
    print("  Check that: (1) extract_particle_field is in plugins.db,")
    print("  (2) simulation_dump_frequency allows at least one dump step.")
    xnb.end(ctx)
    sys.exit(1)

n = len(next(iter(particle_data.values())))
print(f"\n{n} non-ghost particles after last dump step\n")
print(f"{'field':>6}   {'min':>14}   {'max':>14}   {'mean':>14}")
print("-" * 58)
for fname in FIELDS:
    arr = particle_data.get(fname)
    if arr is None:
        print(f"{fname:>6}   (not extracted)")
        continue
    print(f"{fname:>6}   {arr.min():14.6g}   {arr.max():14.6g}   {arr.mean():14.6g}")

# Assemble a (N, 3) position matrix and velocity matrix from the 1-D arrays
if all(f in particle_data for f in ("rx", "ry", "rz")):
    pos = np.stack([particle_data["rx"],
                    particle_data["ry"],
                    particle_data["rz"]], axis=1)  # shape (N, 3)
    print(f"\nPosition matrix  : shape={pos.shape}, dtype={pos.dtype}")
    print(f"Centre of mass   : {pos.mean(axis=0)}")

if all(f in particle_data for f in ("vx", "vy", "vz")):
    vel = np.stack([particle_data["vx"],
                    particle_data["vy"],
                    particle_data["vz"]], axis=1)  # shape (N, 3)
    speeds = np.linalg.norm(vel, axis=1)
    print(f"\nVelocity matrix  : shape={vel.shape}, dtype={vel.dtype}")
    print(f"Speed  min/max   : {speeds.min():.4g} / {speeds.max():.4g}")

# ---------------------------------------------------------------------------
# 6. Finalise
# ---------------------------------------------------------------------------
xnb.end(ctx)
