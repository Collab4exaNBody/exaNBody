#!/usr/bin/env python3
"""
Read global simulation statistics into a Python dict using read_sim_stats.

xnb.read_sim_stats(graph) finds the simulation_stats operator node in the
graph (which is always present in default_simulation) and returns a dict
with the values from its last execution:

    {"kinetic_energy": float, "particle_count": int,
     "min_vel": float, "max_vel": float,
     "min_acc": float, "max_acc": float}

This works because the _exanb_data extension (Tier 1 of PYTHON_DATA_ACCESS.md)
registers a SimulationStatistics extractor into pyonika's slot_as_array
registry, so slot_as_array returns a dict instead of None for that slot type.

The example also demonstrates combining read_sim_stats with
extract_particle_field (Tier 3a) to get a full per-step snapshot.

Usage:
    python pyexanbody_read_sim_stats.py
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
#    simulation_stats is already part of default_simulation (inside
#    first_iteration and print_log_if_triggered), so no extra operators are
#    needed to collect it.  We add extract_particle_field to final_dump to
#    also get per-particle positions at the last step.
# ---------------------------------------------------------------------------
xnb.set_operator_defaults({

    "global": {
        "dt":                        "1 s",
        "rcut_max":                  "280 m",
        "rcut_inc":                  "18 m",
        "simulation_log_frequency":  1,
        "simulation_dump_frequency": 5,
        "simulation_end_iteration":  10,
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

    # Disable Paraview output; use final_dump to extract particle positions
    # at the last step alongside the statistics.
    "dump_data": "nop",
    "final_dump": [
        {"extract_particle_field": {"field_name": fname}}
        for fname in ("rx", "ry", "rz")
    ],
})

# ---------------------------------------------------------------------------
# 3. Build and run the simulation graph.
#    Use "default_simulation" explicitly — main-config.msp defaults to
#    "test_simulation" which has no compute_loop.
# ---------------------------------------------------------------------------
graph = xnb.build_simulation_graph(ctx, ["default_simulation"])
xnb.run_node(ctx, graph)

# ---------------------------------------------------------------------------
# 4. Read global simulation statistics.
#
#    read_sim_stats() traverses the graph with apply_graph(), finds the
#    simulation_stats operator node, and returns the dict from its output
#    slot via the SimulationStatistics extractor registered by _exanb_data.
#    The values correspond to the last iteration of the simulation.
# ---------------------------------------------------------------------------
stats = xnb.read_sim_stats(graph)

if not stats:
    print("No simulation statistics found — running in debug mode:")
    xnb.read_sim_stats(graph, _debug=True)
    xnb.end(ctx)
    sys.exit(1)

print("\n=== Simulation statistics (last iteration) ===\n")
print(f"  Particle count  : {stats['particle_count']}")
print(f"  Kinetic energy  : {stats['kinetic_energy']:.6g} J")
print(f"  Velocity range  : [{stats['min_vel']:.4g}, {stats['max_vel']:.4g}] m/s")
print(f"  Accel range     : [{stats['min_acc']:.4g}, {stats['max_acc']:.4g}] m/s²")

# ---------------------------------------------------------------------------
# 5. Read per-particle positions from final_dump extractors.
#
#    apply_graph() collects each extract_particle_field node's data slot.
#    slot_values()["field_name"] identifies which field each node extracted.
# ---------------------------------------------------------------------------
particle_data = {}

def collect_extractor(node):
    if not node.name().startswith("extract_particle_field"):
        return
    fname = node.slot_values().get("field_name", "").strip('"\'')
    arr = node.slot_as_array("data")
    if arr is not None and fname:
        particle_data[fname] = np.array(arr, copy=True)

graph.apply_graph(collect_extractor)

if all(f in particle_data for f in ("rx", "ry", "rz")):
    pos = np.stack([particle_data["rx"],
                    particle_data["ry"],
                    particle_data["rz"]], axis=1)  # shape (N, 3)
    print(f"\n=== Final particle positions ===\n")
    print(f"  Particle count  : {pos.shape[0]}")
    print(f"  Centre of mass  : {pos.mean(axis=0)}")
    print(f"  Bounding box    : [{pos.min(axis=0)}, {pos.max(axis=0)}]")

    # Quick consistency check: particle_count from stats vs actual array length
    if stats["particle_count"] != pos.shape[0]:
        print(f"\n  Note: stats particle_count={stats['particle_count']} differs from "
              f"extracted count={pos.shape[0]} (MPI reduce vs local rank).")

# ---------------------------------------------------------------------------
# 6. Finalise
# ---------------------------------------------------------------------------
xnb.end(ctx)
