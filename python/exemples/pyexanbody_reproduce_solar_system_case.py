#!/usr/bin/env python3
"""
Python reproduction of contribs/microCosmos/samples/simple_example/solar_system.msp using pyexanbody.

This script builds the gravitational N-body solar-system simulation entirely
from Python using set_operator_defaults() + build_simulation_graph(), without
reading solar_system.msp.  It is the exaNBody equivalent of onika's
print_loop.py example.

Mapping to the .msp file
------------------------
.msp top-level key          Python call
--------------------------  --------------------------------------------------
global: { ... }             set_operator_defaults({"global": { ... }})
input_data: [ ... ]         set_operator_defaults({"input_data": [ ... ]})
compute_force: [ ... ]      set_operator_defaults({"compute_force": [ ... ]})
dump_data: dump_data_paraview  set_operator_defaults({"dump_data": [...]})
simulation: default_simulation  build_simulation_graph(ctx, ["default_simulation"])

Usage:
    python solar_system_graph.py
"""

import os
import sys
import pyexanbody as xnb

# ---------------------------------------------------------------------------
# 1. Bootstrap — load MPI, OpenMP, plugins and all operator definitions from
#    the exaNBody main config (main-config.msp and its includes).
#    This does NOT run any simulation graph.
# ---------------------------------------------------------------------------
main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")
ctx = xnb.init([sys.argv[0], main_config])
if ctx.error_code >= 0:
    sys.exit(ctx.error_code)
# ---------------------------------------------------------------------------
# 2. Override operator defaults.
#    Equivalent to the top-level non-simulation keys in solar_system.msp.
# ---------------------------------------------------------------------------
xnb.set_operator_defaults({

    # ---- global: simulation-wide scalar parameters -------------------------
    # .msp equivalent:
    #   global:
    #     dt: 1 s
    #     rcut_max: 280 m
    #     ...
    "global": {
        "dt":                        "1 s",
        "rcut_max":                  "280 m",
        "rcut_inc":                  "18 m",
        "simulation_log_frequency":  1,
        "simulation_dump_frequency": 5,
        "simulation_end_iteration":  10,
    },

    # ---- input_data: domain definition + particle placement ----------------
    # .msp equivalent:
    #   input_data:
    #     - particle_types: { ... }
    #     - particle_regions: [ ... ]
    #     - domain: { ... }
    #     - init_rcb_grid
    #     - lattice: { structure: CUSTOM, ... }
    #     ...
    "input_data": [
        {"particle_types": {
            "verbose": True,
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
                {"scale":     ["200 m",    "200 m",    "200 m"]},
                {"translate": ["975 m",    "975 m",    "975 m"]},
            ]}}},
            {"MOON": {"quadric": {"shape": "sphere", "transform": [
                {"scale":     ["80 m",      "80 m",      "80 m"]},
                {"translate": ["1072.5 m",  "1072.5 m",  "1072.5 m"]},
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
        # Place Planet and Moon at custom fractional positions
        {"lattice": {
            "structure": "CUSTOM",
            "types":     ["Planet", "Moon"],
            "positions": [[0.5, 0.5, 0.5], [0.55, 0.55, 0.55]],
            "size":      ["1950 m", "1950 m", "1950 m"],
        }},
        {"gaussian_noise_r": {"sigma": "1 m"}},
        # Fill the rest of the grid with Meteors and Asteroids on a BCC lattice
        {"lattice": {
            "structure": "BCC",
            "types":     ["Meteor", "Asteroid"],
            "size":      ["150 m", "150 m", "150 m"],
        }},
        {"gaussian_noise_r": {
            "sigma":     "1 m",
            "sigma_cut": "2 m",
            "region":    "not ( PLANET or MOON )",
        }},
        {"set_velocity": {
            "value":  ["1.5 m/s", 0, 0],
            "region": "MOON",
        }},
        {"gaussian_noise_v": {
            "sigma":  "2 m/s",
            "region": "not ( PLANET or MOON )",
        }},
    ],

    # ---- compute_force: gravitational interaction --------------------------
    # .msp equivalent:
    #   compute_force:
    #     - gravitational_force:
    #         config: { G: 6.67e-11 m^3/kg/s^2 }
    #         rcut: 280 m
    "compute_force": [
        {"gravitational_force": {
            "config": {"G": "6.67e-11 m^3/kg/s^2"},
            "rcut":   "280 m",
        }}
    ],

    # ---- dump_data: output format ------------------------------------------
    # .msp equivalent:  dump_data: dump_data_paraview
    "dump_data": "dump_data_paraview",

})

# ---------------------------------------------------------------------------
# 3. Build the simulation graph and run it.
#
#    "default_simulation" is defined in main-config.msp as the standard
#    exaNBody backbone (logo_banner → mpi_comm_world → global → input_data →
#    nbh_dist → first_iteration → compute_loop → final_dump → …).
#    The operator defaults we set above are picked up transparently.
#
#    This is equivalent to:  simulation: default_simulation
# ---------------------------------------------------------------------------
#graph = xnb.build_simulation_graph(ctx, ["simulation"])
graph = xnb.build_simulation_graph(ctx, ["default_simulation"])
#graph = xnb.build_simulation_graph(ctx, ["test_simulation"])
#graph = xnb.build_simulation_graph(ctx)
xnb.run_node(ctx, graph)

xnb.end(ctx)
