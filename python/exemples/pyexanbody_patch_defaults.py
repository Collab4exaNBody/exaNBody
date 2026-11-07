#!/usr/bin/env python3
"""
Pattern C — load main-config.msp then patch a few values before running.

init() builds the simulation graph internally from the .msp file but does
NOT run it.  set_operator_defaults() merges new values into the factory
defaults (only the supplied keys are overridden; everything else comes from
the .msp).  build_simulation_graph() then produces a new graph from the
updated defaults, and run_node() executes it.

IMPORTANT: run(ctx) would execute the graph built during init(), before any
set_operator_defaults() call — your changes would have no effect.
Always use run_node(ctx, graph) after build_simulation_graph().

Usage:
    python pyexanbody_patch_defaults.py
"""

import os, sys
import pyexanbody as xnb

main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")

ctx = xnb.init([sys.argv[0], main_config])
if ctx.error_code >= 0:
    sys.exit(ctx.error_code)

# Patch only the keys you want to change.
# Everything else (input_data, compute_force, …) comes from main-config.msp.
xnb.set_operator_defaults({
    "global": {
        "simulation_end_iteration": 19,
        "simulation_log_frequency": 1,
    },
})

# Rebuild the graph reusing the simulation: structure stored by init().
# build_simulation_graph(ctx) with no list is equivalent to
# build_simulation_graph(ctx, ["default_simulation"]) here.
graph = xnb.build_simulation_graph(ctx)
xnb.run_node(ctx, graph)

stats = xnb.read_sim_stats(graph)
if stats:
    print(f"particle_count  = {stats['particle_count']}")
    print(f"kinetic_energy  = {stats['kinetic_energy']:.6g}")

xnb.end(ctx)
