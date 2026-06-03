#!/usr/bin/env python3
"""
Python equivalent of:
    onika-exec data/config/main-config.msp
OR
    exaNBody data/config/main-config.msp

Requires the pyexanbody module installed (make install with -DEXANB_BUILD_PYTHON=ON)
and the environment sourced:

    source <exanbody-install-prefix>/bin/setup-env.sh
    python pyexanbody_run_main_config.py
"""
import os
import sys
import pyexanbody

main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")
ctx = pyexanbody.init([sys.argv[0], main_config, "--nogpu"])
if ctx.error_code >= 0:
    sys.exit(ctx.error_code)

print(f"[pyexanbody] rank={ctx.mpi_rank}/{ctx.mpi_nprocs}  "
      f"cpus={ctx.cpucount}  gpus={ctx.ngpus}")

# Overriding the 10000 steps default value in global block
pyexanbody.set_operator_defaults({
    "global": {
        "simulation_end_iteration":  10,
    }
})

# --- inspect the simulation graph -----------------------------------
print("\n[pyexanbody] === simulation graph ===")
operators = []
root = ctx.node("simulation")
if root is not None:
    root.apply_graph(operators.append)
    for op in operators:
        indent = "  " * op.depth()
        print(f"  {indent}{op.pathname()}")
        for name, slot in op.in_slots():
            val = slot.value_as_string() if slot.has_value() else "<unset>"
            print(f"  {indent}  in  {name}: {slot.value_type()} = {val}")
        for name, slot in op.out_slots():
            val = slot.value_as_string() if slot.has_value() else "<unset>"
            print(f"  {indent}  out {name}: {slot.value_type()} = {val}")

# --- factory access -------------------------------------------------
print("\n[pyexanbody] === registered operators ===")
for name in pyexanbody.available_operators():
    print(f"  {name}")

print("\n[pyexanbody] === make_operator demo ===")
op = pyexanbody.make_operator("unit_system", {"verbose": True})
print(f"  created: {op}")
for name, slot in op.in_slots():
    val = slot.value_as_string() if slot.has_value() else "<unset>"
    print(f"    in  {name}: {slot.value_type()} = {val}")

print("\n[pyexanbody] === make_operator demo ===")
op = pyexanbody.make_operator("global", {"dt": 1.0, "nsteps": 20, "timestep": 0, "compute_loop_continue": True})
print(f"  created: {op}")
for name, slot in op.in_slots():
    val = slot.value_as_string() if slot.has_value() else "<unset>"
    print(f"    in  {name}: {slot.value_type()} = {val}")

# --- run & end ------------------------------------------------------
graph = pyexanbody.build_simulation_graph(ctx)
pyexanbody.run_node(ctx, graph)
pyexanbody.end(ctx)

