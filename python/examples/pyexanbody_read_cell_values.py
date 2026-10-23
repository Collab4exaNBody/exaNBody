#!/usr/bin/env python3
"""
Read per-cell grid values as zero-copy numpy arrays using GridCellValuesView.

This example builds a small domain with a spherical region, assigns a scalar
"density" field with set_cell_values (1.0 inside SPHERE, 0.0 elsewhere), then
reads the field back from Python after the simulation initialises.

Two access methods are demonstrated via GridCellValuesView:

  gcv.field("density")
      Flat 1-D view of ALL cells including ghost layers, shape (n_cells,).
      n_cells = nx * ny * nz where nx/ny/nz include the ghost border.
      Useful when you need raw buffer access or want to manage the ghost
      stripping yourself.

  gcv.field_inner("density")
      3-D strided view of domain cells only, shape (inx, iny, inz).
      Ghost layers are skipped via pointer offset and numpy strides — no copy.
      Ready for spatial analysis (slicing, visualisation, numpy operations).

xnb.read_cell_values(graph, field) is a convenience wrapper that calls
field_inner() on the first matching node.

Usage:
    python pyexanbody_read_cell_values.py
"""

import os
import sys
import numpy as np
import pyexanbody as xnb

# ---------------------------------------------------------------------------
# 1. Bootstrap
# ---------------------------------------------------------------------------
main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")
ctx = xnb.init([sys.argv[0], main_config])
if ctx.error_code >= 0:
    sys.exit(ctx.error_code)

# ---------------------------------------------------------------------------
# 2. Configure the simulation.
#
#    10×10×10 grid of 30 m cells (300 m box).
#    A SPHERE of radius 50 m centred in the box is marked with density=1.0;
#    all other cells keep the default value of 0.0.
#
#    set_cell_values runs inside input_data, so grid_cell_values is populated
#    before the compute loop starts.  compute_force is disabled since this
#    example only demonstrates data access, not dynamics.
# ---------------------------------------------------------------------------
xnb.set_operator_defaults({

    "global": {
        "dt":                        "1.0e-3 s",
        "rcut_max":                  "5.0 m",
        "rcut_inc":                  "0.5 m",
        "simulation_log_frequency":  1,
        "simulation_dump_frequency": 1,
        "simulation_end_iteration":  1,
    },

    "input_data": [
        {"particle_types": {
            "particle_type_map": {"A": 0},
        }},
        {"particle_regions": [
            {"SPHERE": {"quadric": {
                "shape": "sphere",
                "transform": [
                    {"scale":     ["50 m", "80 m", "50 m"]},
                    {"xrot": "pi/4."},
                    {"translate": ["150 m", "150 m", "150 m"]},
                ],
            }}},
        ]},
        {"domain": {
            "cell_size":  "10 m",
            "grid_dims":  [30, 30, 30],
            "bounds":     [[0.0, 0.0, 0.0], ["300 m", "300 m", "300 m"]],
            "periodic":   [True, True, True],
            "expandable": False,
            "xform":      [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        }},
        "init_rcb_grid",
        {"lattice": {
            "structure": "SC",
            "types":     ["A"],
            "size":      ["30.0 m", "30.0 m", "30.0 m"],
        }},
        # Assign density=1.0 to cells whose centre falls inside the SPHERE.
        # Cells outside the region keep the default initialisation value of 0.0.
        {"set_cell_values": {
            "grid_subdiv": 2,
            "field_name": "density",
            "value":      [1.0],
            "region":     "SPHERE",
        }},
    ],

    "compute_force": "nop",
    "dump_data":     "nop",
    "final_dump":    "nop",
})

# ---------------------------------------------------------------------------
# 3. Build and run the simulation graph.
# ---------------------------------------------------------------------------
graph = xnb.build_simulation_graph(ctx, ["default_simulation"])
xnb.run_node(ctx, graph)

# ---------------------------------------------------------------------------
# 4. GridCellValuesView — field() vs field_inner()
#
#    node.slot_as_array("grid_cell_values") returns a GridCellValuesView.
#    It exposes two access methods for each field:
#
#    field(name)
#      Flat 1-D view over ALL cells including ghost layers.
#      Shape: (n_cells,) where n_cells = nx * ny * nz (ghost dims).
#      Strides: (tot_comps * 8,)  — one stride, one step per cell.
#      Use this when you need the raw buffer or want to apply your own mask.
#
#    field_inner(name)
#      3-D strided view of inner (domain) cells only — ghosts excluded.
#      Shape: (inx, iny, inz) where inx = nx - 2*gl, etc.
#      Strides: (ny*nz*tot*8, nz*tot*8, tot*8) — spans over ghost cells
#      without copying them.  Ready for spatial operations immediately.
#
#    Both are zero-copy views backed by the same C++ buffer.
# ---------------------------------------------------------------------------
def show_gcv(node):
    gcv = node.slot_as_array("grid_cell_values")
    if gcv is None or not hasattr(gcv, "field_names"):
        return

    nx, ny, nz = gcv.shape
    gl = gcv.ghost_layers
    inx, iny, inz = nx - 2*gl, ny - 2*gl, nz - 2*gl

    print(f"\n=== GridCellValuesView — node: {node.pathname()!r} ===\n")
    print(f"  field_names  : {list(gcv.field_names)}")
    print(f"  ghost_layers : {gl}")
    print(f"  stored dims  : ({nx}, {ny}, {nz})   — includes ghost border")
    print(f"  domain dims  : ({inx}, {iny}, {inz}) — ghost-stripped")

    raw   = gcv.field("density")        # 1-D, all cells including ghosts
    inner = gcv.field_inner("density")  # 3-D, domain cells only

    print(f"\n  field()       shape={raw.shape}         strides={raw.strides}")
    print(f"                n_cells={raw.size}  (domain + ghost border)")
    print(f"                sum={raw.sum():.1f}   (ghost cells are 0, so same as inner)")

    print(f"\n  field_inner() shape={inner.shape}   strides={inner.strides}")
    print(f"                n_cells={inner.size}  (domain only)")
    print(f"                sum={inner.sum():.1f}")

    # Both are zero-copy views of the same C++ buffer.
    # inner's base pointer is offset into raw's memory range by the ghost border.
    raw_start  = raw.ctypes.data
    inner_start = inner.ctypes.data
    raw_end    = raw_start + raw.nbytes
    in_range   = raw_start <= inner_start < raw_end
    print(f"\n  raw   starts at: 0x{raw_start:x}")
    print(f"  inner starts at: 0x{inner_start:x}  (offset by ghost border)")
    print(f"  inner pointer is inside raw's memory range: {in_range}")

graph.apply_graph(show_gcv)

# ---------------------------------------------------------------------------
# 5. High-level helper and spatial visualisation.
#
#    xnb.read_cell_values() is a convenience wrapper around field_inner():
#    it walks the graph, finds the first matching node, and returns the
#    ghost-stripped 3-D view directly.
# ---------------------------------------------------------------------------
density = xnb.read_cell_values(graph, "density")

if density is None:
    print("\nread_cell_values returned None — _exanb_data may not be installed.")
    xnb.end(ctx)
    sys.exit(1)

print(f"\n=== read_cell_values('density') ===\n")
print(f"  shape        : {density.shape}")
print(f"  ndim         : {density.ndim}  {'(inx,iny,inz)' if density.ndim==3 else '(inx,iny,inz,S,S,S)'}")
print(f"  dtype        : {density.dtype}")
print(f"  subcells=1.0 : {int((density == 1.0).sum())}   (inside SPHERE)")
print(f"  subcells=0.0 : {int((density == 0.0).sum())}   (outside SPHERE)")

# Animated sweep through all XY slices along z.
# Each frame overwrites the previous one using ANSI cursor-up escape codes.
import time

CELL_IN  = "▓▓"   # dark shade — inside SPHERE
CELL_OUT = "░░"   # light shade — outside; glyph edges create visible cell boundaries

# Detect subdiv from the array shape:
#   subdiv==1 → shape (inx, iny, inz)
#   subdiv> 1 → shape (inx, iny, inz, S, S, S)
if density.ndim == 3:
    inx, iny, inz_cells = density.shape
    S = 1
else:
    inx, iny, inz_cells, S, _, _ = density.shape

n_frames  = inz_cells * S   # total animation frames along z

def render_slice(z):
    cz = z // S             # cell index along z
    ck = z  % S             # sub-cell index along z (0 when subdiv==1)
    z_label = f"z={cz}" if S == 1 else f"z={cz} sk={ck}"
    header = f"  === XY slice  {z_label}  ({z}/{n_frames-1})  ==="
    rows = [
        "  " + "".join(
            CELL_IN if (density[i, j, cz] if S == 1 else density[i, j, cz, ci, cj, ck]) > 0.5
            else CELL_OUT
            for j in range(iny) for cj in range(S)
        )
        for i in range(inx) for ci in range(S)
    ]
    footer = f"  {CELL_IN} inside SPHERE   {CELL_OUT} outside"
    return [header] + rows + [footer]

prev_lines = 0

try:
    for z in list(range(n_frames)) + list(range(n_frames - 2, -1, -1)):  # forward then back
        frame = render_slice(z)
        if prev_lines:
            # Jump up exactly as many lines as the previous frame had,
            # then erase everything from there to the end of the screen.
            sys.stdout.write(f"\033[{prev_lines}A\033[J")
        sys.stdout.write("\n".join(frame) + "\n")
        sys.stdout.flush()
        prev_lines = len(frame)
        time.sleep(0.12)
except KeyboardInterrupt:
    pass

print()  # leave cursor on a clean line after the animation

xnb.end(ctx)
