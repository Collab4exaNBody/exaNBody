"""
pyexanbody — Python interface to the exaNBody N-body simulation framework.

This package sets up the onika runtime environment for exaNBody (plugins,
configuration and data directories), loads the exaNBody slot extractors and
re-exports the pyonika API, so user code only needs to import this module.

Typical usage::

    import pyexanbody as xnb

    # Run an installed example by bare filename — no full path needed
    xnb.run_example("solar_system.msp")

    # Run any .msp file by path
    xnb.run_file("my_sim.msp")

    # Or drive the simulation graph from Python
    ctx = xnb.init([sys.argv[0], "main-config.msp"])
    graph = xnb.build_simulation_graph(ctx, ["global", "domain", ...])
    xnb.run_node(ctx, graph)
    particles = xnb.particles(graph)   # needs an extract_particle_fields node
    xnb.end(ctx)
"""

import os
import sys

import pyonika
from pyonika import (  # noqa: F401  (re-exported API)
    init,
    run,
    run_node,
    end,
    make_operator,
    available_operators,
    get_operator_defaults,
    set_operator_defaults,
    build_simulation_graph,
    slot_as_array,
    setup_env,
    ApplicationContext,
    OperatorNode,
    OperatorSlotBase,
    OnikaError,
)
from ._paths import ONIKA_INSTALL_DIR, XNB_INSTALL_DIR

# Public constant: root of the installed exaNBody share tree.
XNB_SHARE_DIR = os.path.join(XNB_INSTALL_DIR, "share")

# onika + exaNBody plugins, configuration and data directories.
# Applications built on exaNBody call setup_env() again with their own prefix.
setup_env([ONIKA_INSTALL_DIR, XNB_INSTALL_DIR])

# Registers extractors for SimulationStatistics, GridCellValues and
# ParticleFieldBundle into pyonika's slot_as_array registry.
from . import _exanb_data  # noqa: E402,F401


def _share_subdirs() -> list:
    """Return all first-level subdirectories of XNB_SHARE_DIR that exist."""
    if not os.path.isdir(XNB_SHARE_DIR):
        return []
    return [
        os.path.join(XNB_SHARE_DIR, d)
        for d in os.listdir(XNB_SHARE_DIR)
        if os.path.isdir(os.path.join(XNB_SHARE_DIR, d))
    ]


# ---------------------------------------------------------------------------
# 6. exaNBody-level convenience helpers.
# ---------------------------------------------------------------------------

def find_example(name: str) -> str:
    """Resolve an example name to its full path by searching XNB_SHARE_DIR.

    ``name`` can be a bare filename (``"solar_system.msp"``) or a relative
    path within a share subdirectory (``"simple_example/solar_system.msp"``).
    The search walks all first-level subdirectories of ``XNB_SHARE_DIR``.

    Returns the absolute path of the first match found.
    Raises ``FileNotFoundError`` if no match is found.

    Examples::

        xnb.find_example("solar_system.msp")
        # → '.../install-exanbody/share/examples/simple_example/solar_system.msp'

        xnb.find_example("deterministic/deterministic_lj.msp")
        # → '.../install-exanbody/share/examples/deterministic/deterministic_lj.msp'
    """
    for share_sub in _share_subdirs():
        candidate = os.path.join(share_sub, name)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
        # Also walk one level deeper to match bare filenames inside subdirs.
        for entry in os.scandir(share_sub):
            if entry.is_dir():
                candidate = os.path.join(entry.path, name)
                if os.path.isfile(candidate):
                    return os.path.abspath(candidate)
    raise FileNotFoundError(
        f"Example '{name}' not found under {XNB_SHARE_DIR}.\n"
        f"Available examples: {list_examples()}"
    )


def list_examples() -> list:
    """Return a sorted list of all installed example .msp files.

    Each entry is a relative path from its share subdirectory root, suitable
    for passing directly to ``find_example()`` or ``run_example()``.

    Example::

        xnb.list_examples()
        # ['deterministic/deterministic_lj.msp', ..., 'simple_example/solar_system.msp', ...]
    """
    found = []
    for share_sub in _share_subdirs():
        for dirpath, _, filenames in os.walk(share_sub):
            for fname in filenames:
                if fname.endswith(".msp"):
                    rel = os.path.relpath(os.path.join(dirpath, fname), share_sub)
                    found.append(rel)
    return sorted(found)


def run_example(name: str, argv0: str = "pyexanbody", **extra_args) -> int:
    """Find an installed example by name and run it end-to-end.

    ``name`` is resolved via ``find_example()`` — a bare filename or a
    relative path within a share subdirectory is accepted.

    Args:
        name:       Example filename or relative path (e.g. ``"solar_system.msp"``).
        argv0:      Program name shown in onika usage messages.
        **extra_args: forwarded to ``run_file()`` — see its docstring.

    Returns:
        0 on normal completion, or the early-exit code from onika.
    """
    return run_file(find_example(name), argv0=argv0, **extra_args)


def run_file(msp_path: str, argv0: str = "pyexanbody", overrides: dict = None, **extra_args) -> int:
    """Run a complete simulation from an .msp file and return 0 on success.

    Equivalent to: init → run → end, with early-exit handling.

    Args:
        msp_path:   Path to the .msp configuration file.
        argv0:      Program name shown in onika usage messages.
        overrides:  operator definitions merged on top of the input file, e.g.
                    ``{"global": {"dt": 1e-3, "simulation_end_iteration": 100}}``.
        **extra_args: ``configuration:`` options, e.g. ``omp_num_threads=4``
                    (nested dicts allowed, e.g. ``logging={"debug": True}``).

    Returns:
        0 on normal completion, or the early-exit code from onika.
    """
    return pyonika.run_file(os.path.abspath(msp_path), program=argv0,
                            overrides=overrides, config=extra_args)


def read_cell_values(graph, field_name: str):
    """Return a numpy array (copy) for a named grid cell field, ghost-stripped.

    Searches the simulation graph for any operator with a ``grid_cell_values``
    slot that contains the requested field and returns the inner (non-ghost)
    cells.  The array is a copy and stays valid after ``end()``.

    For scalar fields the shape is ``(inx, iny, inz)`` (the domain cell dims).
    For multi-component fields the shape is ``(inx, iny, inz, n_components)``.
    Use ``GridCellValuesView.field()`` directly if you need the raw flat array
    including ghost cells.

    Args:
        graph:       The ``OperatorNode`` returned by ``build_simulation_graph()``.
        field_name:  Name of the field to retrieve (e.g. ``"density"``).

    Returns:
        A numpy array, or ``None`` if no matching field is found.
    """
    result = None

    def _collect(node):
        nonlocal result
        if result is not None:
            return
        val = node.slot_as_array("grid_cell_values")
        if val is None or not hasattr(val, "has_field"):
            return
        if val.has_field(field_name):
            result = val.field_inner(field_name)

    graph.apply_graph(_collect)
    return result


def read_sim_stats(graph, _debug: bool = False) -> dict:
    """Return simulation statistics from the most recent execution of the
    ``simulation_stats`` operator as a Python dict.

    Uses the ``SimulationStatistics`` extractor registered by ``_exanb_data``.

    Args:
        graph:  The ``OperatorNode`` returned by ``build_simulation_graph()``.
        _debug: Print diagnostic information when True.

    Returns:
        dict with keys: ``kinetic_energy``, ``particle_count``,
        ``min_vel``, ``max_vel``, ``min_acc``, ``max_acc``.
        Empty dict if no ``simulation_stats`` node was found or if
        ``_exanb_data`` is not installed.
    """
    result = {}
    candidates = []

    def _collect(node):
        if node.name() != "simulation_stats":
            return
        candidates.append(node.pathname())
        val = node.slot_as_array("simulation_stats")
        if _debug:
            print(f"  [read_sim_stats] node={node.pathname()!r}  "
                  f"slot_as_array → type={type(val).__name__}  value={val!r}")
        if isinstance(val, dict):
            # several instances may exist (first_iteration, compute loop):
            # the last one in graph order is the most recently updated
            result.clear()
            result.update(val)

    graph.apply_graph(_collect)

    if _debug and not result:
        print(f"  [read_sim_stats] found {len(candidates)} simulation_stats node(s): {candidates}")
        print("  [read_sim_stats] slot_as_array returned None or non-dict for all of them.")
        print("  [read_sim_stats] _exanb_data loaded: "
              f"{'pyexanbody._exanb_data' in sys.modules!r}")

    return result


def particles(graph, node: str = None) -> dict:
    """Return per-particle fields copied by an ``extract_particle_fields`` operator.

    The operator must be part of the executed graph, e.g. in the .msp file::

        dump_particles:
          - extract_particle_fields: { fields: [ rx, ry, rz, vx, vy, vz ] }

    Every field is a numpy array in its native dtype, all in the same particle
    order: scalars have shape ``(N,)``, Vec3d ``(N,3)``, Mat3d ``(N,3,3)`` and
    Quaternion ``(N,4)`` (w, x, y, z).  ``id`` is included when the grid has it.

    Args:
        graph: an ``ApplicationContext`` (its ``graph``) or the ``OperatorNode``
               returned by ``build_simulation_graph()``.
        node:  pathname of the operator to read from; by default the last
               ``extract_particle_fields`` node, in graph order, that ran.

    Returns:
        dict ``{field name: numpy array}``, or ``None`` if no such node ran.
    """
    if isinstance(graph, ApplicationContext):
        graph = graph.graph
    result = None

    def _collect(op):
        nonlocal result
        if node is not None and op.pathname() != node:
            return
        if node is None and op.name() != "extract_particle_fields":
            return
        val = op.slot_as_array("particles")
        if isinstance(val, dict):
            result = val  # keep the last one in graph order

    graph.apply_graph(_collect)
    return result
