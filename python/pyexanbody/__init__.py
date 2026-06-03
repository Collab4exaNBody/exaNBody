"""
pyexanbody — Python interface to the exaNBody N-body simulation framework.

This package is a thin wrapper around pyonika that ensures the exaNBody
plugins and configuration are loaded before the first pyonika.init() call.
It re-exports the entire pyonika API so user code only needs to import this
one module.

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
    xnb.end(ctx)
"""

import ctypes
import ctypes.util
import os
import sys

# ---------------------------------------------------------------------------
# 1. Resolve install paths (baked in by CMake, used as fallback when the
#    shell environment has not been set up via setup-env.sh).
# ---------------------------------------------------------------------------
from pyexanbody._paths import ONIKA_INSTALL_DIR, XNB_INSTALL_DIR

# Public constant: root of the installed exaNBody share tree.
XNB_SHARE_DIR = os.path.join(XNB_INSTALL_DIR, "share")


def _share_subdirs() -> list:
    """Return all first-level subdirectories of XNB_SHARE_DIR that exist."""
    if not os.path.isdir(XNB_SHARE_DIR):
        return []
    return [
        os.path.join(XNB_SHARE_DIR, d)
        for d in os.listdir(XNB_SHARE_DIR)
        if os.path.isdir(os.path.join(XNB_SHARE_DIR, d))
    ]


def _ensure_env() -> None:
    onika_lib     = os.path.join(ONIKA_INSTALL_DIR, "lib")
    xnb_lib       = os.path.join(XNB_INSTALL_DIR, "lib")
    onika_plugins = os.path.join(ONIKA_INSTALL_DIR, "plugins")
    xnb_plugins   = os.path.join(XNB_INSTALL_DIR, "plugins")
    xnb_config    = os.path.join(XNB_INSTALL_DIR, "data", "config")

    # sys.path — make pyonika importable
    if onika_lib not in sys.path:
        sys.path.insert(0, onika_lib)

    # LD_LIBRARY_PATH — ensures libexanbCore.so etc. are found when onika's
    # plugin loader calls dlopen() on the exaNBody plugin .so files.
    ld_parts = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    changed = False
    for p in (onika_lib, xnb_lib):
        if p not in ld_parts:
            ld_parts.insert(0, p)
            changed = True
    if changed:
        os.environ["LD_LIBRARY_PATH"] = ":".join(ld_parts)

    # ONIKA_CONFIG_PATH — point at exaNBody's config dir (which includes
    # main-config.msp with domain, compute-loop, etc.).
    if not os.environ.get("ONIKA_CONFIG_PATH"):
        os.environ["ONIKA_CONFIG_PATH"] = xnb_config

    # ONIKA_PLUGIN_PATH — onika built-ins + all exaNBody plugins.
    pp_parts = [p for p in os.environ.get("ONIKA_PLUGIN_PATH", "").split(":") if p]
    changed = False
    for p in (onika_plugins, xnb_plugins):
        if p not in pp_parts:
            pp_parts.append(p)
            changed = True
    if changed:
        os.environ["ONIKA_PLUGIN_PATH"] = ":".join(pp_parts)

    # ONIKA_DATA_PATH — add all share/ subdirs so that data files referenced
    # inside example .msp files (dump outputs, particle data, etc.) are found
    # without full paths.  ONIKA_DATA_PATH is colon-separated and supports
    # multiple directories, unlike ONIKA_CONFIG_PATH.
    dp_parts = [p for p in os.environ.get("ONIKA_DATA_PATH", "").split(":") if p]
    changed = False
    for p in _share_subdirs():
        if p not in dp_parts:
            dp_parts.append(p)
            changed = True
    if changed:
        os.environ["ONIKA_DATA_PATH"] = ":".join(dp_parts)


_ensure_env()

# ---------------------------------------------------------------------------
# 2. Import pyonika now that the environment is ready.
# ---------------------------------------------------------------------------
import pyonika  # noqa: E402

# ---------------------------------------------------------------------------
# 3. Pre-initialize MPI so onika treats it as externally owned.
#
# onika::app::initialize_mpi() calls MPI_Initialized() and sets
# external_mpi_init=true if MPI is already up. When that flag is set,
# onika::app::end() skips MPI_Finalize(), which lets the user call
# init()/run()/end() multiple times in the same Python session.
#
# The MPI library is already loaded as a transitive dependency of
# libonika.so, so we locate it via /proc/self/maps and call MPI_Init
# exactly once if MPI has not yet been initialized.
# ---------------------------------------------------------------------------

def _ensure_mpi_external() -> None:
    lib = None

    # Locate the loaded libmpi.so via the process memory map (Linux).
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                cols = line.split()
                if len(cols) >= 6:
                    path = cols[5]
                    if os.path.isfile(path) and "/libmpi" in os.path.basename(path):
                        try:
                            lib = ctypes.CDLL(path)
                            break
                        except OSError:
                            continue
    except OSError:
        pass

    # Fallback: search for MPI by standard library name.
    if lib is None:
        for candidate in ("mpi", "mpi_cxx", "openmpi", "mpich"):
            path = ctypes.util.find_library(candidate)
            if path:
                try:
                    lib = ctypes.CDLL(path)
                    break
                except OSError:
                    continue

    if lib is None:
        return

    try:
        initialized = ctypes.c_int(0)
        lib.MPI_Initialized(ctypes.byref(initialized))
        if not initialized.value:
            # Use MPI_Init_thread requesting MPI_THREAD_MULTIPLE (value=3 in the
            # MPI standard).  This matches what onika's initialize_mpi() would
            # request when it owns MPI init, so MPI_Query_thread() returns the
            # same level and onika does not emit the thread-support warning.
            provided = ctypes.c_int(0)
            lib.MPI_Init_thread(None, None, ctypes.c_int(3), ctypes.byref(provided))

            # We called MPI_Init_thread, so we own MPI finalisation.
            # onika's end() skips MPI_Finalize when external_mpi_init=true
            # (which is always the case here), so without this atexit handler
            # MPI_Finalize would never be called — causing mpirun to report an
            # improper termination when running with multiple processes.
            import atexit as _atexit
            _lib_ref = lib
            def _mpi_finalize():
                try:
                    _finalized = ctypes.c_int(0)
                    _lib_ref.MPI_Finalized(ctypes.byref(_finalized))
                    if not _finalized.value:
                        _lib_ref.MPI_Finalize()
                except Exception:
                    pass
            _atexit.register(_mpi_finalize)
    except AttributeError:
        pass


_ensure_mpi_external()

# ---------------------------------------------------------------------------
# 4. Load exaNBody-specific slot extractors.
#
# _exanb_data is a pybind11 extension that calls register_slot_extractor()
# (a symbol in pyonika.so) at import time.  Python loads extensions with
# RTLD_LOCAL by default, which hides their symbols from other extensions.
# We therefore re-open pyonika with RTLD_GLOBAL first so that its symbols
# (including register_slot_extractor) are visible when _exanb_data.so is
# loaded by dlopen.
# ---------------------------------------------------------------------------

try:
    from pyexanbody import _exanb_data  # noqa: F401 (imported for side effects)
except Exception as _e:
    import warnings as _w
    _w.warn(f"pyexanbody: _exanb_data not loaded ({_e}); "
            "SimulationStatistics and GridCellValues will not be accessible via slot_as_array. "
            "read_sim_stats() and read_cell_values() will return empty results.")
    del _w, _e

# ---------------------------------------------------------------------------
# 5. Re-export the full public API so callers can use pyexanbody as a drop-in.
# ---------------------------------------------------------------------------
from pyonika import (  # noqa: F401
    init,
    run,
    run_node,
    end,
    make_operator,
    available_operators,
    set_operator_defaults,
    build_simulation_graph,
    slot_as_array,
    ApplicationContext,
    OperatorNode,
    OperatorSlotBase,
    OnikaError,
)

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


def run_file(msp_path: str, argv0: str = "pyexanbody", **extra_args) -> int:
    """Run a complete simulation from an .msp file and return 0 on success.

    Equivalent to: init → run → end, with early-exit handling.

    Args:
        msp_path:   Path to the .msp configuration file.
        argv0:      Program name shown in onika usage messages.
        **extra_args: onika configuration overrides passed as ``--key value``
                    pairs.  Keys land under ``configuration:`` only; use ``_``
                    to keep flat (e.g. ``omp_num_threads=4``) or ``-`` in the
                    key name to produce nested YAML.  Simulation parameters
                    (global, input_data, …) cannot be set this way — use
                    ``set_operator_defaults()`` after ``init()`` instead.

    Returns:
        0 on normal completion, or the early-exit code from onika.
    """
    argv = [argv0, os.path.abspath(msp_path)]
    for k, v in extra_args.items():
        # onika cmdline parser expects space-separated "--key value" pairs.
        # Keys go under configuration: only; use "_" to keep flat, "-" to nest.
        argv.append(f"--{k}")
        argv.append(str(v))
    ctx = pyonika.init(argv)
    if ctx.error_code >= 0:
        return ctx.error_code
    pyonika.run(ctx)
    pyonika.end(ctx)
    return 0


def read_cell_values(graph, field_name: str):
    """Return a zero-copy numpy array for a named grid cell field, ghost-stripped.

    Searches the simulation graph for any operator with a ``grid_cell_values``
    slot that contains the requested field and returns a strided numpy view of
    the inner (non-ghost) cells.  No data is copied — keep the owning
    ``ApplicationContext`` alive for as long as the returned array is in use.

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

    Requires Tier 1 of the data-access plan (``_exanb_data`` extension built
    and ``SimulationStatistics`` extractor registered).

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
        if result:
            return
        val = node.slot_as_array("simulation_stats")
        if _debug:
            print(f"  [read_sim_stats] node={node.pathname()!r}  "
                  f"slot_as_array → type={type(val).__name__}  value={val!r}")
        if isinstance(val, dict):
            result.update(val)

    graph.apply_graph(_collect)

    if _debug and not result:
        print(f"  [read_sim_stats] found {len(candidates)} simulation_stats node(s): {candidates}")
        print("  [read_sim_stats] slot_as_array returned None or non-dict for all of them.")
        print("  [read_sim_stats] Verify _exanb_data loaded: "
              f"{'_exanb_data' in dir()!r} — check for warnings at import time.")

    return result
