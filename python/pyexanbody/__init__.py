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
            lib.MPI_Init(None, None)
    except AttributeError:
        pass


_ensure_mpi_external()

# ---------------------------------------------------------------------------
# 4. Re-export the full public API so callers can use pyexanbody as a drop-in.
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
# 5. exaNBody-level convenience helpers.
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
        **extra_args: onika config overrides (use ``__`` as the ``.`` separator).

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
        **extra_args: onika config overrides (use ``__`` as the ``.`` separator,
                    e.g. ``configuration__omp_num_threads=4``).

    Returns:
        0 on normal completion, or the early-exit code from onika.
    """
    argv = [argv0, os.path.abspath(msp_path)]
    for k, v in extra_args.items():
        argv.append(f"--{k.replace('__', '.')}={v}")
    ctx = pyonika.init(argv)
    if ctx.error_code >= 0:
        return ctx.error_code
    pyonika.run(ctx)
    pyonika.end(ctx)
    return 0
