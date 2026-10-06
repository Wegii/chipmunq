from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import pickle
import queue
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

sys.path.append(os.path.join(os.getcwd(), "."))

from experiments.exp_utils.circuit_generator import get_tqec_cnot_rotated, generate_gross_code
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.utils import *
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit

# GHZ / Bernstein-Vazirani lattice-surgery generators
from experiments.exp_utils.ghz_circuit_generator import get_tqec_ghz
from experiments.exp_utils.bv_circuit_generator import get_tqec_bv
from experiments.exp_utils.gross_bridge_circuit_generator import get_gross_bridge
from experiments.exp_utils.bv_color_code_circuit_generator import get_color_code_bv, chiplet_long_range_offsets
from experiments.exp_utils.qft_color_code_circuit_generator import get_color_code_qft

# MECH
sys.path.append(os.path.join(os.getcwd(), "./external/baseline/MECH"))
from external.baseline.MECH.Circuit import *
from external.baseline.MECH.Chiplet import *
from external.baseline.MECH.HighwayOccupancy import *
from external.baseline.MECH.Router import *
from external.baseline.MECH.MECHBenchmarks import *
from external.baseline.MECH.transpile_mech import transpile_circuit_MECH
import networkx as nx
from networkx.classes import Graph
from experiments.related_work_exps.utils import calc_circuit_mech_stats, generate_qecc_synth_backend_from_mech, generate_simple_backend

# QECC-Synth
sys.path.append(os.path.join(os.getcwd(), "./external/baseline/QECC_Synth/SurfStitch/MyCode/src"))
from external.baseline.QECC_Synth.SurfStitch.MyCode.src.transpile_qeccsynth import transpile_circuit_QECCSynth

# OLSQ2, SEQC, Murali et al.: same entry points as the related-work comparison. Every method (incl.
# Chipmunq, LightSABRE and MECH) runs in its own process group under the same wall-clock limit.
from experiments.exp_utils.transpilation_utils import (
    EXTRA_METHODS,
    FAILED,
    METHOD_STYLES,
    STARTUP_TIMEOUT_S,
    TIMEOUT,
    TIMEOUT_S,
    _kill,
    custom_partitioned_transpilation,
    run_with_timeout,
    sabre_transpilation,
)

# The `from ... import *` lines above pull numpy's sum/any/all/min/max/... into this module and shadow the
# Python builtins (np.any(generator) is always True, np.sum(generator) is deprecated). Restore the builtins.
from builtins import abs, all, any, max, min, round, sum

# Plotting
import matplotlib.pyplot as plt
import numpy as np


def _make_tqec_database_process_safe() -> None:
    """tqec caches detectors in ONE pickle shared by all processes (~/.local/share/TQEC/detector_database.pkl
    or $TQEC_DETECTOR_DATABASE_PATH) and rewrites it in place. With parallel generation jobs, one worker reads
    the file while another is writing it ("pickle data was truncated"), tqec then moves the "faulty" file away
    and the next worker's rename fails. Patch its reader/writer so that

      * writes go to a temp file that atomically replaces the database, so readers see the old or new file,
        never half a file;
      * writers hold an exclusive file lock and first merge entries other workers saved in the meantime, so
        no worker throws away another one's detectors (it is only a cache, so losing entries would just
        cost recomputation, not correctness);
      * a read that races with tqec's own "move faulty database" returns an empty database, not a crash.

    Runs at import, so every spawned worker (which re-imports this module) is patched too.
    """
    try:
        from tqec.compile.detectors.database import DetectorDatabase
    except Exception:
        return
    if getattr(DetectorDatabase, "_chipmunq_process_safe", False):
        return
    try:
        import fcntl
    except ImportError:  # Windows: atomic replace still prevents truncated reads
        fcntl = None

    def atomic_writer(original, merge: bool):
        def write(filepath, database):
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath.with_name(filepath.name + ".lock"), "a") as lock:
                if fcntl:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                if merge and filepath.exists():
                    try:
                        with open(filepath, "rb") as f:
                            other = pickle.load(f)
                        mine, theirs = getattr(database, "mapping", None), getattr(other, "mapping", None)
                        if (isinstance(mine, dict) and isinstance(theirs, dict)
                                and getattr(other, "version", None) == getattr(database, "version", None)):
                            for k, v in theirs.items():
                                mine.setdefault(k, v)
                    except Exception:
                        pass  # unreadable old file: just overwrite it
                fd, tmp = tempfile.mkstemp(dir=filepath.parent, prefix=filepath.name + ".", suffix=".tmp")
                os.close(fd)
                try:
                    original(Path(tmp), database)
                    os.replace(tmp, filepath)
                finally:
                    if os.path.exists(tmp):
                        os.remove(tmp)
        return write

    def tolerant_reader(original):
        def read(filepath):
            try:
                return original(filepath)
            except FileNotFoundError:  # another worker moved/replaced the file while we read it
                return DetectorDatabase()
        return read

    for fmt, writer in list(DetectorDatabase._WRITERS.items()):
        DetectorDatabase._WRITERS[fmt] = atomic_writer(writer, merge=(fmt == "pickle"))
    for fmt, reader in list(DetectorDatabase._READERS.items()):
        DetectorDatabase._READERS[fmt] = tolerant_reader(reader)
    DetectorDatabase._chipmunq_process_safe = True


_make_tqec_database_process_safe()


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

# Code distances to evaluate, as tqec scales k (d = 2k + 1). The CNOT generator has hand-written
# partitions for k = 1, 2, 3, 7 only (d = 3, 5, 7, 15), and Chipmunq's mapper places rotated
# surface-code patches for d = 3, 5, 7, 9, 15.
DISTANCE_SCALES = [1, 2, 3]#[1, 2, 3]#[2, 3, 4, 5]  # d = 5, 9, 11
NUM_INTER_CHIPLET_CONNECTIONS = 8
PS_INTER = 1e-4

# Benchmarks whose code distance does not depend on k (fixed-distance codes): run once, drawn as one bar
DISTANCE_INDEPENDENT = {"gross_bridge": 12, "gross": 12}


def distance(ks: int) -> int:
    return 2 * ks + 1


def chiplet_shape(ks: int) -> tuple[int, int]:
    """Physical rows x cols of a chiplet holding one patch + pipe strip: (2d + 5) x (d + 3) (15 x 8 at d = 5)."""
    d = distance(ks)
    return (2 * d + 5, d + 3)


def n_inter(ks: int) -> int:
    """Inter-chiplet links per chiplet edge. BackendChipletV2 places them around the edge centres
    (rows centre +- (n - 1), cols centre - n/2 .. centre + n/2 - 1), so n <= d + 3 (= 8 at d = 5)."""
    return min(NUM_INTER_CHIPLET_CONNECTIONS, distance(ks) + 3)

GHZ_N = 12  # same number of logical qubits as the old 6x CNOT (6 x 2)
QFT_N = 48  # logical qubits of the Clifford QFT (all-to-all: QFT_N (QFT_N - 1) transversal CNOTs in 2 QFT_N - 3 parallel steps)
BV_SECRET = "110100111010110100111010110100111010110100111010110100111010110100111010110100111010110100111010"  # 12-bit secret with mixed 0/1 so the bus is partially trimmed
GROSS_BRIDGE_N_INTER = 6  # links between neighbouring chiplets of the Gross-bridge backend (max 6)

# Wall-clock limit per (benchmark, distance, method) for EVERY method (Chipmunq, LightSABRE, MECH, OLSQ2,
# SEQC, Murali et al.). Process start-up is not counted; a run exceeding the limit is killed (with all its
# child processes) and reported as T/O. Set in transpilation_utils (TIMEOUT_S = 50 s).
METHOD_TIMEOUT_S = TIMEOUT_S
# Longer limits for individual benchmarks (overrides METHOD_TIMEOUT_S / --timeout for these keys): the 48-qubit
# Clifford QFT is ~10-100x larger than the other circuits.
BENCHMARK_TIMEOUT_S = {"qft_color": TIMEOUT_S}

RESULTS_DIR = Path("experiments/evaluation/scalability")
PLOT_PREFIX = RESULTS_DIR / "cnot_scaling_overhead_split"

# Order = order of the bar groups in the plot.
BENCHMARKS = ["cnot", "qft_color", "bv", "bv_color", "gross_bridge"]  # QFT (color) replaces GHZ
# Short names: used as x tick labels (two lines where needed) and in the console output.
# Logical-qubit counts (QFT_N, len(BV_SECRET)) go in the caption. "ghz" stays generatable but is not plotted.
TITLES = {
    "cnot": "CNOT",
    "ghz": "GHZ",
    "bv": "BV",
    "bv_color": "BV\n(color)",
    "qft_color": "QFT\n(color)",
    "gross_bridge": "Gross\nsurgery",
    "gross": "Gross",  # memory only; no longer in BENCHMARKS
}


def _title(key: str) -> str:
    """Single-line name for console output."""
    return TITLES[key].replace("\n", " ")

# --- Transpilation methods -------------------------------------------------------------
# "ideal" is the untranspiled circuit; everything else is a method that can be run.
# The extra baselines are the ones of the related-work comparison that transpilation_utils provides.
#EXTRA_TOOLS = tuple(k for k in ("mech", "olsq2", "seqc", "murali", ) if k in EXTRA_METHODS)
# ("seqc",) needs the trailing comma: ("seqc") is a plain string, which iterates over its characters and
# leaves EXTRA_TOOLS empty (SEQC would silently not run).
EXTRA_TOOLS = tuple(k for k in ("seqc", "murali", ) if k in EXTRA_METHODS)
RUN_TOOLS = ("custom", "sabre", *EXTRA_TOOLS)
ALL_TOOLS = ("ideal", *RUN_TOOLS)

# What the extra methods receive as their "cm" argument. The chiplet backend's coupling map matches
# the related-work comparison. Murali et al. is noise-adaptive, so it gets the full backend: that way
# it can see the inter-chiplet error rates (the noise_adaptive_transpilation wrapper accepts both).
# Set a method to "coupling_map" here if its wrapper in transpilation_utils only takes a CouplingMap.
EXTRA_INPUT = {"olsq2": "coupling_map", "seqc": "coupling_map", "murali": "backend"}

METRICS = (["depth_overall", "gate_overall"]
           + [f"{t}_{s}" for t in RUN_TOOLS for s in ("depth", "overhead", "all")])

# Keys used by the old version of this script (n_patches / 12 logical qubits) -> new names.
LEGACY_KEYS = {1: "cnot", 12: "gross"}


# --------------------------------------------------------------------------------------
# Backends and benchmarks
# --------------------------------------------------------------------------------------


def _chipmunq_backend(size: tuple[int, int, int, int], ks: int) -> BackendChipletV2:
    return BackendChipletV2(
        size=size,
        n_inter=n_inter(ks),
        connectivity="nn",
        topology="rotated_grid",
        inter_chiplet_noise=PS_INTER,
        inter_chiplet_amplification=1,
        inter_chiplet_noise_type="constant",
        num_defective_qubits=0,
    )


def _chiplet_grid_for(partitions: list[dict], ks: int, slack: int = 1) -> tuple[int, int, int, int]:
    """Square chiplet grid that covers the block footprint of a tqec layout (one chiplet per block)."""
    pos = [p["position"] for p in partitions if p["type"] == "rotated_surface_code"]
    w = max(x for x, _ in pos) + 1
    h = max(y for _, y in pos) + 1
    side = max(w, h) + slack
    return (side, side, *chiplet_shape(ks))


def generate_circuit(key: str, ks: int):
    """Return (stim circuit, partitions). This is the expensive part, done once per (benchmark, distance)."""
    if key == "cnot":
        return get_tqec_cnot_rotated(distance_scale=ks, n1=1, n2=0)
    if key == "ghz":
        return get_tqec_ghz(GHZ_N, distance_scale=ks)
    if key == "bv":
        return get_tqec_bv(BV_SECRET, distance_scale=ks)
    if key == "bv_color":
        # Same secret as "bv", but on flagged color-code patches with transversal H / CNOT (no lattice
        # surgery), d = 2ks + 1.
        return get_color_code_bv(BV_SECRET, distance=distance(ks))
    if key == "qft_color":
        # Clifford QFT (controlled-phase rotations replaced by S / S_DAG) on flagged color-code patches with
        # transversal H / S / CNOT, d = 2ks + 1. Keeps the QFT's all-to-all logical interaction structure.
        return get_color_code_qft(QFT_N, distance=distance(ks))
    if key == "gross_bridge":
        # Two [[144,12,12]] modules + bridge measuring Zbar_a Zbar_b (fixed d = 12, ks is not used).
        return get_gross_bridge()
    if key == "gross":
        return generate_gross_code(num_qubits=1)
    raise ValueError(f"Unknown benchmark '{key}'.")


def make_backend(key: str, ks: int, partitions: list[dict]):
    """Return (Chipmunq backend, MECH config) for a generated benchmark (cheap, rebuilt in every job)."""
    if key == "cnot":
        backend = _chipmunq_backend((2, 2, *chiplet_shape(ks)), ks)
        # MECH grid side 14 was chosen for d = 5; other distances use the automatic size
        return backend, {"mode": "run", "side": 14 if ks == 2 else None}

    if key in ("ghz", "bv"):
        return _chipmunq_backend(_chiplet_grid_for(partitions, ks), ks), {"mode": "run", "side": None}

    if key in ("bv_color", "qft_color"):
        # One chiplet per patch; the chiplets have the extra links under which a patch needs no SWAPs
        # (no wrap-around), so all routing overhead comes from the transversal CNOTs between patches.
        size = max(max(p["height"], p["width"]) for p in partitions)
        side = math.ceil(math.sqrt(len(partitions)))
        backend = BackendChipletV2(
            size=(side, side, size, size),
            n_inter=min(NUM_INTER_CHIPLET_CONNECTIONS, size // 2),  # links must fit on the chiplet edge
            connectivity="long_range",
            topology="grid",
            long_range_offsets=chiplet_long_range_offsets(flags=True),
            num_defective_qubits=0,
        )
        return backend, {"mode": "run", "side": None}

    if key == "gross_bridge":
        # One chiplet per module and one for the bridge. n_inter <= 6: with 12 rows per chiplet,
        # BackendChipletV2 places the horizontal links at +-(n_inter - 1) rows around the centre.
        backend = BackendChipletV2(
            size=(1, 3, 12, 24),
            n_inter=GROSS_BRIDGE_N_INTER,
            connectivity="torus",
            topology="grid",
            long_range_offsets=[(1, 0), (2, 0), (3, 0), (0, 1), (0, 2), (0, 3)],
            num_defective_qubits=0,
        )
        # MECH is not run on BB codes (it timed out on the Gross memory), same as before.
        return backend, {"mode": "timeout"}

    if key == "gross":
        backend = BackendChipletV2(
            # 1 chiplet with 288 qubits, long-range connections, grid layout
            size=(1, 1, 12, 24),
            n_inter=1,
            connectivity="torus",
            topology="grid",
            long_range_offsets=[(1, 0), (2, 0), (3, 0), (0, 1), (0, 2), (0, 3)],
            num_defective_qubits=0,
        )
        return backend, {"mode": "timeout"}

    raise ValueError(f"Unknown benchmark '{key}'.")


def build_benchmark(key: str, ks: int):
    """Return (stim circuit, partitions, Chipmunq backend, MECH config)."""
    circuit, partitions = generate_circuit(key, ks)
    return (circuit, partitions, *make_backend(key, ks, partitions))


# --------------------------------------------------------------------------------------
# Generic timed runner (Chipmunq, LightSABRE, OLSQ2, SEQC, Murali et al.)
# --------------------------------------------------------------------------------------


def run_timed(fn, *args, timeout: float, **kwargs) -> tuple[object, dict]:
    """Run ``fn(*args, **kwargs)`` in a fresh process group that is killed after ``timeout`` seconds.

    Returns (circuit or None, status entry). The entry has "status" ok / timeout / error; the clock
    (and "runtime_s") covers only the call itself, not process start-up.
    """
    circuit, runtime, status = run_with_timeout(fn, *args, timeout=timeout, **kwargs)
    if status == "ok" and circuit is not None:
        return circuit, {"status": "ok", "runtime_s": runtime}
    if status == TIMEOUT:
        return None, {"status": "timeout", "timeout_s": timeout, "elapsed_s": runtime}
    return None, {"status": "error", "error": str(status), "elapsed_s": runtime}


def run_extra_method(method: str, qc, backend, timeout: float) -> tuple[object, dict]:
    """OLSQ2 / SEQC / Murali et al. with the coupling map (or the full backend, see EXTRA_INPUT)."""
    target = backend if EXTRA_INPUT.get(method, "coupling_map") == "backend" else backend.coupling_map
    print(f"  {method}: input = {'backend' if target is backend else 'coupling map'}, limit {timeout:.0f} s")
    return run_timed(EXTRA_METHODS[method], qc, target, timeout=timeout)


# --------------------------------------------------------------------------------------
# MECH with a wall-clock timeout
# --------------------------------------------------------------------------------------


# Keyword names under which MECH wrappers commonly accept their own time limit. If
# transpile_circuit_MECH takes one of these, our timeout is forwarded so the wrapper's
# internal default (e.g. 60 s) does not cut the run short.
_MECH_TIMEOUT_KWARGS = ("timeout", "time_limit", "timeout_s", "timeout_sec", "max_time", "time_out")


def _mech_timeout_kwargs(timeout_s: float) -> dict:
    import inspect
    try:
        params = inspect.signature(transpile_circuit_MECH).parameters
    except (TypeError, ValueError):
        return {}
    return {k: timeout_s for k in _MECH_TIMEOUT_KWARGS if k in params}


def _mech_worker(qc, side: int, timeout_s: float, out) -> None:
    import traceback
    # Own process group, so a timeout also kills processes started by the MECH wrapper
    try:
        os.setsid()
    except (AttributeError, OSError):
        pass
    out.put(("started", None))  # spawn, imports and argument unpickling are done: the clock starts now
    t0 = time.time()
    try:
        backend, _, _ = generate_simple_backend(side, side)
        circ = transpile_circuit_MECH(qc, backend, **_mech_timeout_kwargs(timeout_s))
        if circ is None:  # some wrappers return None when their internal limit is hit
            out.put(("error", f"transpile_circuit_MECH returned None after {time.time() - t0:.0f} s"))
            return
        out.put(("ok", calc_circuit_mech_stats(circ, qc)))
    except BaseException as e:  # incl. TimeoutError / KeyboardInterrupt raised by MECH's own limits
        out.put(("error", f"{e!r} after {time.time() - t0:.0f} s\n{traceback.format_exc()}"))


def run_mech_with_timeout(qc, side: int, timeout_s: float) -> dict:
    """Run MECH in a child process. Returns the stats dict with a "status" of ok / timeout / error.

    Same semantics as run_with_timeout: the clock starts once the child has started up, and on timeout
    the child's whole process group is killed.

    * "spawn", not "fork": the parent has already run Qiskit's multi-threaded (Rust) SABRE, and
      forking a process that owns a thread pool can deadlock or crash the child.
    * not a daemon: daemonic processes may not start children, which MECH wrappers that enforce
      their own time limit via multiprocessing need.
    """
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    proc = ctx.Process(target=_mech_worker, args=(qc, side, timeout_s, out), daemon=False)
    proc.start()
    fwd = _mech_timeout_kwargs(timeout_s)
    print(f"  MECH on {side}x{side} grid, limit {timeout_s:.0f} s"
          + (f" (forwarded to wrapper as {list(fwd)})" if fwd else ""))

    try:
        # --- Start-up (not counted towards the limit) ---
        t_start = time.time()
        while True:
            try:
                msg = out.get(timeout=1)
                break
            except queue.Empty:
                if not proc.is_alive():
                    return {"status": "error", "elapsed_s": 0.0,
                            "error": f"MECH process exited with code {proc.exitcode} during start-up"}
                if time.time() - t_start > STARTUP_TIMEOUT_S:
                    return {"status": "error", "elapsed_s": 0.0,
                            "error": f"MECH process did not start within {STARTUP_TIMEOUT_S:.0f} s"}
        if msg[0] != "started":  # should not happen, but do not lose a result/error
            status, payload = msg
        else:
            # --- The run itself ---
            t0 = time.time()
            while True:
                try:
                    status, payload = out.get(timeout=1)
                    break
                except queue.Empty:
                    if not proc.is_alive():
                        return {"status": "error", "elapsed_s": time.time() - t0,
                                "error": f"MECH process exited with code {proc.exitcode} "
                                         f"after {time.time() - t0:.0f} s"}
                    if time.time() - t0 > timeout_s:
                        print(f"  MECH killed after {timeout_s:.0f} s (timeout)")
                        return {"status": "timeout", "timeout_s": timeout_s, "elapsed_s": timeout_s}
            runtime = time.time() - t0
    finally:
        _kill(proc)  # no-op if it already finished; otherwise kills MECH and all its children
        out.close()
        out.cancel_join_thread()

    if status == "error":
        return {"status": "error", "error": payload, "elapsed_s": runtime}
    result = dict(payload)
    result.update(status="ok", side=side, runtime_s=runtime)
    return result


def _status_label(entry) -> str | None:
    """Bar annotation for a result entry: None if it ran, T/O on timeout, N/A on error."""
    if isinstance(entry, dict) and "status" in entry:
        if entry["status"] == "error" and "timeout" in str(entry.get("error", "")).lower():
            return "T/O"  # the method's own time limit fired
        return {"ok": None, "timeout": "T/O", "error": "N/A"}.get(entry["status"], "N/A")
    if entry == -1:  # legacy pickles
        return "T/O"
    return None


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------

PASTEL_BLUE = "#A7D9ED"
PASTEL_ORANGE = "#F7C6A2"
PASTEL_GREEN = "#B5D8B0"

# (label, colour, hatch) per tool. The extra baselines take their style from METHOD_STYLES so they look
# the same as in the related-work figures; the fallbacks are only used if a key is missing there.
TOOL_STYLE = {
    "ideal": ("Ideal", "lightcoral", "//"),
    "custom": ("Chipmunq", PASTEL_BLUE, "/"),
    "sabre": ("LightSABRE", PASTEL_ORANGE, "o"),
    "mech": ("MECH", PASTEL_GREEN, "//"),
    "olsq2": ("OLSQ2", "#D7BDE2", "\\\\"),
    "seqc": ("SEQC", "#F9E79F", ".."),
    "murali": ("Murali et al.", "#D5D8DC", "xx"),
}
for _key, _label, _color, _hatch in METHOD_STYLES:
    if _key in EXTRA_TOOLS:
        TOOL_STYLE[_key] = (_label, _color, _hatch)

DARKEST = 0.55  # lightness reduction of the largest distance (0 = base colour)
TIMEOUT_BAR_FACTOR = 1.3  # a timed-out run is drawn as a hollow bar 30 % above the ideal



# --------------------------------------------------------------------------------------
# Paper panels b) - d): drawn at their printed size for four panels side by side across the full text width
# (Fig. 10 style; include at natural width, no scaling). Titles are centred on the plot rectangle.
# --------------------------------------------------------------------------------------
TEXT_WIDTH_IN = 7.0                   # full text width of the paper (two-column IEEE/ACM: ~7.0 in)
PANEL_W = TEXT_WIDTH_IN / 4           # 1.75 in
PANEL_H = 1.6
FONT_PT = 7
PANEL_MARGINS = dict(left=0.27, right=0.97, top=0.80, bottom=0.24)  # identical for all panels -> axes line up
PANEL_TICKS = {"cnot": "CNOT", "ghz": "GHZ", "qft_color": "QFT\ncolor", "bv": "BV", "bv_color": "BV\ncolor",
               "gross_bridge": "Gross", "gross": "Gross"}


def _panel_fonts() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": FONT_PT,
        "axes.labelsize": FONT_PT,
        "axes.titlesize": FONT_PT,
        "legend.fontsize": FONT_PT,
        "xtick.labelsize": FONT_PT - 1,
        "ytick.labelsize": FONT_PT - 1,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "ytick.minor.size": 1.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.labelpad": 2,
        "axes.linewidth": 0.6,
        "hatch.linewidth": 0.4,
        "lines.linewidth": 1.0,
        "errorbar.capsize": 1.5,
    })


def _panel_title(ax, text: str, better: str = "Lower is better ↓") -> None:
    """Panel title and the "better" hint, centred on the plot rectangle (axes)."""
    ax.text(0.5, 1.03, text, transform=ax.transAxes, fontweight="bold", ha="center", va="bottom")
    ax.text(0.5, 1.16, better, transform=ax.transAxes, fontweight="bold", color=plot_lib_color,
            ha="center", va="bottom")


def _shade(color, level: float):
    """Base colour darkened by ``level`` in [0, 1] (0 = base colour, 1 = DARKEST)."""
    import colorsys
    from matplotlib.colors import to_rgb
    h, l, s = colorsys.rgb_to_hls(*to_rgb(color))
    return colorsys.hls_to_rgb(h, l * (1 - DARKEST * level), s)


def _levels(ks_list: list[int]) -> dict[int, float]:
    """Darkness level per distance: smallest d = base colour, largest d = darkest."""
    if len(ks_list) == 1:
        return {ks_list[0]: 0.0}
    return {ks: i / (len(ks_list) - 1) for i, ks in enumerate(sorted(ks_list))}


def _tex_fonts(label_scale: float) -> dict:
    return {
        "font.family": "serif",
        "axes.labelsize": FONTSIZE * label_scale,
        "font.size": FONTSIZE * 1.2,
        "legend.fontsize": (FONTSIZE - 2) * 1.3,
        "xtick.labelsize": (FONTSIZE - 1) * 1,
        "ytick.labelsize": (FONTSIZE - 1) * 1.3,
        "axes.titlesize": 10,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "lines.markeredgewidth": 1.5,
        "lines.markeredgecolor": "black",
        "errorbar.capsize": 3,
    }


def _nice(v: float, up: bool = True) -> float:
    if v <= 0:
        return 0
    step = 5 * 10 ** (math.floor(math.log10(v)) - 1)
    return (math.ceil if up else math.floor)(v / step) * step


def _draw_bars(fig, data, keys, ks_list, labels, tools, ylim=None):
    """Grouped bars (one per tool in ``tools``), one overlaid segment per code distance.

    ``data[tool][i]`` maps ks -> value for benchmark ``keys[i]``; ``labels[tool][i]`` maps ks -> "T/O" /
    "N/A" for runs without a result. For every tool the bars of all distances share one x position; the
    largest distance is drawn first (darkest, at the back) and smaller distances in front of it, so each
    segment shows the value of one distance "stacked" on the previous one.

    Runs without a result are drawn as white, hatched, dashed bars without any text: a timeout
    TIMEOUT_BAR_FACTOR above the ideal, N/A (never run / error) just below the ideal.
    """
    x = np.arange(len(keys))
    width = 0.8 / len(tools)
    ax = fig.add_subplot(111)
    levels = _levels(ks_list)
    # Log scale: axis limits snap to full decades around the positive values
    allv = [v for t in tools for per_key in data[t] for v in per_key.values() if v > 0]
    ymin = ylim[0] if ylim else 10 ** math.floor(math.log10(min(allv)))
    ymax = ylim[1] if ylim else 10 ** math.ceil(math.log10(1.3 * max(allv)))

    for j, tool in enumerate(tools):
        label, color, hatch = TOOL_STYLE[tool]
        xpos = x + (j - (len(tools) - 1) / 2) * width
        for i, key in enumerate(keys):
            per_ks = data[tool][i]
            no_result = labels[tool][i]
            for rank, ks in enumerate(sorted(per_ks, reverse=True)):  # largest d first (back)
                lvl = 1.0 if key in DISTANCE_INDEPENDENT else levels[ks]
                bar = ax.bar(xpos[i], per_ks[ks], width, color=_shade(color, lvl), hatch=hatch,
                             edgecolor="black", linewidth=0.4, zorder=2 + rank)[0]
                if ks in no_result:  # no result: white, hatched, dashed outline
                    bar.set_facecolor("white")
                    bar.set_linestyle("--")
                    bar.set_hatch("xxx")
                    bar.set_linewidth(0.8)
                    bar.set_edgecolor(_shade(color, lvl))

    ax.set_yscale("log", nonpositive="clip")
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x)
    # Short labels: at 1.75 in each benchmark group gets ~0.24 in (full names in the caption)
    ax.set_xticklabels([PANEL_TICKS.get(k, TITLES[k]) for k in keys], fontsize=FONT_PT - 1.5, linespacing=0.95)
    ax.tick_params(axis="x", which="both", top=False)
    ax.tick_params(axis="y", length=2.5)
    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Legend handles: tools (medium shade) and distances (grey ramp)
    from matplotlib.patches import Patch
    tool_handles = [Patch(facecolor=_shade(TOOL_STYLE[t][1], 0.5), hatch=TOOL_STYLE[t][2], edgecolor="black",
                          label=TOOL_STYLE[t][0]) for t in tools]
    dist_handles = [Patch(facecolor=_shade("#d9d9d9", levels[ks]), edgecolor="black", label=f"d = {distance(ks)}")
                    for ks in sorted(ks_list)]
    return ax, tool_handles + dist_handles


def _fit_vertical(fig, ax, pad_px: float = 3.0, max_iter: int = 10) -> None:
    """Use the full figure height: move the top and bottom margins so that everything drawn with the axes
    (title and "Lower is better" above, tick labels and x label below) ends ``pad_px`` from the
    figure edge. Iterates because the title sits in axes coordinates and moves with the axes."""
    for _ in range(max_iter):
        fig.canvas.draw()
        bb = ax.get_tightbbox(fig.canvas.get_renderer())
        h = fig.bbox.height
        d_top = (h - pad_px - bb.y1) / h  # > 0: free space above -> raise the axes top
        d_bot = (bb.y0 - pad_px) / h      # > 0: free space below -> lower the axes bottom
        if abs(d_top) * h < 0.5 and abs(d_bot) * h < 0.5:
            return
        p = fig.subplotpars
        top, bottom = p.top + d_top, p.bottom - d_bot
        if top - bottom < 0.1:  # never collapse the axes
            return
        fig.subplots_adjust(top=top, bottom=bottom)


def _pct(v: float, ideal: float) -> str:
    return f"{100 * (v - ideal) / ideal:.0f}%"


def _ks_for(key: str, ks_list: list[int], results: dict) -> list[int]:
    """Distances plotted for a benchmark (one entry for distance-independent benchmarks)."""
    if key in DISTANCE_INDEPENDENT:
        have = [ks for ks in results["depth_overall"].get(key, {})]
        return have[:1]
    return list(ks_list)


def _no_result_label(results: dict, tool: str, key: str, ks: int) -> str | None:
    """None if ``tool`` has a result for (key, ks); otherwise "T/O" or "N/A"."""
    if tool == "ideal":
        return None
    if not _has(results, key, ks, tool):
        return "N/A"  # never run
    return _status_label(results.get(f"{tool}_all", {}).get(key, {}).get(ks))


def plot_combined_split(results: dict, keys: list[str], ks_list: list[int], filename: str,
                        tools=ALL_TOOLS, ylim_depth=None, ylim_gates=None) -> None:
    """Bar plots over several code distances for the methods in ``tools`` (default: all).
    ylim_* = (lo, hi) to override the automatic y range."""
    tools = [t for t in ALL_TOOLS if t in tools]  # canonical order

    # No-result labels first: they decide the height of the hollow bars below
    labels = {t: [] for t in tools}
    for key in keys:
        for t in tools:
            labs = {}
            for ks in _ks_for(key, ks_list, results):
                lab = _no_result_label(results, t, key, ks)
                if lab:
                    labs[ks] = lab
            labels[t].append(labs)

    def _collect(overall: str, suffix: str) -> dict:
        out = {t: [] for t in tools}
        for i, key in enumerate(keys):
            ks_here = _ks_for(key, ks_list, results)
            ideal = {ks: results[overall][key][ks] for ks in ks_here}
            for t in tools:
                if t == "ideal":
                    out[t].append(ideal)
                    continue
                vals = {}
                for ks in ks_here:
                    if labels[t][i].get(ks) == "T/O":
                        vals[ks] = TIMEOUT_BAR_FACTOR * ideal[ks]  # hollow bar 30 % above the ideal
                    else:  # -1 = no result (N/A: hollow bar just below the ideal)
                        vals[ks] = ideal[ks] + results[f"{t}_{suffix}"].get(key, {}).get(ks, -1)
                out[t].append(vals)
        return out

    depth = _collect("depth_overall", "depth")
    gates = _collect("gate_overall", "overhead")

    # Summary
    for i, key in enumerate(keys):
        for ks in sorted(depth[tools[0]][i]):
            d = DISTANCE_INDEPENDENT.get(key, distance(ks))
            di = results["depth_overall"][key][ks]
            gi = results["gate_overall"][key][ks]
            dep = [f"{TOOL_STYLE[t][0]}={labels[t][i].get(ks) or _pct(depth[t][i][ks], di)}"
                   for t in tools if t != "ideal"]
            gat = [f"{TOOL_STYLE[t][0]}={labels[t][i].get(ks) or _pct(gates[t][i][ks], gi)}"
                   for t in tools if t != "ideal"]
            print(f"[{_title(key)} d={d}] depth ideal={di}  " + "  ".join(dep)
                  + f"  | 2q ideal={gi}  " + "  ".join(gat))

    # ---------------- a) compilation time ----------------
    plot_runtime(results, keys, ks_list, f"{filename}_runtime.pdf", tools)

    # ---------------- b) depth ----------------
    _panel_fonts()
    fig = plt.figure(figsize=(PANEL_W, PANEL_H))
    ax, _ = _draw_bars(fig, depth, keys, ks_list, labels, tools, ylim_depth)
    ax.set_ylabel("Circuit depth")
    _panel_title(ax, "b) Circuit depth")
    fig.subplots_adjust(**PANEL_MARGINS)
    fig.savefig(f"{filename}_depth.pdf", format="pdf")
    plt.close(fig)

    # ---------------- c) 2q gates ----------------
    fig = plt.figure(figsize=(PANEL_W, PANEL_H))
    ax, handles = _draw_bars(fig, gates, keys, ks_list, labels, tools, ylim_gates)
    ax.set_ylabel("#2q gates")
    _panel_title(ax, "c) #2q gates")
    fig.subplots_adjust(**PANEL_MARGINS)
    fig.savefig(f"{filename}_overhead.pdf", format="pdf")
    plt.close(fig)

    # ---------------- d) relative overhead vs. the ideal circuit ----------------
    plot_relative_overhead(results, keys, ks_list, f"{filename}_relative.pdf", tools)

    # ---------------- legend ----------------
    # Two rows with the same number of entries: methods first, then code distances, filled row by row.
    # matplotlib fills legends column by column, so the two rows are interleaved; with an odd number of
    # entries the second row gets one invisible entry at the end.
    from matplotlib.patches import Patch
    ncols = math.ceil(len(handles) / 2)
    row1, row2 = handles[:ncols], handles[ncols:]
    row2 += [Patch(visible=False, label="") for _ in range(ncols - len(row2))]
    two_rows = [h for pair in zip(row1, row2) for h in pair]
    legend_fig = plt.figure(figsize=(TEXT_WIDTH_IN, 0.4))  # one legend for panels b) - d)
    legend_fig.legend(handles=two_rows, loc="center", frameon=False, ncols=ncols, columnspacing=1.0,
                      handlelength=1.2, handletextpad=0.35)
    legend_fig.savefig(f"{filename}legend.pdf", bbox_inches="tight", format="pdf")
    plt.close(legend_fig)



def plot_runtime(results: dict, keys: list[str], ks_list: list[int], filename: str, tools=ALL_TOOLS) -> None:
    """a) Compilation time of every method per benchmark and distance (same bar layout as b) and c)).
    Timed-out runs are hollow bars at the time limit; runs without a result (N/A) are left empty."""
    tools = [t for t in tools if t != "ideal"]  # the ideal circuit is not compiled
    data, labels = {t: [] for t in tools}, {t: [] for t in tools}
    for key in keys:
        for t in tools:
            vals, labs = {}, {}
            for ks in _ks_for(key, ks_list, results):
                entry = results.get(f"{t}_all", {}).get(key, {}).get(ks)
                lab = _no_result_label(results, t, key, ks)
                if lab == "T/O":
                    limit = entry.get("timeout_s") if isinstance(entry, dict) else None
                    vals[ks] = limit or BENCHMARK_TIMEOUT_S.get(key, METHOD_TIMEOUT_S)
                    labs[ks] = lab
                elif lab or not isinstance(entry, dict) or "runtime_s" not in entry:
                    vals[ks] = float("nan")  # no result (or a legacy entry without runtime): no bar
                    labs[ks] = lab or "N/A"
                else:
                    vals[ks] = entry["runtime_s"]
            data[t].append(vals)
            labels[t].append(labs)
    for i, key in enumerate(keys):
        for ks in _ks_for(key, ks_list, results):
            d = DISTANCE_INDEPENDENT.get(key, distance(ks))
            print(f"[{_title(key)} d={d}] runtime " + "  ".join(
                f"{TOOL_STYLE[t][0]}={labels[t][i].get(ks) or f'{data[t][i][ks]:.1f} s'}" for t in tools))
    _panel_fonts()
    fig = plt.figure(figsize=(PANEL_W, PANEL_H))
    ax, _ = _draw_bars(fig, data, keys, ks_list, labels, tools)
    ax.set_ylabel("Runtime [s]")
    _panel_title(ax, "a) Compilation time")
    fig.subplots_adjust(**PANEL_MARGINS)
    fig.savefig(filename, format="pdf")
    plt.close(fig)


def relative_overhead(results: dict, keys: list[str], ks_list: list[int], tool: str) -> dict[str, float]:
    """Mean relative increase over the ideal circuit, (compiled - ideal) / ideal, of depth and #2q gates,
    averaged over all benchmarks and distances where ``tool`` has a result (T/O and N/A runs are left out).
    Returns {"depth": ..., "gates": ..., "n": #runs}."""
    rel = {"depth": [], "gates": []}
    for key in keys:
        for ks in _ks_for(key, ks_list, results):
            if _no_result_label(results, tool, key, ks):
                continue
            for name, overall, suffix in (("depth", "depth_overall", "depth"), ("gates", "gate_overall", "overhead")):
                ideal = results[overall][key][ks]
                extra = results[f"{tool}_{suffix}"].get(key, {}).get(ks)
                if ideal and extra is not None and extra >= 0:
                    rel[name].append(extra / ideal)
    return {"depth": float(np.mean(rel["depth"])) if rel["depth"] else float("nan"),
            "gates": float(np.mean(rel["gates"])) if rel["gates"] else float("nan"),
            "n": len(rel["depth"])}


def plot_relative_overhead(results: dict, keys: list[str], ks_list: list[int], filename: str, tools) -> None:
    """d) Relative overhead of every method over the ideal circuit, in depth and #2q gates (one bar group per
    metric, one bar per method in the method colours of b) and c)). Averaged only over the benchmarks for which
    every shown method compiled every code distance, so all methods are compared on the same circuits."""
    tools = [t for t in tools if t != "ideal"]
    # Same circuits for every method: only benchmarks where every shown method has a result for every code
    # distance (a method that times out on the larger circuits would otherwise be averaged over easier ones)
    complete = [key for key in keys
                if all(_no_result_label(results, t, key, ks) is None
                       for t in tools for ks in _ks_for(key, ks_list, results))]
    left_out = [key for key in keys if key not in complete]
    print("[relative overhead] circuits: " + (", ".join(_title(k) for k in complete) or "none")
          + (f" (left out, not every method has every distance: {', '.join(_title(k) for k in left_out)})"
             if left_out else ""))
    stats = {t: relative_overhead(results, complete, ks_list, t) for t in tools}
    for t in tools:
        print(f"[relative overhead] {TOOL_STYLE[t][0]}: depth +{100 * stats[t]['depth']:.0f} %, "
              f"#2q gates +{100 * stats[t]['gates']:.0f} % (mean over {stats[t]['n']} runs)")
    _panel_fonts()
    fig = plt.figure(figsize=(PANEL_W, PANEL_H))
    ax = fig.add_subplot(111)
    x = np.arange(2)
    width = 0.8 / len(tools)
    vals = [100 * stats[t][m] for t in tools for m in ("depth", "gates") if stats[t][m] > 0]
    ymin = 10 ** math.floor(math.log10(min(vals))) if vals else 1
    ymax = 10 ** math.ceil(math.log10(1.3 * max(vals))) if vals else 100
    for j, t in enumerate(tools):
        label, color, hatch = TOOL_STYLE[t]
        for i, m in enumerate(("depth", "gates")):
            v = 100 * stats[t][m]
            xpos = x[i] + (j - (len(tools) - 1) / 2) * width
            if np.isfinite(v) and v > 0:
                ax.bar(xpos, v, width, color=color, hatch=hatch, edgecolor="black", linewidth=0.4,
                       zorder=2)  # pastel base colour of the method (b, c shade it per code distance)
            else:  # no result in any benchmark: hollow bar, as in b), c)
                ax.bar(xpos, ymin * 2, width, color="white", hatch="xxx", edgecolor=color,
                       linewidth=0.8, linestyle="--", zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x, ["Depth", "#2q gates"])
    ax.tick_params(axis="x", which="both", length=0)
    ax.tick_params(axis="y", length=2.5)
    ax.set_ylabel("Overhead vs. ideal [%]")
    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    _panel_title(ax, "d) Relative overhead")
    fig.subplots_adjust(**PANEL_MARGINS)
    fig.savefig(filename, format="pdf")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------------------


# Stim annotations carried through the Qiskit circuit (and barriers = Stim TICKs). They are not operations, but
# Qiskit's depth() counts them, and they roughly triple the depth of the input circuits. Methods that drop them
# (SEQC, MECH) would then look better than the ideal circuit, so depth counts real operations only.
NON_OPS = {"DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS", "TICK", "barrier"}


def circuit_depth(circuit) -> int:
    """Depth over real operations only (no Stim annotations, no barriers), identical for every method."""
    return circuit.depth(filter_function=lambda ins: ins.operation.name not in NON_OPS)


def num_2q_gates(circuit) -> int:
    """All 2-qubit operations (cx, cz, swap, ecr, ...), so every method's native output is counted."""
    return sum(1 for inst in circuit.data
               if inst.operation.num_qubits == 2 and inst.operation.name != "barrier")


# --- Stored circuits ----------------------------------------------------------------------
# Every compiled circuit (and the ideal one) is stored, so metrics can be recomputed later without rerunning
# the compilations (--recompute-metrics). MECH yields no Qiskit circuit; its saved result holds its own depth.
CIRCUIT_DIR = RESULTS_DIR / "circuits"
STORE_CIRCUITS = True  # --no-store-circuits to skip (the QFT circuits are large)


def circuit_path(key: str, ks: int, method: str) -> Path:
    """File of the stored circuit of (benchmark, distance, method); method "ideal" = untranspiled circuit."""
    return CIRCUIT_DIR / f"{key}_d{DISTANCE_INDEPENDENT.get(key, distance(ks))}_{method}.pkl.gz"


def save_circuit(path: Path, circuit) -> None:
    """gzip-compressed pickle, written to a temp file and moved into place (parallel workers, no half files)."""
    import gzip
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with gzip.open(tmp, "wb", compresslevel=3) as f:
        pickle.dump(circuit, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_circuit(path: Path):
    import gzip
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def recompute_metrics(results: dict, ks_list: list[int], methods=RUN_TOOLS) -> int:
    """Recompute depth and #2q-gate metrics of the ideal circuits and every method from the stored circuits
    (MECH: from the depth in its saved result). Returns the number of updated entries; results are saved."""
    updated = 0
    for key in BENCHMARKS:
        kss = ks_list[:1] if key in DISTANCE_INDEPENDENT else ks_list
        if key in DISTANCE_INDEPENDENT and results["depth_overall"].get(key):
            kss = list(results["depth_overall"][key])[:1]
        for ks in kss:
            ideal_path = circuit_path(key, ks, "ideal")
            if not ideal_path.exists():
                continue
            qc = load_circuit(ideal_path)
            depth0, gates0 = circuit_depth(qc), num_2q_gates(qc)
            results["depth_overall"].setdefault(key, {})[ks] = depth0
            results["gate_overall"].setdefault(key, {})[ks] = gates0
            updated += 1
            for m in methods:
                if m == "mech":
                    entry = results["mech_all"].get(key, {}).get(ks)
                    if isinstance(entry, dict) and entry.get("status") == "ok" and "depth" in entry:
                        results["mech_depth"].setdefault(key, {})[ks] = entry["depth"] - depth0
                        updated += 1
                    continue
                path = circuit_path(key, ks, m)
                if path.exists():
                    circ = load_circuit(path)
                    results[f"{m}_depth"].setdefault(key, {})[ks] = circuit_depth(circ) - depth0
                    results[f"{m}_overhead"].setdefault(key, {})[ks] = num_2q_gates(circ) - gates0
                    updated += 1
    _save_results(results)
    return updated


def _load_results() -> dict:
    results = {}
    for m in METRICS:
        path = RESULTS_DIR / f"{m}.pkl"
        data = {}
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
            for old, new in LEGACY_KEYS.items():  # migrate pickles from the old script
                if old in data and new not in data:
                    data[new] = data[old]
        results[m] = data
    return results


def _save_results(results: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for m in METRICS:
        with open(RESULTS_DIR / f"{m}.pkl", "wb") as f:
            pickle.dump(results[m], f)


# --- Jobs -------------------------------------------------------------------------------
# One generation job per (benchmark, distance) and one job per (benchmark, distance, method). A method job
# starts as soon as its circuit exists, so generation, distances and methods all run in parallel. Jobs
# return {metric: value} dicts; only the main process touches `results` and the pickles.


def _run_method(method: str, qc, partitions, backend, mech_cfg, timeout: float,
                store: Path | None = None) -> dict:
    """Run one transpilation method under the wall-clock limit ``timeout``; return the metrics to store.
    ``store``: file to keep the compiled circuit in (None = do not store)."""
    depth0, gates0 = circuit_depth(qc), num_2q_gates(qc)

    def from_circuit(circ, entry: dict) -> dict:
        if circ is not None and store is not None:
            try:
                save_circuit(store, circ)
            except Exception as e:  # storing is a convenience; never lose the run because of it
                print(f"  could not store the circuit ({e!r})")
        return {f"{method}_all": entry,
                f"{method}_depth": circuit_depth(circ) - depth0 if circ is not None else -1,
                f"{method}_overhead": num_2q_gates(circ) - gates0 if circ is not None else -1}

    if method == "custom":
        print(f"  Chipmunq: limit {timeout:.0f} s")
        return from_circuit(*run_timed(custom_partitioned_transpilation, qc, backend,
                                       pre_defined_partitions=partitions, timeout=timeout))

    if method == "sabre":
        print(f"  LightSABRE: limit {timeout:.0f} s")
        return from_circuit(*run_timed(sabre_transpilation, qc, backend, timeout=timeout))

    if method == "mech":
        if mech_cfg["mode"] == "run":
            side = mech_cfg["side"] or math.ceil(math.sqrt(1.3 * qc.num_qubits))
            mech = run_mech_with_timeout(qc, side, timeout)
        else:
            mech = {"status": "timeout", "note": "not rerun"}
        ok = mech["status"] == "ok"
        return {"mech_all": mech,
                # MECH's own depth has no annotations: compare it with the annotation-free ideal depth
                # (its "depth_overhead" subtracts the annotated depth and understates the overhead)
                "mech_depth": mech["depth"] - depth0 if ok else -1,
                "mech_overhead": mech["2q_gates_overhead"] if ok else -1}

    if method in EXTRA_TOOLS:
        return from_circuit(*run_extra_method(method, qc, backend, timeout))

    raise ValueError(f"Unknown method '{method}' (known: {RUN_TOOLS}).")


def generation_job(key: str, ks: int, cache: str | None, store: bool = True) -> dict:
    """Generate the circuit, cache (qc, partitions) for the method jobs, return the ideal statistics.
    ``store``: also keep the ideal circuit in CIRCUIT_DIR (for --recompute-metrics)."""
    t0 = time.time()
    try:
        circuit, partitions = generate_circuit(key, ks)
        qc = StimCodeCircuit(stim_circuit=circuit).qc
        if store:
            save_circuit(circuit_path(key, ks, "ideal"), qc)
        if cache:
            try:
                with open(cache + ".tmp", "wb") as f:
                    pickle.dump((qc, partitions), f)
                os.replace(cache + ".tmp", cache)
            except Exception as e:  # method jobs then regenerate the circuit themselves
                print(f"  {key} d={distance(ks)}: circuit not cacheable ({e!r}), method jobs regenerate it")
                if os.path.exists(cache + ".tmp"):
                    os.remove(cache + ".tmp")
        return {"ok": True, "elapsed_s": time.time() - t0,
                "metrics": {"depth_overall": circuit_depth(qc), "gate_overall": num_2q_gates(qc)}}
    except Exception:
        return {"ok": False, "elapsed_s": time.time() - t0, "error": traceback.format_exc()}


def method_job(key: str, ks: int, method: str, cache: str | None, timeout: float, store: bool = True) -> dict:
    """Run one method on a cached (or regenerated) circuit. Never raises: a crash is returned as an error
    entry *without* depth/overhead, so it is drawn as N/A and --run-missing retries it."""
    timeout = BENCHMARK_TIMEOUT_S.get(key, timeout)
    t0 = time.time()
    try:
        if cache and os.path.exists(cache):
            with open(cache, "rb") as f:
                qc, partitions = pickle.load(f)
        else:
            circuit, partitions = generate_circuit(key, ks)
            qc = StimCodeCircuit(stim_circuit=circuit).qc
        backend, mech_cfg = make_backend(key, ks, partitions)
        metrics = _run_method(method, qc, partitions, backend, mech_cfg, timeout,
                              store=circuit_path(key, ks, method) if store else None)
        return {"ok": True, "elapsed_s": time.time() - t0, "metrics": metrics}
    except Exception:
        err = traceback.format_exc()
        return {"ok": False, "elapsed_s": time.time() - t0, "error": err,
                "metrics": {f"{method}_all": {"status": "error", "error": err, "elapsed_s": time.time() - t0}}}


def _limit_threads(threads_per_job: int) -> None:
    """Keep N parallel jobs from each spawning a full thread pool (Qiskit's Rust SABRE, BLAS, HiGHS).
    Set in the parent before the pool starts, so every worker and its children inherit it."""
    for var in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(threads_per_job))
    os.environ.setdefault("QISKIT_PARALLEL", "FALSE")  # no nested process pools inside qiskit.transpile


def execute_jobs(to_run: dict[tuple[str, int], list[str]], results: dict, jobs: int,
                 timeout: float, threads_per_job: int = 1) -> None:
    """Run all (benchmark, distance) -> methods in ``to_run`` with ``jobs`` worker processes and merge the
    results into ``results`` (saved after every finished job). Every method run is limited to ``timeout`` s."""
    n_total = len(to_run) + sum(len(ms) for ms in to_run.values())
    done, lost = 0, []

    def merge(key, ks, what, out) -> None:
        nonlocal done
        done += 1
        for metric, value in out.get("metrics", {}).items():
            results[metric].setdefault(key, {})[ks] = value
        _save_results(results)
        if what == "circuit":
            status = "ok" if out["ok"] else "FAILED"
        elif not out["ok"]:
            status = "crashed"
        else:
            status = out["metrics"].get(f"{what}_all", {}).get("status", "ok")
        print(f"[{done}/{n_total}] {_title(key)} d={DISTANCE_INDEPENDENT.get(key, distance(ks))} "
              f"{what}: {status} ({out['elapsed_s']:.0f} s)", flush=True)
        if not out["ok"]:
            print(out["error"], flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(tempfile.mkdtemp(prefix=".circuits_", dir=RESULTS_DIR))
    cache_of = {pair: str(cache_dir / f"{pair[0]}_{pair[1]}.pkl") for pair in to_run}
    try:
        if jobs <= 1:  # in-process, one job after the other (easiest to debug)
            for (key, ks), methods in to_run.items():
                gen = generation_job(key, ks, cache_of[(key, ks)], STORE_CIRCUITS)
                merge(key, ks, "circuit", gen)
                if gen["ok"]:
                    for method in methods:
                        merge(key, ks, method, method_job(key, ks, method, cache_of[(key, ks)], timeout,
                                                          STORE_CIRCUITS))
            return

        _limit_threads(threads_per_job)
        # "spawn": the parent has already run Qiskit's multi-threaded (Rust) code, forking it is unsafe.
        # ProcessPoolExecutor workers are not daemonic (Python >= 3.9), so MECH / run_with_timeout can
        # still start their own child processes for the time limits.
        ex = ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn"))
        try:
            pending = {ex.submit(generation_job, key, ks, cache_of[(key, ks)], STORE_CIRCUITS): (key, ks, "circuit")
                       for key, ks in to_run}
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    key, ks, what = pending.pop(fut)
                    try:
                        out = fut.result()
                    except BrokenProcessPool:
                        lost.append((key, ks, what))
                        continue
                    merge(key, ks, what, out)
                    if what == "circuit" and out["ok"]:
                        for method in to_run[(key, ks)]:
                            f = ex.submit(method_job, key, ks, method, cache_of[(key, ks)], timeout,
                                          STORE_CIRCUITS)
                            pending[f] = (key, ks, method)
        except KeyboardInterrupt:
            print("Interrupted: finished jobs are saved; rerun with --run-missing to continue.")
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        ex.shutdown(wait=True)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
    if lost:
        print("A worker process died (out of memory?), so the pool stopped. Not finished: "
              + ", ".join(f"{k} d={DISTANCE_INDEPENDENT.get(k, distance(ks))} {w}" for k, ks, w in lost)
              + ". Rerun with --run-missing (and fewer --jobs) to continue.")


def _has(results: dict, key: str, ks: int, tool: str) -> bool:
    """True if (key, ks) has saved results for ``tool`` ("ideal" = the untranspiled statistics)."""
    metrics = ("depth_overall", "gate_overall") if tool == "ideal" else (f"{tool}_depth", f"{tool}_overhead")
    return all(ks in results[m].get(key, {}) for m in metrics)


def _missing(results: dict, ks_list: list[int], methods) -> dict[tuple[str, int], list[str]]:
    """{(benchmark, ks): methods without saved results}; distance-independent benchmarks need one run only."""
    out = {}
    for key in BENCHMARKS:
        if key in DISTANCE_INDEPENDENT:
            saved = list(results["depth_overall"].get(key, {}))
            pairs = [(key, saved[0] if saved else ks_list[0])]
        else:
            pairs = [(key, ks) for ks in ks_list]
        for k, ks in pairs:
            todo = [t for t in methods if not _has(results, k, ks, t)]
            if todo:
                out[(k, ks)] = todo
    return out


def run_exp_statistics(reproduce: list[str] | None = None, ks_list: list[int] | None = None,
                       run_missing: bool = False, methods=RUN_TOOLS, show=ALL_TOOLS,
                       jobs: int = 1, threads_per_job: int = 1, timeout: float | None = None,
                       recompute: bool = False) -> None:
    """Plot the saved results for all distances in ``ks_list``. By default nothing is run.

    :param reproduce: benchmarks to (re)run with ``methods`` before plotting, for every distance in ``ks_list``
    :param run_missing: also run every (benchmark, distance, method in ``methods``) without saved results
    :param methods: transpilation methods that ``reproduce`` / ``run_missing`` may run (default: all)
    :param show: methods drawn in the plots (default: all; "ideal" is the untranspiled circuit)
    :param jobs: worker processes; circuit generation, distances and methods run in parallel (1 = serial)
    :param threads_per_job: threads each job may use (Rust SABRE, BLAS, ...), only used if jobs > 1
    :param timeout: wall-clock limit in seconds for every method run (default METHOD_TIMEOUT_S)
    """
    ks_list = sorted(ks_list or DISTANCE_SCALES)
    methods = [t for t in RUN_TOOLS if t in methods]
    timeout = METHOD_TIMEOUT_S if timeout is None else timeout
    results = _load_results()

    if recompute:  # metrics from the stored circuits, no compilation
        n = recompute_metrics(results, ks_list)
        print(f"Recomputed {n} entries from the stored circuits in {CIRCUIT_DIR}")

    to_run: dict[tuple[str, int], list[str]] = {}
    for key in reproduce or []:
        for ks in ([ks_list[0]] if key in DISTANCE_INDEPENDENT else ks_list):
            to_run[(key, ks)] = list(methods)
    if run_missing:
        for pair, todo in _missing(results, ks_list, methods).items():
            to_run.setdefault(pair, [])
            to_run[pair] += [t for t in todo if t not in to_run[pair]]
    order = {k: i for i, k in enumerate(BENCHMARKS)}
    to_run = dict(sorted(to_run.items(), key=lambda kv: (order[kv[0][0]], kv[0][1])))

    if to_run:
        print(f"Running (limit {timeout:.0f} s per method"
              + "".join(f", {k}: {v:.0f} s" for k, v in BENCHMARK_TIMEOUT_S.items()) + "): "
              + "; ".join(f"{k} d={distance(ks)} [{', '.join(ms)}]" for (k, ks), ms in to_run.items()))
        execute_jobs(to_run, results, jobs, timeout, threads_per_job)

    # A benchmark is plotted if its ideal statistics exist for every distance; a shown method without a
    # result is drawn as a hollow bar (N/A: just below the ideal; T/O: 30 % above the ideal), unlabelled.
    no_ideal = [k for k in BENCHMARKS if k in DISTANCE_INDEPENDENT and not results["depth_overall"].get(k)
                or k not in DISTANCE_INDEPENDENT and not all(_has(results, k, ks, "ideal") for ks in ks_list)]
    if no_ideal:
        print("No saved results for " + ", ".join(no_ideal)
              + "; those benchmarks are not plotted (use --reproduce or --run-missing)")
    missing = _missing(results, ks_list, [t for t in show if t != "ideal"])
    for (k, ks), todo in missing.items():
        if k not in no_ideal:
            print(f"  {k} d={distance(ks)}: no results for {', '.join(todo)} (drawn as N/A)")
    keys = [k for k in BENCHMARKS if k not in no_ideal]
    if not keys:
        raise SystemExit("Nothing to plot.")
    plot_combined_split(results, keys, ks_list, str(PLOT_PREFIX), tools=show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the scalability results (default) or run the experiments.")
    parser.add_argument("--reproduce", nargs="*", choices=BENCHMARKS, default=None,
                        help="benchmarks to run before plotting, for every distance (no names = all)")
    parser.add_argument("--run-missing", action="store_true",
                        help="also run every (benchmark, distance, method) without saved results")
    parser.add_argument("--methods", nargs="+", choices=RUN_TOOLS, default=list(RUN_TOOLS),
                        help="methods that --reproduce / --run-missing run (default: all)")
    parser.add_argument("--show", nargs="+", choices=ALL_TOOLS, default=list(ALL_TOOLS),
                        help="methods drawn in the plots (default: all)")
    parser.add_argument("--distances", nargs="+", type=int, default=[distance(k) for k in DISTANCE_SCALES],
                        help="code distances to evaluate/plot (odd, e.g. 3 5 7)")
    parser.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 1,
                        help="worker processes for generation / distances / methods (default: all cores; 1 = serial)")
    parser.add_argument("--threads-per-job", type=int, default=1,
                        help="threads per job for Qiskit's Rust passes, BLAS etc. (default 1)")
    parser.add_argument("--timeout", type=float, default=METHOD_TIMEOUT_S,
                        help=f"wall-clock limit per method run in seconds, for all methods "
                             f"(default {METHOD_TIMEOUT_S:.0f}); longer runs are killed and shown as T/O")
    parser.add_argument("--recompute-metrics", action="store_true",
                        help="recompute depth / #2q gates from the stored circuits (no compilation), then plot")
    parser.add_argument("--no-store-circuits", action="store_true",
                        help=f"do not keep the compiled circuits in {CIRCUIT_DIR}")
    args = parser.parse_args()
    STORE_CIRCUITS = not args.no_store_circuits
    if any(d < 3 or d % 2 == 0 for d in args.distances):
        parser.error("--distances must be odd and >= 3")
    to_run = None if args.reproduce is None else (args.reproduce or BENCHMARKS)
    run_exp_statistics(reproduce=to_run, ks_list=[(d - 1) // 2 for d in args.distances],
                       run_missing=args.run_missing, methods=args.methods, show=args.show,
                       jobs=args.jobs, threads_per_job=args.threads_per_job, timeout=args.timeout,
                       recompute=args.recompute_metrics)