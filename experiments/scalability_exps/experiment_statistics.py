from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import pickle
import queue
import sys
import time
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

# The `from ... import *` lines above pull numpy's sum/any/all/min/max/... into this module and shadow the
# Python builtins (np.any(generator) is always True, np.sum(generator) is deprecated). Restore the builtins.
from builtins import abs, all, any, max, min, round, sum

# Plotting
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

# Code distances to evaluate, as tqec scales k (d = 2k + 1). The CNOT generator has hand-written
# partitions for k = 1, 2, 3, 7 only (d = 3, 5, 7, 15), and Chipmunq's mapper places rotated
# surface-code patches for d = 3, 5, 7, 9, 15.
DISTANCE_SCALES = [1, 2, 3]  # d = 3, 5, 7
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
BV_SECRET = "110100111010110100111010110100111010110100111010110100111010110100111010110100111010110100111010"  # 12-bit secret with mixed 0/1 so the bus is partially trimmed
GROSS_BRIDGE_N_INTER = 6  # links between neighbouring chiplets of the Gross-bridge backend (max 6)

MECH_TIMEOUT_S = 10#0.0

RESULTS_DIR = Path("experiments/evaluation/scalability")
PLOT_PREFIX = RESULTS_DIR / "cnot_scaling_overhead_split"

# Order = order of the bar groups in the plot.
BENCHMARKS = ["cnot", "ghz", "bv", "gross_bridge"]
TITLES = {
    "cnot": "CNOT",
    "ghz": f"GHZ-{GHZ_N}",
    "bv": f"BV-{len(BV_SECRET)}",
    "gross_bridge": "Gross surgery",
    "gross": "Gross Code",  # memory only; no longer in BENCHMARKS
}

METRICS = [
    "custom_depth", "custom_overhead",
    "sabre_depth", "sabre_overhead",
    "mech_depth", "mech_overhead", "mech_all",
    "depth_overall", "gate_overall",
]

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


def build_benchmark(key: str, ks: int):
    """Return (stim circuit, partitions, Chipmunq backend, MECH config)."""
    if key == "cnot":
        circuit, partitions = get_tqec_cnot_rotated(distance_scale=ks, n1=1, n2=0)
        backend = _chipmunq_backend((2, 2, *chiplet_shape(ks)), ks)
        # MECH grid side 14 was chosen for d = 5; other distances use the automatic size
        return circuit, partitions, backend, {"mode": "run", "side": 14 if ks == 2 else None}

    if key == "ghz":
        circuit, partitions = get_tqec_ghz(GHZ_N, distance_scale=ks)
        return circuit, partitions, _chipmunq_backend(_chiplet_grid_for(partitions, ks), ks), {"mode": "run", "side": None}

    if key == "bv":
        circuit, partitions = get_tqec_bv(BV_SECRET, distance_scale=ks)
        return circuit, partitions, _chipmunq_backend(_chiplet_grid_for(partitions, ks), ks), {"mode": "run", "side": None}

    if key == "gross_bridge":
        # Two [[144,12,12]] modules + bridge measuring Zbar_a Zbar_b (fixed d = 12, ks is not used).
        # One chiplet per module and one for the bridge. n_inter <= 6: with 12 rows per chiplet,
        # BackendChipletV2 places the horizontal links at +-(n_inter - 1) rows around the centre.
        circuit, partitions = get_gross_bridge()
        backend = BackendChipletV2(
            size=(1, 3, 12, 24),
            n_inter=GROSS_BRIDGE_N_INTER,
            connectivity="torus",
            topology="grid",
            long_range_offsets=[(1, 0), (2, 0), (3, 0), (0, 1), (0, 2), (0, 3)],
            num_defective_qubits=0,
        )
        # MECH is not run on BB codes (it timed out on the Gross memory), same as before.
        return circuit, partitions, backend, {"mode": "timeout"}

    if key == "gross":
        circuit, partitions = generate_gross_code(num_qubits=1)
        backend = BackendChipletV2(
            # 1 chiplet with 288 qubits, long-range connections, grid layout
            size=(1, 1, 12, 24),
            n_inter=1,
            connectivity="torus",
            topology="grid",
            long_range_offsets=[(1, 0), (2, 0), (3, 0), (0, 1), (0, 2), (0, 3)],
            num_defective_qubits=0,
        )
        return circuit, partitions, backend, {"mode": "timeout"}

    raise ValueError(f"Unknown benchmark '{key}'.")


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

    * "spawn", not "fork": the parent has already run Qiskit's multi-threaded (Rust) SABRE, and
      forking a process that owns a thread pool can deadlock or crash the child.
    * not a daemon: daemonic processes may not start children, which MECH wrappers that enforce
      their own time limit via multiprocessing need.
    """
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    proc = ctx.Process(target=_mech_worker, args=(qc, side, timeout_s, out), daemon=False)
    proc.start()
    t0 = time.time()
    fwd = _mech_timeout_kwargs(timeout_s)
    print(f"  MECH on {side}x{side} grid, limit {timeout_s:.0f} s"
          + (f" (forwarded to wrapper as {list(fwd)})" if fwd else ""))
    while True:
        try:
            status, payload = out.get(timeout=5)
            break
        except queue.Empty:
            if not proc.is_alive():
                return {"status": "error", "elapsed_s": time.time() - t0,
                        "error": f"MECH process exited with code {proc.exitcode} after {time.time() - t0:.0f} s"}
            if time.time() - t0 > timeout_s:
                proc.terminate()
                proc.join()
                return {"status": "timeout", "timeout_s": timeout_s, "elapsed_s": time.time() - t0}
    proc.join()
    out.close()
    out.join_thread()
    if status == "error":
        return {"status": "error", "error": payload, "elapsed_s": time.time() - t0}
    result = dict(payload)
    result.update(status="ok", side=side, runtime_s=time.time() - t0)
    return result


def _mech_label(entry) -> str | None:
    """Bar annotation for a MECH result: None if it ran, T/O on timeout, N/A on error."""
    if isinstance(entry, dict) and "status" in entry:
        if entry["status"] == "error" and "timeout" in str(entry.get("error", "")).lower():
            return "T/O"  # MECH's own time limit fired
        return {"ok": None, "timeout": "T/O", "error": "N/A"}[entry["status"]]
    if entry == -1:  # legacy pickles
        return "T/O"
    return None


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------

PASTEL_BLUE = "#A7D9ED"
PASTEL_ORANGE = "#F7C6A2"
PASTEL_GREEN = "#B5D8B0"
SERIES_STYLE = [
    ("Ideal", "lightcoral", "//"),
    ("Chipmunq", PASTEL_BLUE, "/"),
    ("LightSABRE", PASTEL_ORANGE, "o"),
    ("MECH", PASTEL_GREEN, "//"),
]
TOOLS = ("ideal", "custom", "sabre", "mech")
DARKEST = 0.55  # lightness reduction of the largest distance (0 = base colour)


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


def _draw_bars(fig, data, keys, ks_list, mech_labels, ylim=None, width=0.19):
    """Grouped bars (Ideal / Chipmunq / LightSABRE / MECH), one overlaid segment per code distance.

    ``data[tool][i]`` maps ks -> value for benchmark ``keys[i]``. For every tool the bars of all
    distances share one x position; the largest distance is drawn first (darkest, at the back) and
    smaller distances in front of it, so each segment shows the value of one distance "stacked" on the
    previous one.
    """
    x = np.arange(len(keys))
    ax = fig.add_subplot(111)
    levels = _levels(ks_list)
    allv = [v for t in TOOLS for per_key in data[t] for v in per_key.values()]
    ymax = ylim[1] if ylim else _nice(1.05 * max(allv))

    for j, (tool, (label, color, hatch)) in enumerate(zip(TOOLS, SERIES_STYLE)):
        xpos = x + (j - 1.5) * width
        for i, key in enumerate(keys):
            per_ks = data[tool][i]
            timed_out = mech_labels[i] if tool == "mech" else {}
            for rank, ks in enumerate(sorted(per_ks, reverse=True)):  # largest d first (back)
                lvl = 1.0 if key in DISTANCE_INDEPENDENT else levels[ks]
                bar = ax.bar(xpos[i], per_ks[ks], width, color=_shade(color, lvl), hatch=hatch,
                             edgecolor="black", linewidth=0.8, zorder=2 + rank)[0]
                if ks in timed_out:  # MECH without a result: white, hatched, dashed outline
                    bar.set_facecolor("white")
                    bar.set_linestyle("--")
                    bar.set_hatch("xxx")
                    bar.set_linewidth(2)
                    bar.set_edgecolor(_shade("#B2D8B2", lvl))
            if timed_out:
                top = max(per_ks[ks] for ks in timed_out)
                lab = "T/O" if len(timed_out) == len(per_ks) else \
                      "T/O " + ",".join(f"d={distance(k)}" for k in sorted(timed_out))
                ax.annotate(lab, xy=(xpos[i], min(top, ymax)), xytext=(0, 2), textcoords="offset points",
                            ha="left", va="bottom", rotation=45, rotation_mode="anchor",
                            color="red", fontweight="bold", fontsize=10, annotation_clip=False, zorder=20)

    ax.set_ylim(*(ylim or (0, ymax)))
    ax.set_xticks(x)
    ax.set_xticklabels([TITLES[k] + (f"\n(d={DISTANCE_INDEPENDENT[k]})" if k in DISTANCE_INDEPENDENT else "")
                        for k in keys])
    ax.tick_params(axis="x", which="both", top=False)
    ax.tick_params(axis="y", length=5)
    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Legend handles: tools (medium shade) and distances (grey ramp)
    from matplotlib.patches import Patch
    tool_handles = [Patch(facecolor=_shade(c, 0.5), hatch=h, edgecolor="black", label=l) for l, c, h in SERIES_STYLE]
    dist_handles = [Patch(facecolor=_shade("#d9d9d9", levels[ks]), edgecolor="black", label=f"d = {distance(ks)}")
                    for ks in sorted(ks_list)]
    return ax, tool_handles + dist_handles


def _pct(v: float, ideal: float) -> str:
    return f"{100 * (v - ideal) / ideal:.0f}%"


def _ks_for(key: str, ks_list: list[int], results: dict) -> list[int]:
    """Distances plotted for a benchmark (one entry for distance-independent benchmarks)."""
    if key in DISTANCE_INDEPENDENT:
        have = [ks for ks in results["depth_overall"].get(key, {})]
        return have[:1]
    return list(ks_list)


def plot_combined_split(results: dict, keys: list[str], ks_list: list[int], filename: str,
                        ylim_depth=None, ylim_gates=None) -> None:
    """Bar plots over several code distances. ylim_* = (lo, hi) to override the automatic y range."""

    def _collect(overall: str, suffix: str) -> dict:
        out = {t: [] for t in TOOLS}
        for key in keys:
            ks_here = _ks_for(key, ks_list, results)
            ideal = {ks: results[overall][key][ks] for ks in ks_here}
            out["ideal"].append(ideal)
            for t in ("custom", "sabre", "mech"):
                out[t].append({ks: ideal[ks] + results[f"{t}_{suffix}"][key][ks] for ks in ks_here})
        return out

    depth = _collect("depth_overall", "depth")
    gates = _collect("gate_overall", "overhead")
    mech_labels = []
    for key in keys:
        labs = {}
        for ks in _ks_for(key, ks_list, results):
            lab = _mech_label(results["mech_all"].get(key, {}).get(ks))
            if lab:
                labs[ks] = lab
        mech_labels.append(labs)

    # Summary
    for i, key in enumerate(keys):
        for ks in sorted(depth["ideal"][i]):
            di, dc, ds, dm = (depth[t][i][ks] for t in TOOLS)
            gi, gc, gsab, gm = (gates[t][i][ks] for t in TOOLS)
            mech_d = mech_labels[i].get(ks) or _pct(dm, di)
            mech_g = mech_labels[i].get(ks) or _pct(gm, gi)
            d = DISTANCE_INDEPENDENT.get(key, distance(ks))
            print(f"[{TITLES[key]} d={d}] depth ideal={di}  ours={_pct(dc, di)}  sabre={_pct(ds, di)}  mech={mech_d}"
                  f"  | 2q ideal={gi}  ours={_pct(gc, gi)}  sabre={_pct(gsab, gi)}  mech={mech_g}")

    # ---------------- depth ----------------
    plt.rcParams.update(_tex_fonts(1.5))
    fig = plt.figure(figsize=(HEIGHT_FIGSIZE * 2.5, WIDTH_FIGSIZE * 0.5))
    ax, _ = _draw_bars(fig, depth, keys, ks_list, mech_labels, ylim_depth)
    ax.set_xlabel("Circuit type")
    fig.text(0.03, 0.5, "Circuit depth", va="center", rotation="vertical", fontsize=FONTSIZE * 1.5)
    title = ax.text(-0.18, 1.04, "b) Compilation overhead on circuit depth", transform=ax.transAxes,
                    fontweight="bold", va="bottom", ha="left")
    # "Lower is better" centred directly above the title, independent of figure size
    ax.annotate("Lower is better ↓", xy=(0.5, 1.0), xycoords=title, xytext=(0, 2), textcoords="offset points",
                fontweight="bold", color=plot_lib_color, va="bottom", ha="center")
    fig.subplots_adjust(left=0.24, right=0.95, top=0.72, bottom=0.21)
    fig.savefig(f"{filename}_depth.pdf", format="pdf")
    plt.close(fig)

    # ---------------- 2q gates ----------------
    plt.rcParams.update(_tex_fonts(1.3))
    fig = plt.figure(figsize=(HEIGHT_FIGSIZE * 2.5, WIDTH_FIGSIZE * 0.5))
    ax, handles = _draw_bars(fig, gates, keys, ks_list, mech_labels, ylim_gates)
    fig.text(0.025, 0.5, "#2q gates", va="center", rotation="vertical", fontsize=FONTSIZE * 1.5)
    fig.text(0.46, 0.055, "Circuit type", va="center", rotation="horizontal", fontsize=FONTSIZE * 1.5)
    title = ax.text(-0.075, 1.04, "c) Compilation overhead on #2q gates", transform=ax.transAxes,
                    fontweight="bold", va="bottom", ha="left")
    # "Lower is better" centred directly above the title, independent of figure size
    ax.annotate("Lower is better ↓", xy=(0.5, 1.0), xycoords=title, xytext=(0, 2), textcoords="offset points",
                fontweight="bold", color=plot_lib_color, va="bottom", ha="center")
    fig.subplots_adjust(left=0.24, right=0.95, top=0.72, bottom=0.21)
    fig.savefig(f"{filename}_overhead.pdf", format="pdf")
    plt.close(fig)

    # ---------------- legend ----------------
    legend_fig = plt.figure(figsize=(4, 2))
    legend_fig.legend(handles=handles, loc="center", frameon=False, ncols=len(handles), columnspacing=1.5)
    legend_fig.savefig(f"{filename}legend.pdf", bbox_inches="tight", format="pdf")
    plt.close(legend_fig)


# --------------------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------------------


def num_2q_gates(circuit) -> int:
    ops = circuit.count_ops()
    return sum(ops.get(g, 0) for g in ("cx", "cz", "swap"))


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


def run_benchmark(key: str, ks: int, results: dict) -> None:
    print(f"=== {TITLES[key]} (ks={ks}) ===")
    circuit, partitions, backend, mech_cfg = build_benchmark(key, ks)
    stim_code_circuit = StimCodeCircuit(stim_circuit=circuit)
    qc = stim_code_circuit.qc

    print("Custom")
    custom_circuit = custom_partitioned_transpilation(qc, backend, pre_defined_partitions=partitions)
    print("Sabre")
    sabre_circuit = sabre_transpilation(qc, backend)

    print("MECH")
    if mech_cfg["mode"] == "run":
        side = mech_cfg["side"] or math.ceil(math.sqrt(1.3 * qc.num_qubits))
        mech = run_mech_with_timeout(qc, side, MECH_TIMEOUT_S)
        elapsed = mech.get("runtime_s", mech.get("elapsed_s", 0))
        print(f"  MECH status: {mech['status']} after {elapsed:.0f} s"
              + (f"\n{mech['error']}" if mech["status"] == "error" else ""))
    else:
        mech = {"status": "timeout", "note": "not rerun"}

    def put(metric: str, value) -> None:
        results[metric].setdefault(key, {})[ks] = value

    depth0, gates0 = qc.depth(), num_2q_gates(qc)
    put("depth_overall", depth0)
    put("gate_overall", gates0)
    put("custom_depth", custom_circuit.depth() - depth0)
    put("custom_overhead", num_2q_gates(custom_circuit) - gates0)
    put("sabre_depth", sabre_circuit.depth() - depth0)
    put("sabre_overhead", num_2q_gates(sabre_circuit) - gates0)
    put("mech_all", mech)
    if mech["status"] == "ok":
        put("mech_depth", mech["depth_overhead"])
        put("mech_overhead", mech["2q_gates_overhead"])
    else:
        put("mech_depth", -1)
        put("mech_overhead", -1)


_REQUIRED_METRICS = ("custom_depth", "custom_overhead", "sabre_depth", "sabre_overhead",
                     "mech_depth", "mech_overhead", "depth_overall", "gate_overall")


def _has(results: dict, key: str, ks: int) -> bool:
    return all(ks in results[m].get(key, {}) for m in _REQUIRED_METRICS)


def _missing(results: dict, ks_list: list[int]) -> list[tuple[str, int]]:
    """(benchmark, ks) pairs without saved results; distance-independent benchmarks need one run only."""
    out = []
    for key in BENCHMARKS:
        if key in DISTANCE_INDEPENDENT:
            if not any(_has(results, key, ks) for ks in results["depth_overall"].get(key, {})):
                out.append((key, ks_list[0]))
        else:
            out += [(key, ks) for ks in ks_list if not _has(results, key, ks)]
    return out


def run_exp_statistics(reproduce: list[str] | None = None, ks_list: list[int] | None = None,
                       run_missing: bool = False) -> None:
    """Plot the saved results for all distances in ``ks_list``. By default nothing is run.

    :param reproduce: benchmarks to (re)run before plotting, for every distance in ``ks_list``
    :param run_missing: also run every (benchmark, distance) that has no saved results
    """
    ks_list = sorted(ks_list or DISTANCE_SCALES)
    results = _load_results()
    to_run = []
    for key in reproduce or []:
        to_run += [(key, ks_list[0])] if key in DISTANCE_INDEPENDENT else [(key, ks) for ks in ks_list]
    if run_missing:
        to_run += _missing(results, ks_list)
    order = {k: i for i, k in enumerate(BENCHMARKS)}
    to_run = sorted(dict.fromkeys(to_run), key=lambda p: (order[p[0]], p[1]))

    if to_run:
        print("Running: " + ", ".join(f"{k} d={distance(ks)}" for k, ks in to_run))
    for key, ks in to_run:
        run_benchmark(key, ks, results)
        _save_results(results)  # save after every run, so a crash doesn't lose finished runs

    missing = _missing(results, ks_list)
    if missing:
        print("No saved results for " + ", ".join(f"{k} d={distance(ks)}" for k, ks in missing)
              + "; those benchmarks are not plotted (use --reproduce or --run-missing)")
    keys = [k for k in BENCHMARKS if not any(m[0] == k for m in missing)]
    if not keys:
        raise SystemExit("Nothing to plot.")
    plot_combined_split(results, keys, ks_list, str(PLOT_PREFIX))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the scalability results (default) or run the experiments.")
    parser.add_argument("--reproduce", nargs="*", choices=BENCHMARKS, default=None,
                        help="benchmarks to run before plotting, for every distance (no names = all)")
    parser.add_argument("--run-missing", action="store_true",
                        help="also run every (benchmark, distance) without saved results")
    parser.add_argument("--distances", nargs="+", type=int, default=[distance(k) for k in DISTANCE_SCALES],
                        help="code distances to evaluate/plot (odd, e.g. 3 5 7)")
    parser.add_argument("--mech-timeout", type=float, default=MECH_TIMEOUT_S,
                        help=f"MECH wall-clock limit in seconds (default {MECH_TIMEOUT_S})")
    args = parser.parse_args()
    MECH_TIMEOUT_S = args.mech_timeout
    if any(d < 3 or d % 2 == 0 for d in args.distances):
        parser.error("--distances must be odd and >= 3")
    to_run = None if args.reproduce is None else (args.reproduce or BENCHMARKS)
    run_exp_statistics(reproduce=to_run, ks_list=[(d - 1) // 2 for d in args.distances],
                       run_missing=args.run_missing)