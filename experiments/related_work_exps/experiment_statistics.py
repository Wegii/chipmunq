"""Related-work comparison on surface-code memory circuits.

Each method transpiles each circuit once. That single run gives both the compilation time and
the circuit statistics, which are plotted as

    a) compilation time              -> memory_scaling_mono.pdf
    b) #2q gate overhead             -> memory_overhead.pdf
    c) #inter-chiplet gates          -> memory_inter_chiplet.pdf
    legend for all three             -> memory_legend.pdf

Results are stored per method in related_work/results_<method>.pkl as
    {d: {"runtime": seconds | TIMEOUT | FAILED, "stats": dict | TIMEOUT | FAILED}}
and written after every distance, so an interrupted run keeps what it finished.
"""

from __future__ import annotations

import os
import sys
import time
import traceback

sys.path.append(os.path.join(os.getcwd(), "."))

# MECH
sys.path.append(os.path.join(os.getcwd(), "./external/baseline/MECH"))
from external.baseline.MECH.Circuit import *
from external.baseline.MECH.Chiplet import *
from external.baseline.MECH.HighwayOccupancy import *
from external.baseline.MECH.Router import *
from external.baseline.MECH.MECHBenchmarks import *
from external.baseline.MECH.transpile_mech import transpile_circuit_MECH

# QECC-Synth
sys.path.append(os.path.join(os.getcwd(), "./external/baseline/QECC_Synth/SurfStitch/MyCode/src"))
from external.baseline.QECC_Synth.SurfStitch.MyCode.src.transpile_qeccsynth import transpile_circuit_QECCSynth

# SABRE
sys.path.append(os.path.join(os.getcwd(), "./external/baseline/SABRE"))
from external.baseline.SABRE.transpile_sabre import transpile_circuit_SABRE

# OLSQ2, SEQC, Murali et al.
from experiments.exp_utils.transpilation_utils import (
    ALL_METHODS,
    EXTRA_METHODS,
    FAILED,
    METHOD_STYLES,
    TIMEOUT,
    TIMEOUT_S,
    load_results,
    run_with_timeout,
    save_results,
    style_timeout_bar,
)

# Plotting
from matplotlib.ticker import MaxNLocator
from experiments.related_work_exps.utils import *
from experiments.exp_utils.utils import *
from pathlib import Path

OUTPUT_DIR = Path("experiments/evaluation/related_work")
RESULTS_PATTERN = "results_{}.pkl"

# Surface-code scale d (code distance 2d + 1); one transpilation per method and d.
CODE_DISTANCES = [2, 3, 4, 5]

# The original runtime figure marked QECC-Synth at these d as timed out by hand
# (QECC-Synth returns after its own internal limit). Only affects the runtime plot.
MANUAL_RUNTIME_TIMEOUTS = {"qeccsynth": {3, 4}}

PLOTS = {
    # metric: (title, title x-shift, y label, file name)
    "runtime": ("a) Effect of distance on compilation time", -0.06, "Runtime [s]", "memory_scaling_mono"),
    "gate_overhead": ("b) Effect of distance on #2q gate overhead", -0.09, "#2q gate overhead", "memory_overhead"),
    "inter_chiplet": ("c) Effect of distance on #inter-chiplet gates", -0.12, "#inter-chiplet gates",
                      "memory_inter_chiplet"),
}


# --------------------------------------------------------------------------------------
# Running the methods
# --------------------------------------------------------------------------------------


def _timed(fn, *args, **kwargs):
    """In-process run: (result, runtime, status); an exception is recorded, not raised."""
    start = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
    except Exception:
        print(f"[{getattr(fn, '__name__', fn)}] failed:\n{traceback.format_exc()}")
        return None, time.perf_counter() - start, FAILED
    return result, time.perf_counter() - start, "ok"


def _stats(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        print(f"[{getattr(fn, '__name__', fn)}] failed:\n{traceback.format_exc()}")
        return FAILED


def run_method(method: str, d: int, code, monolithic_backend, architecture, cm, n: int, m: int) -> dict:
    """Transpile once; return {"runtime": ..., "stats": ...}."""
    if method == "mech":
        circuit, runtime, status = _timed(transpile_circuit_MECH, code.qc, monolithic_backend)
        stats = _stats(calc_circuit_mech_stats, circuit, code.qc) if status == "ok" else status

    elif method == "qeccsynth":
        out, runtime, status = _timed(transpile_circuit_QECCSynth, d, architecture, f"square_{n}_{m}_{m}")
        if status == "ok":
            circuit, result = out
            stats = _stats(calc_circuit_qiskit_stats, circuit, monolithic_backend, result)
        else:
            stats = status

    elif method == "sabre":
        circuit, runtime, status = _timed(transpile_circuit_SABRE, circuit=code.qc, coupling_map=cm)
        stats = _stats(calc_circuit_qiskit_stats, circuit, monolithic_backend,
                       initial_circuit=code.qc) if status == "ok" else status

    elif method in EXTRA_METHODS:
        # Separate process with a TIMEOUT_S limit (OLSQ2 in particular can run for hours).
        circuit, runtime, status = run_with_timeout(EXTRA_METHODS[method], code.qc, cm)
        stats = _stats(calc_circuit_qiskit_stats, circuit, monolithic_backend,
                       initial_circuit=code.qc) if status == "ok" else status

    else:
        raise ValueError(f"Unknown method '{method}' (known: {ALL_METHODS}).")

    return {"runtime": runtime if status == "ok" else status, "stats": stats}


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------


def _value(key: str, entry, metric: str):
    """Plotted value for one (method, d); TIMEOUT / FAILED / None if there is no number.

    Keeps the conventions of the original figures: MECH as reported by
    calc_circuit_mech_stats; QECC-Synth +1 on both statistics; all qiskit-stats methods
    (LightSABRE, OLSQ2, SEQC, Murali) +1 on the 2q overhead only.
    """
    if entry is None:
        return None
    if metric == "runtime":
        return entry["runtime"]
    stats = entry["stats"]
    if not isinstance(stats, dict):
        return stats  # TIMEOUT or FAILED
    if metric == "gate_overhead":
        v = stats["2q_gates_overhead"]
        return v if key == "mech" else 1 + v
    v = stats["cross-chip"]
    return 1 + v if key == "qeccsynth" else v


def _set_style() -> None:
    plt.rcParams.update({
        # Use LaTeX to write all text
        # "text.usetex": True,
        "font.family": "serif",
        # Font sizes
        "axes.labelsize": FONTSIZE * 1.5,
        "font.size": FONTSIZE * 1.2,
        "legend.fontsize": (FONTSIZE - 2) * 1.5,
        "xtick.labelsize": (FONTSIZE - 1) * 1.5,
        "ytick.labelsize": (FONTSIZE - 1) * 1.5,
        "axes.titlesize": 10,
        # Line and marker styles
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "lines.markeredgewidth": 1.5,
        "lines.markeredgecolor": "black",
        # Error bar cap size
        "errorbar.capsize": 3,
    })


def plot_metric(results: dict, metric: str, distances: list[int]) -> list:
    """Grouped bars per code distance, one bar per method. Returns the bar handles.

    Runtime: a timed-out run is a hollow bar at TIMEOUT_S labelled T/O.
    Statistics: a timed-out or failed run has no circuit, so no bar, only a T/O / fail label.
    """
    title, shift, ylabel, fname = PLOTS[metric]
    _set_style()
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE * 2.5, WIDTH_FIGSIZE * 0.5))

    x = np.arange(len(distances))
    styles = [s for s in METHOD_STYLES if s[0] in results]
    width = 0.8 / len(styles)
    handles = []

    for i, (key, label, color, hatch) in enumerate(styles):
        offset = (i - (len(styles) - 1) / 2) * width
        values = [_value(key, results[key].get(d), metric) for d in distances]
        numeric = [isinstance(v, (int, float)) and not isinstance(v, bool) for v in values]
        timed_out = [v == TIMEOUT for v in values]
        if metric == "runtime":
            for j, d in enumerate(distances):
                if d in MANUAL_RUNTIME_TIMEOUTS.get(key, ()):
                    timed_out[j] = True
            heights = [v if ok else (TIMEOUT_S if t else 0) for v, ok, t in zip(values, numeric, timed_out)]
        else:
            heights = [v if ok else 0 for v, ok in zip(values, numeric)]

        bars = ax.bar(x + offset, heights, width, label=label, color=color, hatch=hatch, edgecolor="black")
        handles.append(bars)

        for j, v in enumerate(values):
            if metric == "runtime" and timed_out[j]:
                style_timeout_bar(bars[j], color)
                ax.text(x[j] + offset, max(heights[j], TIMEOUT_S) * 1.2, "T/O", ha="center", va="bottom",
                        color="red", fontweight="bold", fontsize=9, rotation=90)
            elif not numeric[j] and v is not None:  # no bar: say why
                ax.text(x[j] + offset, 0.03, "T/O" if v == TIMEOUT else "fail", transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", color="red", fontweight="bold", fontsize=9, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([2 * d + 1 for d in distances])
    ax.text(shift, 1.02, title, transform=ax.transAxes, fontweight="bold")
    ax.text(0.3, 1.15, "Lower is better ↓", transform=ax.transAxes, fontweight="bold", color=plot_lib_color)

    plt.tick_params(axis="both", labelsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Surface code distance", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.yscale("log")
    if metric == "runtime":
        ax.set_ylim(top=2e4)  # headroom for the rotated T/O labels above TIMEOUT_S
    plt.grid(True, which="major", linestyle="--", alpha=0.5)

    fig.subplots_adjust(left=0.175, right=0.95, top=0.83, bottom=0.2)
    plt.savefig(OUTPUT_DIR / f"{fname}.pdf", format="pdf")
    plt.close(fig)
    return handles


def plot_all(results: dict, distances: list[int]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    handles = None
    for metric in PLOTS:
        handles = plot_metric(results, metric, distances)

    # One legend for all three plots
    legend_fig = plt.figure(figsize=(3, 2))
    legend_fig.legend(handles=handles, loc="center", frameon=False, ncols=3)
    legend_fig.savefig(OUTPUT_DIR / "memory_legend.pdf", bbox_inches="tight", format="pdf")
    plt.close(legend_fig)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def run_experiment(reproduce: bool = False, methods=ALL_METHODS, distances=CODE_DISTANCES) -> None:
    """``methods`` selects what is re-run when ``reproduce`` is set. The plots always show
    every method that has a result file, so ``methods=["olsq2", "seqc", "murali"]`` adds the
    new baselines without re-running QECC-Synth. Re-running a method replaces its file."""
    if reproduce:
        results = {mth: {} for mth in methods}

        for d in distances:
            cycles = d
            code = get_surface_code_stim(d, cycles)

            # TODO: calculate chiplet size based on distance
            n = m = int(d * 1.5)

            monolithic_backend, qubit_num, data_qubit_num = generate_simple_backend(n, m)
            architecture = generate_qecc_synth_backend_from_mech(monolithic_backend)
            cm = generate_qiskit_backend_from_mech(monolithic_backend)

            # Print backend to file
            # display_simple_backend(monolithic_backend, f"experiments/evaluation/related_work/backends/mono_{n}_{m}.png")

            for mth in methods:
                print(f"d={d} ({2 * d + 1}): {mth} ...", flush=True)
                results[mth][d] = run_method(mth, d, code, monolithic_backend, architecture, cm, n, m)
                print(f"    runtime: {results[mth][d]['runtime']}", flush=True)

            # Save after every distance so an interrupted run keeps its results
            save_results(OUTPUT_DIR, RESULTS_PATTERN, results)

    # Load pre-computed results
    results = load_results(OUTPUT_DIR, RESULTS_PATTERN)
    plot_all(results, distances)


if __name__ == "__main__":
    run_experiment(reproduce=True)