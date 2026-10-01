from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.getcwd(), "."))

from experiments.exp_utils.circuit_generator import get_tqec_cnot_rotated
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.utils import *
from experiments.exp_utils.ablation_utils import (
    check_routed,
    chipmunq_mapping,
    overhead_stats,
    quiet,
    route_from_mapping,
    symmetric_remote_noise,
)
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit
from qeccm.backends.backend_utils import plot_circuit_layout_utilization

# Plotting
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Patch, Rectangle
from pathlib import Path

OUTPUT_DIR = Path("experiments/evaluation/inter_chiplet")

# Routers compared, all starting from the *same* Chipmunq mapping on the *same* backend.
#   basic / tradeoff / focus : Chipmunq's CostRouter (configs below)
#   sabre                    : LightSABRE's SabreSwap, given the compile-time budget of the Chipmunq routers
#   murali                   : Murali et al. (ASPLOS'19) noise-adaptive routing, sees the same link noise
#   seqc                     : SEQC's noise-aware inter-chiplet routing (links: SWAP only), sees the same link noise
# key, label, colour (low p_inter), colour (high p_inter)
METHODS = [
    ("basic", "Basic", "#A7D9ED", "#5B9BD5"),
    ("tradeoff", "Tradeoff", "#F7C6A2", "#E68A5C"),
    ("focus", "Focus", "#F7A2A2", "#D65C5C"),
    ("sabre", "LightSABRE", "#D9D9D9", "#7F7F7F"),
    ("murali", "Murali et al.", "#B2D8B2", "#5E9E5E"),
    ("seqc", "SEQC", "#C9B7E3", "#8E6BBF"),
]
LABEL = {m: l for m, l, _, _ in METHODS}

# Hatch per link-noise level (light shade = low p_inter, dark shade = high p_inter)
HATCHES = {"low": "ooo", "high": "xxx"}  # dense: the bars are narrow at the 1.75 in panel width

# Cost routing configurations [alpha factor, beta]
COST_CONFIGS = {"basic": [0, 0], "tradeoff": [1, 1], "focus": [6, 1]}

# Link-noise levels. alpha scales with 1 / p_inter exactly as in the original experiment.
NOISE_LEVELS = {
    "low": dict(p_inter=1e-4, alpha=lambda c: c * 1e4, text=r"$1e^{-4}$"),
    "high": dict(p_inter=1e-2, alpha=lambda c: 1.5 * c * 1e2, text=r"$1e^{-2}$"),
}

SECTION_TITLES = ["Full", "Half", "Limited"]

# --------------------------------------------------------------------------------------
# Panel size: identical to the noise-aware routing panels (b-d), so this plot sits next to them -- four panels
# side by side across the full text width of a two-column paper. Drawn at printed size: include it in LaTeX at its
# natural width (or width=0.24\textwidth for a 7.0 in text width), no further scaling.
# --------------------------------------------------------------------------------------
TEXT_WIDTH_IN = 7.0                   # full text width of the paper (two-column IEEE/ACM: ~7.0 in)
N_PANELS = 4                          # panels in one row
PANEL_W = TEXT_WIDTH_IN / N_PANELS    # 1.75 in
PANEL_H = 1.6
FONT_PT = 7
MARGINS = dict(left=0.27, right=0.95, top=0.80, bottom=0.24)


def _panel_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": FONT_PT,
        "axes.labelsize": FONT_PT,
        "axes.titlesize": FONT_PT,
        "legend.fontsize": FONT_PT,
        "xtick.labelsize": FONT_PT - 1,
        "ytick.labelsize": FONT_PT - 1,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.labelpad": 2,
        "axes.linewidth": 0.6,
        "hatch.linewidth": 0.4,
        "lines.linewidth": 1.0,
        "errorbar.capsize": 1.5,
    })


def _panel_title(ax, title: str, better: str) -> None:
    """Panel title and the "better" hint directly above it, both centred above the axes (plot rectangle)."""
    t = ax.text(0.5, 1.03, title, transform=ax.transAxes, fontweight="bold", ha="center", va="bottom")
    ax.annotate(better, xy=(0.5, 1.0), xycoords=t, xytext=(0, 1), textcoords="offset points",
                fontweight="bold", color=plot_lib_color, ha="center", va="bottom")


def num_2q_gates(circuit):
    """Original metric of this experiment: every cx/cz/swap counts once."""
    ops = circuit.count_ops()
    two_qubit_gate_names = ["cx", "cz", "swap"]
    return sum(ops.get(g, 0) for g in two_qubit_gate_names)


def link_error_sum(routed, backend) -> float:
    """Expected number of link faults: sum of the link error of every 2q operation on an
    inter-chiplet link (a SWAP gets one DEPOLARIZE2 in the noise model, so it counts once).
    This is the quantity the cost routers trade against extra SWAPs."""
    remote = symmetric_remote_noise(backend)
    total = 0.0
    for ins in routed.data:
        if ins.operation.num_qubits == 2 and ins.operation.name != "barrier":
            q = tuple(routed.find_bit(x).index for x in ins.qubits)
            total += remote.get(q, 0.0)
    return total


def plot_metric(results: dict, metric: str, ylabel: str, title: str, filename: str, logy: bool = False) -> None:
    """Grouped bars: one group per interconnection density, one bar per (router, noise level)."""
    nis = sorted({ni for m in results for ni in results[m]["low"]}, reverse=True)
    methods = [m for m, *_ in METHODS if m in results]
    x = np.arange(len(nis))
    n_bars = 2 * len(methods)
    width = 0.8 / n_bars

    _panel_style()
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    fig.subplots_adjust(**MARGINS)

    i = 0
    for level in ("low", "high"):
        for m, label, c_low, c_high in METHODS:
            if m not in results:
                continue
            vals = [results[m][level][ni][metric] for ni in nis]
            ax.bar(
                x + (i - (n_bars - 1) / 2) * width,
                vals,
                width,
                label=f"{label}, " + r"$p_{inter}$ = " + NOISE_LEVELS[level]["text"],
                color=c_low if level == "low" else c_high,
                hatch=HATCHES[level],
                edgecolor="black",
                linewidth=0.4,
            )
            i += 1

    ax.set_xticks(x)
    ax.set_xticklabels([SECTION_TITLES[j] if j < len(SECTION_TITLES) else str(ni) for j, ni in enumerate(nis)])
    ax.set_xlabel("Interconnection density")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    _panel_title(ax, title, "Lower is better ↓")
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    fig.savefig(filename, format="pdf")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Compressed legend: one entry per router, its handle split into the light (low p_inter) and the dark
# (high p_inter) bar style, plus two grey entries that explain light/dark (and their hatches).
# --------------------------------------------------------------------------------------


class SplitPatch:
    """Legend proxy: a rectangle whose left and right halves have different styles (Rectangle kwargs)."""

    def __init__(self, left: dict, right: dict):
        self.left, self.right = left, right


class HandlerSplitPatch(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        half = width / 2
        return [Rectangle((-xdescent + i * half, -ydescent), half, height, transform=trans, **style)
                for i, style in enumerate((orig_handle.left, orig_handle.right))]


def save_legend(results: dict, filename: str) -> None:
    _panel_style()

    def style(color, level):
        return dict(facecolor=color, hatch=HATCHES[level], edgecolor="black", linewidth=0.4)

    methods = [(label, c_low, c_high) for m, label, c_low, c_high in METHODS if m in results]
    handles = [SplitPatch(style(c_low, "low"), style(c_high, "high")) for _, c_low, c_high in methods]
    labels = [label for label, _, _ in methods]
    # Shade / hatch key
    handles += [Patch(**style("#E6E6E6", "low")), Patch(**style("#8C8C8C", "high"))]
    labels += [r"Light: $p_{inter}$ = " + NOISE_LEVELS["low"]["text"],
               r"Dark: $p_{inter}$ = " + NOISE_LEVELS["high"]["text"]]

    legend_fig = plt.figure(figsize=(3, 2))
    legend_fig.legend(handles, labels, handler_map={SplitPatch: HandlerSplitPatch()}, loc="center",
                      frameon=False, ncols=int(np.ceil(len(handles) / 2)), columnspacing=1.5, handlelength=2.4)
    legend_fig.savefig(filename + "legend.pdf", bbox_inches="tight", format="pdf")
    plt.close(legend_fig)


def plot_combined(results: dict, filename: str) -> None:
    # Short titles: the panel is 1.75 in wide (details in the caption)
    plot_metric(results, "depth", "Depth overhead", "a) Circuit depth", f"{filename}_depth.pdf")
    plot_metric(results, "overhead", "#2q gate overhead", "a) #2q gates", f"{filename}_overhead.pdf")
    plot_metric(results, "link_error_sum", "Expected link faults", "a) Link errors", f"{filename}_link_errors.pdf",
                logy=True)
    plot_metric(results, "routing_s", "Routing time [s]", "a) Routing time", f"{filename}_runtime.pdf", logy=True)
    save_legend(results, filename)


def print_summary(results: dict) -> None:
    nis = sorted({ni for m in results for ni in results[m]["low"]}, reverse=True)
    for level in ("low", "high"):
        print(f"\np_inter = {NOISE_LEVELS[level]['p_inter']}")
        print(f"{'router':14s} {'#links':>6s} {'depth':>7s} {'2q ovh':>7s} {'link faults':>11s} {'time [s]':>9s}")
        for m, *_ in METHODS:
            if m not in results:
                continue
            for ni in nis:
                r = results[m][level][ni]
                print(f"{LABEL[m]:14s} {ni:6d} {r['depth']:7d} {r['overhead']:7d} {r['link_error_sum']:11.3f} "
                      f"{r['routing_s']:9.2f}")


def run_exp_inter_chiplet(reproduce: bool = False, quick: bool = False) -> None:
    # Number of circuits
    np_ = 1
    # Number of interconnects
    num_inter_chiplet_connections = [8, 4, 1]
    # Code distance of surface code
    ks = 2
    chiplet_size = (19, 10)
    if quick:  # smoke test only
        num_inter_chiplet_connections, ks, chiplet_size = [4, 2, 1], 1, (11, 6)

    # Backend seed (best of the original iterations)
    iter_c = 0

    output_dir = OUTPUT_DIR / ("quick" if quick else "")
    if reproduce:
        (output_dir / "layout_utilization").mkdir(parents=True, exist_ok=True)
        # Generate circuit
        with quiet():
            circuit, partitions = get_tqec_cnot_rotated(distance_scale=ks, n1=np_, n2=0)
        # Stim to qiskit
        stim_code_circuit = StimCodeCircuit(stim_circuit=circuit)
        qc = stim_code_circuit.qc
        qc_depth, qc_2q = qc.depth(), num_2q_gates(qc)
        print(f"Input circuit: {qc_2q} two-qubit gates, depth {qc_depth}")

        # results[method][level][ni] = {"depth", "overhead", "link_error_sum", "routing_s", ...}
        results = {m: {"low": {}, "high": {}} for m, *_ in METHODS}

        for ni in num_inter_chiplet_connections:
            for level, nl in NOISE_LEVELS.items():
                backend = BackendChipletV2(
                    size=(np_ * 2, np_ * 2, *chiplet_size),
                    n_inter=ni,
                    connectivity="nn",
                    topology="rotated_grid",
                    inter_chiplet_noise=nl["p_inter"],
                    inter_chiplet_amplification=1,
                    inter_chiplet_noise_type="random",
                    num_defective_qubits=0,
                    rng_seed=iter_c,
                )

                # One mapping, shared by all routers
                mapping, t_map = chipmunq_mapping(qc, backend, partitions)

                routed = {}
                for m, (a_factor, beta) in COST_CONFIGS.items():
                    routed[m] = route_from_mapping(
                        qc, backend, mapping, m, nl["p_inter"],
                        alpha=nl["alpha"](a_factor), beta=beta,
                    )
                budget = max(routed[m][1]["routing_s"] for m in COST_CONFIGS)
                routed["sabre"] = route_from_mapping(qc, backend, mapping, "sabre", nl["p_inter"], seed=iter_c,
                                                     budget_s=budget)
                routed["murali"] = route_from_mapping(qc, backend, mapping, "murali", nl["p_inter"])
                routed["seqc"] = route_from_mapping(qc, backend, mapping, "seqc", nl["p_inter"], seed=iter_c)

                for m, (out, info) in routed.items():
                    # still a valid QEC experiment on this backend (SEQC re-attaches its detectors itself)
                    check_routed(out, circuit, backend, stim_circuit=info.get("stim"))
                    stats = overhead_stats(out, qc, backend)
                    results[m][level][ni] = {
                        # Depth without Stim annotations (DETECTOR/QUBIT_COORDS/... sit on qubit 0 of the input and
                        # inflate its depth; SEQC's output carries none of them, so the original
                        # ``out.depth() - qc.depth()`` is not comparable across routers). Old value kept below.
                        "depth": stats["depth_overhead"],
                        "depth_with_annotations": out.depth() - qc_depth,
                        # Original 2q metric of this experiment (every cx/cz/swap counts once)
                        "overhead": num_2q_gates(out) - qc_2q,
                        # Added
                        "link_error_sum": link_error_sum(out, backend),
                        "routing_s": info["routing_s"],
                        "mapping_s": t_map,
                        "trials": info.get("trials"),
                        "budget_s": budget,
                        **stats,
                    }
                    if m in COST_CONFIGS and level == "low":
                        try:
                            plot_circuit_layout_utilization(
                                out, backend,
                                filename=str(output_dir / "layout_utilization" / f"mapping_{m}_{ni}.png"),
                            )
                        except Exception as e:  # visualisation only
                            print(f"layout plot skipped: {e}")
                print(f"[links {ni}, {level}] budget {budget * 1e3:.0f} ms | " + ", ".join(
                    f"{LABEL[m]} {results[m][level][ni]['overhead']} 2q / {results[m][level][ni]['link_error_sum']:.3f}"
                    for m in routed))

        # Store values for evaluation. The per-config pickles keep the original format
        # ({ni: {ks: value}}) so older plotting code can still read them.
        output_dir.mkdir(parents=True, exist_ok=True)
        for m in results:
            for level, name in (("low", "low_error"), ("high", "high_error")):
                for metric, fname in (("depth", "depth"), ("overhead", "overhead")):
                    with open(output_dir / f"{name}_{fname}_{m}.pkl", "wb") as f:
                        pickle.dump({ni: {ks: v[metric]} for ni, v in results[m][level].items()}, f)
        with open(output_dir / "inter_chiplet_results.pkl", "wb") as f:
            pickle.dump(results, f)

    with open(output_dir / "inter_chiplet_results.pkl", "rb") as f:
        results = pickle.load(f)

    print_summary(results)
    plot_combined(results, str(output_dir / "cnot_inter_chiplet_overhead"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="d=3 on small chiplets (smoke test)")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    run_exp_inter_chiplet(reproduce=not a.plot_only, quick=a.quick)