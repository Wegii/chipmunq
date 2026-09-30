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
    ("sabre", "SABRE-SWAP", "#D9D9D9", "#7F7F7F"),
    ("murali", "Murali et al.", "#B2D8B2", "#5E9E5E"),
    ("seqc", "SEQC", "#C9B7E3", "#8E6BBF"),
]
LABEL = {m: l for m, l, _, _ in METHODS}

# Cost routing configurations [alpha factor, beta]
COST_CONFIGS = {"basic": [0, 0], "tradeoff": [1, 1], "focus": [6, 1]}

# Link-noise levels. alpha scales with 1 / p_inter exactly as in the original experiment.
NOISE_LEVELS = {
    "low": dict(p_inter=1e-4, alpha=lambda c: c * 1e4, text=r"$1e^{-4}$"),
    "high": dict(p_inter=1e-2, alpha=lambda c: 1.5 * c * 1e2, text=r"$1e^{-2}$"),
}

SECTION_TITLES = ["Full", "Half", "Limited"]


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


def plot_metric(results: dict, metric: str, ylabel: str, title: str, filename: str, logy: bool = False) -> list:
    """Grouped bars: one group per interconnection density, one bar per (router, noise level)."""
    nis = sorted({ni for m in results for ni in results[m]["low"]}, reverse=True)
    methods = [m for m, *_ in METHODS if m in results]
    x = np.arange(len(nis))
    n_bars = 2 * len(methods)
    width = 0.8 / n_bars
    hatches = {"low": "o", "high": "xx"}

    tex_fonts = {
        "font.family": "serif",
        "axes.labelsize": FONTSIZE * 1.5,
        "font.size": FONTSIZE * 1.2,
        "legend.fontsize": (FONTSIZE - 2) * 1.3,
        "xtick.labelsize": (FONTSIZE - 1) * 1.3,
        "ytick.labelsize": (FONTSIZE - 1) * 1.3,
        "axes.titlesize": 10,
        "hatch.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "lines.markeredgewidth": 1.5,
        "lines.markeredgecolor": "black",
        "errorbar.capsize": 3,
    }
    plt.rcParams.update(tex_fonts)
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE * 2.5, WIDTH_FIGSIZE * 0.5))

    handles, i = [], 0
    for level in ("low", "high"):
        for m, label, c_low, c_high in METHODS:
            if m not in results:
                continue
            vals = [results[m][level][ni][metric] for ni in nis]
            h = ax.bar(
                x + (i - (n_bars - 1) / 2) * width,
                vals,
                width,
                label=f"{label}, " + r"$p_{inter}$ = " + NOISE_LEVELS[level]["text"],
                color=c_low if level == "low" else c_high,
                hatch=hatches[level],
                edgecolor="black",
            )
            handles.append(h)
            i += 1

    ax.set_xticks(x)
    ax.set_xticklabels([SECTION_TITLES[j] if j < len(SECTION_TITLES) else str(ni) for j, ni in enumerate(nis)])
    ax.set_xlabel("Interconnection density")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.text(-0.1, 1.04, title, transform=ax.transAxes, fontweight="bold")
    ax.text(0.3, 1.14, "Lower is better ↓", transform=ax.transAxes, fontweight="bold", color=plot_lib_color)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.subplots_adjust(left=0.22, right=0.95, top=0.85, bottom=0.21)
    fig.savefig(filename, format="pdf")
    plt.close(fig)
    return handles


def plot_combined(results: dict, filename: str) -> None:
    plot_metric(results, "depth", "Depth Overhead", "a) Connectivity affecting circuit depth", f"{filename}_depth.pdf")
    plot_metric(results, "overhead", "#2q gate overhead ", "a) Effect of cost-routing on #2q gates",
                f"{filename}_overhead.pdf")
    plot_metric(results, "link_error_sum", "Expected link faults", "a) Link errors accumulated by routing",
                f"{filename}_link_errors.pdf", logy=True)
    handles = plot_metric(results, "routing_s", "Routing time [s]", "a) Routing time", f"{filename}_runtime.pdf",
                          logy=True)
    legend_fig = plt.figure(figsize=(3, 2))
    legend_fig.legend(handles=handles, loc="center", frameon=False, ncols=len(handles) // 2, columnspacing=1.5)
    legend_fig.savefig(filename + "legend.pdf", bbox_inches="tight", format="pdf")
    plt.close(legend_fig)


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