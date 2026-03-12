from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.getcwd(), "."))

from experiments.exp_utils.circuit_generator import get_tqec_cnot_rotated
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.utils import *
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit
from qeccm.backends.backend_utils import plot_circuit_layout_utilization

# Plotting
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_combined(low_depth, low_overhead, high_depth, high_overhead, filename: str = ""):
    ks = list(next(iter(low_depth[0].values())).keys())[0]
    np_values = [8, 4, 1]  # low_depth.keys()#sorted(low_depth.keys())

    section_titles = ["Full", "Half", "Limited"]

    # Extract values
    low_depth_vals = [low_depth[0][np][ks] for np in np_values]
    high_depth_vals = [high_depth[0][np][ks] for np in np_values]
    low_over_vals = [low_overhead[0][np][ks] for np in np_values]
    high_over_vals = [high_overhead[0][np][ks] for np in np_values]
    print(low_over_vals)

    low_depth_vals_tradeoff = [low_depth[1][np][ks] for np in np_values]
    high_depth_vals_tradeoff = [high_depth[1][np][ks] for np in np_values]
    low_over_vals_tradeoff = [low_overhead[1][np][ks] for np in np_values]
    high_over_vals_tradeoff = [high_overhead[1][np][ks] for np in np_values]
    print(np.array(low_over_vals) / np.array(low_over_vals_tradeoff))
    print(np.array(high_over_vals) / np.array(high_over_vals_tradeoff))

    low_depth_vals_focus = [low_depth[2][np][ks] for np in np_values]
    high_depth_vals_focus = [high_depth[2][np][ks] for np in np_values]
    low_over_vals_focus = [low_overhead[2][np][ks] for np in np_values]
    high_over_vals_focus = [high_overhead[2][np][ks] for np in np_values]
    print(np.array(low_over_vals) / np.array(low_over_vals_focus))
    print(np.array(high_over_vals) / np.array(high_over_vals_focus))

    x = np.arange(len(np_values))
    width = 0.35 / 3

    colors = ["#A7D9ED", "#F7C6A2", "#4682B4", "#F08080"]
    hatches = [
        "o",
        "xx",
    ]

    tex_fonts = {
        # Use LaTeX to write all text
        # "text.usetex": True,
        "font.family": "serif",
        # Font sizes
        "axes.labelsize": FONTSIZE * 1.5,
        "font.size": FONTSIZE * 1.2,
        "legend.fontsize": (FONTSIZE - 2) * 1.3,
        "xtick.labelsize": (FONTSIZE - 1) * 1.3,
        "ytick.labelsize": (FONTSIZE - 1) * 1.3,
        "axes.titlesize": 10,
        # Hatches
        "hatch.linewidth": 0.5,
        # Line and marker styles
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "lines.markeredgewidth": 1.5,
        "lines.markeredgecolor": "black",
        # Error bar cap size
        "errorbar.capsize": 3,
    }

    plt.rcParams.update(tex_fonts)

    # Pastel colors
    pastel_blue = "#A7D9ED"
    pastel_blue_dark = "#5B9BD5"
    pastel_orange = "#F7C6A2"
    pastel_orange_dark = "#E68A5C"
    pastel_red = "#F7A2A2"
    pastel_red_dark = "#D65C5C"

    # Create depth statistics
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE * 2.5, WIDTH_FIGSIZE * 0.5))

    handles = []
    # Basic
    h = ax.bar(
        x - 2.5 * width,
        low_depth_vals,
        width,
        label=r"Basic, $p_{inter}$ = $1e^{-4}$",
        color=pastel_blue,
        hatch=hatches[0],
        edgecolor="black",
    )
    handles.append(h)
    h = ax.bar(
        x + 0.5 * width,
        high_depth_vals,
        width,
        label=r"Basic, $p_{inter}$ = $1e^{-2}$",
        color=pastel_blue_dark,
        hatch=hatches[1],
        edgecolor="black",
    )
    handles.append(h)
    # Tradeoff
    h = ax.bar(
        x - 1.5 * width,
        low_depth_vals_tradeoff,
        width,
        label=r"Tradeoff, $p_{inter}$ = $1e^{-4}$",
        color=pastel_orange,
        hatch=hatches[0],
        edgecolor="black",
    )
    handles.append(h)
    h = ax.bar(
        x + 1.5 * width,
        high_depth_vals_tradeoff,
        width,
        label=r"Tradeoff, $p_{inter}$ = $1e^{-2}$",
        color=pastel_orange_dark,
        hatch=hatches[1],
        edgecolor="black",
    )
    handles.append(h)
    # Focus
    h = ax.bar(
        x - 0.5 * width,
        low_depth_vals_focus,
        width,
        label=r"Focus, $p_{inter}$ = $1e^{-4}$",
        color=pastel_red,
        hatch=hatches[0],
        edgecolor="black",
    )
    handles.append(h)
    h = ax.bar(
        x + 2.5 * width,
        high_depth_vals_focus,
        width,
        label=r"Focus, $p_{inter}$ = $1e^{-2}$",
        color=pastel_red_dark,
        hatch=hatches[1],
        edgecolor="black",
    )
    handles.append(h)

    ax.set_xticks(x)
    ax.set_xticklabels(section_titles)
    ax.set_xlabel("Interconnection density")
    ax.set_ylabel("Depth Overhead")
    # ax.legend()

    # Add annotation

    ax.text(-0.1, 1.04, "a) Connectivity affecting circuit depth", transform=ax.transAxes, fontweight="bold")

    ax.text(
        0.3,
        1.14,
        "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.subplots_adjust(left=0.22, right=0.95, top=0.85, bottom=0.21)
    fig.savefig(f"{filename}_depth.pdf", format="pdf")
    plt.close(fig)

    # Create 2q gate overhead
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE * 2.5, WIDTH_FIGSIZE * 0.5))

    handles = []

    # Basic
    h = ax.bar(
        x - 2.5 * width,
        low_over_vals,
        width,
        label=r"Basic, $p_{inter}$ = $1e^{-4}$",
        color=pastel_blue,
        hatch=hatches[0],
        edgecolor="black",
    )
    handles.append(h)
    h = ax.bar(
        x + 0.5 * width,
        high_over_vals,
        width,
        label=r"Basic, $p_{inter}$ = $1e^{-2}$",
        color=pastel_blue_dark,
        hatch=hatches[1],
        edgecolor="black",
    )
    handles.append(h)
    # Tradeoff
    h = ax.bar(
        x - 1.5 * width,
        low_over_vals_tradeoff,
        width,
        label=r"Tradeoff, $p_{inter}$ = $1e^{-4}$",
        color=pastel_orange,
        hatch=hatches[0],
        edgecolor="black",
    )
    handles.append(h)
    h = ax.bar(
        x + 1.5 * width,
        high_over_vals_tradeoff,
        width,
        label=r"Tradeoff, $p_{inter}$ = $1e^{-2}$",
        color=pastel_orange_dark,
        hatch=hatches[1],
        edgecolor="black",
    )
    handles.append(h)
    # Focus
    h = ax.bar(
        x - 0.5 * width,
        low_over_vals_focus,
        width,
        label=r"Focus, $p_{inter}$ = $1e^{-4}$",
        color=pastel_red,
        hatch=hatches[0],
        edgecolor="black",
    )
    handles.append(h)
    h = ax.bar(
        x + 2.5 * width,
        high_over_vals_focus,
        width,
        label=r"Focus, $p_{inter}$ = $1e^{-2}$",
        color=pastel_red_dark,
        hatch=hatches[1],
        edgecolor="black",
    )
    handles.append(h)

    ax.set_xticks(x)
    ax.set_xticklabels(section_titles)
    ax.set_xlabel("Interconnection density")
    ax.set_ylabel("#2q gate overhead ")
    # ax.legend()
    # ax.set_ylim(0, 6100)

    # Add annotation
    # ax.text(0.57, 1.04, "Lower is better ↓",
    #    transform=ax.transAxes,
    #    #fontsize=10,
    #    fontweight='bold',
    #    color = plot_lib_color,
    #    va='top',
    #    ha='left')

    ax.text(-0.05, 1.04, "a) Effect of cost-routing on #2q gates", transform=ax.transAxes, fontweight="bold")

    ax.text(
        0.3,
        1.14,
        "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    # fig.tight_layout()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.subplots_adjust(left=0.22, right=0.95, top=0.85, bottom=0.21)
    fig.savefig(f"{filename}_overhead.pdf", format="pdf")
    plt.close(fig)

    legend_fig = plt.figure(figsize=(3, 2))
    legend = legend_fig.legend(handles=handles, loc="center", frameon=False, ncols=3, columnspacing=1.5)
    legend_fig.savefig(filename + "legend.pdf", bbox_inches="tight", format="pdf")
    plt.close(legend_fig)


def run_exp_inter_chiplet(reproduce: bool = False) -> None:

    low_error_depth = {}
    low_error_overhead = {}
    high_error_depth = {}
    high_error_overhead = {}

    # Number of circuits
    np = 1
    # Number of interconnects
    num_inter_chiplet_connections = [8, 4, 1]

    # Code distance of surface code
    ks = 3

    # Cost routing configurations
    # Focus
    config_focus = [6, 1]
    # Tradeoff
    config_tradeoff = [1, 1]
    # Basic
    config_basic = [0, 0]

    if reproduce:
        # Generate circuit
        circuit, partitions = get_tqec_cnot_rotated(distance_scale=ks, n1=np, n2=0)

        for config in ["basic", "tradeoff", "focus"]:
            for ni in num_inter_chiplet_connections:
                iter_c = 0  # 1 best

                if ni not in low_error_depth:
                    low_error_depth[ni] = {}
                    low_error_overhead[ni] = {}
                    high_error_depth[ni] = {}
                    high_error_overhead[ni] = {}

                if ks not in low_error_depth[ni]:
                    low_error_depth[ni][ks] = 0
                    low_error_overhead[ni][ks] = 0
                    high_error_depth[ni][ks] = 0
                    high_error_overhead[ni][ks] = 0

                backend_1e4 = BackendChipletV2(
                    size=(np * 2, np * 2, 19, 10),
                    n_inter=ni,
                    connectivity="nn",
                    topology="rotated_grid",
                    inter_chiplet_noise=1e-4,
                    inter_chiplet_amplification=1,
                    inter_chiplet_noise_type="random",
                    num_defective_qubits=0,
                    rng_seed=iter_c,
                )

                backend_1e2 = BackendChipletV2(
                    size=(np * 2, np * 2, 19, 10),
                    n_inter=ni,
                    connectivity="nn",
                    topology="rotated_grid",
                    inter_chiplet_noise=1e-2,
                    inter_chiplet_amplification=1,
                    inter_chiplet_noise_type="random",
                    num_defective_qubits=0,
                    rng_seed=iter_c,
                )

                # Stim to qiskit
                stim_code_circuit = StimCodeCircuit(stim_circuit=circuit)

                if config == "basic":
                    cost_config = config_basic
                elif config == "tradeoff":
                    cost_config = config_tradeoff
                elif config == "focus":
                    cost_config = config_focus

                # Low error transpilation
                low_error_circuit = custom_cost_transpilation(
                    stim_code_circuit.qc,
                    backend_1e4,
                    pre_defined_partitions=partitions,
                    routing_alpha=cost_config[0] * 1e4,
                    routing_beta=cost_config[1],
                )

                # High error transpilation
                high_error_circuit = custom_cost_transpilation(
                    stim_code_circuit.qc,
                    backend_1e2,
                    pre_defined_partitions=partitions,
                    routing_alpha=1.5 * cost_config[0] * 1e2,
                    routing_beta=cost_config[1],
                )

                plot_circuit_layout_utilization(
                    low_error_circuit,
                    backend_1e4,
                    filename=f"experiments/evaluation/inter_chiplet/layout_utilization/mapping_{config}_{ni}.png",
                )

                def num_2q_gates(circuit):
                    ops = circuit.count_ops()
                    two_qubit_gate_names = ["cx", "cz", "swap"]
                    return sum(ops.get(g, 0) for g in two_qubit_gate_names)

                # Estimated two-qubit gates: 17120
                print(num_2q_gates(stim_code_circuit.qc))
                # Estimated depth: 5261
                print((stim_code_circuit.qc).depth())

                low_error_depth[ni][ks] = low_error_circuit.depth() - (stim_code_circuit.qc).depth()
                low_error_overhead[ni][ks] = num_2q_gates(low_error_circuit) - num_2q_gates(stim_code_circuit.qc)

                high_error_depth[ni][ks] = high_error_circuit.depth() - (stim_code_circuit.qc).depth()
                high_error_overhead[ni][ks] = num_2q_gates(high_error_circuit) - num_2q_gates(stim_code_circuit.qc)

            # Store values for evaluation
            output_dir = Path("experiments/evaluation/inter_chiplet")
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / f"low_error_depth_{config}.pkl", "wb") as f:
                pickle.dump(low_error_depth, f)
            with open(output_dir / f"low_error_overhead_{config}.pkl", "wb") as f:
                pickle.dump(low_error_overhead, f)
            with open(output_dir / f"high_error_depth_{config}.pkl", "wb") as f:
                pickle.dump(high_error_depth, f)
            with open(output_dir / f"high_error_overhead_{config}.pkl", "wb") as f:
                pickle.dump(high_error_overhead, f)

    # Load circuit statistics for the three runs
    # Basic
    with open("experiments/evaluation/inter_chiplet/low_error_depth_basic.pkl", "rb") as f:
        low_error_depth_basic = pickle.load(f)
    with open("experiments/evaluation/inter_chiplet/low_error_overhead_basic.pkl", "rb") as f:
        low_error_overhead_basic = pickle.load(f)
    with open("experiments/evaluation/inter_chiplet/high_error_depth_basic.pkl", "rb") as f:
        high_error_depth_basic = pickle.load(f)
    with open("experiments/evaluation/inter_chiplet/high_error_overhead_basic.pkl", "rb") as f:
        high_error_overhead_basic = pickle.load(f)

    # Tradeoff
    with open("experiments/evaluation/inter_chiplet/low_error_depth_tradeoff.pkl", "rb") as f:
        low_error_depth_tradeoff = pickle.load(f)
    with open("experiments/evaluation/inter_chiplet/low_error_overhead_tradeoff.pkl", "rb") as f:
        low_error_overhead_tradeoff = pickle.load(f)
    with open("experiments/evaluation/inter_chiplet/high_error_depth_tradeoff.pkl", "rb") as f:
        high_error_depth_tradeoff = pickle.load(f)
    with open("experiments/evaluation/inter_chiplet/high_error_overhead_tradeoff.pkl", "rb") as f:
        high_error_overhead_tradeoff = pickle.load(f)

    # Focus
    with open("experiments/evaluation/inter_chiplet/low_error_depth_focus.pkl", "rb") as f:
        low_error_depth_focus = pickle.load(f)
    with open("experiments/evaluation/inter_chiplet/low_error_overhead_focus.pkl", "rb") as f:
        low_error_overhead_focus = pickle.load(f)
    with open("experiments/evaluation/inter_chiplet/high_error_depth_focus.pkl", "rb") as f:
        high_error_depth_focus = pickle.load(f)
    with open("experiments/evaluation/inter_chiplet/high_error_overhead_focus.pkl", "rb") as f:
        high_error_overhead_focus = pickle.load(f)

    plot_combined(
        [low_error_depth_basic, low_error_depth_tradeoff, low_error_depth_focus],
        [low_error_overhead_basic, low_error_overhead_tradeoff, low_error_overhead_focus],
        [high_error_depth_basic, high_error_depth_tradeoff, high_error_depth_focus],
        [high_error_overhead_basic, high_error_overhead_tradeoff, high_error_overhead_focus],
        "experiments/evaluation/inter_chiplet/cnot_inter_chiplet_overhead",
    )


if __name__ == "__main__":
    run_exp_inter_chiplet(reproduce=True)
