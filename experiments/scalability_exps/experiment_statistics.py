from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.getcwd(), "."))

from experiments.exp_utils.circuit_generator import get_tqec_cnot_rotated
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.utils import *
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit

# Plotting
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from pathlib import Path


def plot_combined_split(
    custom_depth, custom_overhead, sabre_depth, sabre_overhead, depth_overall, gate_overall, filename: str = ""
):

    pastel_blue = "#A7D9ED"
    pastel_orange = "#F7C6A2"

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
        # Line and marker styles
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "lines.markeredgewidth": 1.5,
        "lines.markeredgecolor": "black",
        # Error bar cap size
        "errorbar.capsize": 3,
    }

    plt.rcParams.update(tex_fonts)

    ks = list(next(iter(custom_depth.values())).keys())[0]
    np_values = sorted(custom_depth.keys())

    section_titles = ["Small", "Medium", "Big"]

    # Extract values
    custom_depth_vals = [depth_overall[np][ks] + custom_depth[np][ks] for np in np_values]
    sabre_depth_vals = [depth_overall[np][ks] + sabre_depth[np][ks] for np in np_values]
    general_depth_vals = [depth_overall[np][ks] for np in np_values]

    custom_over_vals = [gate_overall[np][ks] + custom_overhead[np][ks] for np in np_values]
    sabre_over_vals = [gate_overall[np][ks] + sabre_overhead[np][ks] for np in np_values]
    general_over_vals = [gate_overall[np][ks] for np in np_values]

    # print(custom_depth_vals)
    overhead_ours = []
    overhead_sabre = []
    overhead_gates_ours = []
    overhead_gates_sabre = []
    for i in range(3):
        print((custom_depth_vals[i] - general_depth_vals[i]) / (sabre_depth_vals[i] - general_depth_vals[i]))
        print(general_depth_vals[i])
        print((custom_over_vals[i] - general_over_vals[i]) / (sabre_over_vals[i] - general_over_vals[i]))
        print(general_over_vals[i])

        overhead_ours.append(f"{100 * ((custom_depth_vals[i] - general_depth_vals[i]) / general_depth_vals[i]):.0f}%")
        # overhead_ours.append("{:.1f}x".format(((custom_depth_vals[i] - general_depth_vals[i])/general_depth_vals[i])))

        overhead_sabre.append(f"{100 * ((sabre_depth_vals[i] - general_depth_vals[i]) / general_depth_vals[i]):.0f}%")
        overhead_gates_ours.append(
            f"{100 * ((custom_over_vals[i] - general_over_vals[i]) / general_over_vals[i]):.0f}%"
        )
        overhead_gates_sabre.append(
            f"{100 * ((sabre_over_vals[i] - general_over_vals[i]) / general_over_vals[i]):.0f}%"
        )

        # Depth: 1 - (0.17329910141206675 + 0.12664165103189493 + 0.10834670947030497)/3 = 0.8639041793619111
        # 2q_gates: 1 - (0.09606831524639743 + 0.08065720687079911 + 0.08100405020251013)/3 = 0.9140901425600978

    x = np.arange(len(np_values))
    width = 0.25  # 0.35

    # Create depth statistics
    # fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.5, WIDTH_FIGSIZE*0.92))
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE * 2.5, WIDTH_FIGSIZE * 0.5))

    ax.bar(x - width, general_depth_vals, width, label="Ideal", color="lightcoral", hatch="//", edgecolor="black")

    bars_ours = ax.bar(x, custom_depth_vals, width, label="Chipmunq", color=pastel_blue, hatch="/", edgecolor="black")

    bars_sabre = ax.bar(
        x + width, sabre_depth_vals, width, label="LightSABRE", color=pastel_orange, hatch="o", edgecolor="black"
    )

    # ax.bar_label(bars_ours, labels = overhead_ours, padding = 9, rotation = 90)
    # ax.bar_label(bars_sabre, labels = overhead_sabre, padding = 9, rotation = 90)

    ax.set_xticks(x)
    ax.set_xticklabels(section_titles)
    ax.set_xlabel("Circuit size")
    ax.set_ylabel("Circuit depth")
    # ax.legend(loc='upper left')

    # Add annotation
    # ax.text(0.53, 1.04, 'Lower is better ↓',
    #        transform=ax.transAxes,
    #        #fontsize=10,
    #        fontweight='bold',
    #        color = plot_lib_color,
    #        va='top',
    #        ha='left')

    ax.text(-0.18, 1.05, "b) Compilation overhead on circuit depth", transform=ax.transAxes, fontweight="bold")

    ax.text(
        0.28,
        1.17,
        "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    ax.set_ylim(0, 15500)

    # Arrow and text for first bars
    ax.text(
        0.16,
        0.22,
        "-5.5x",
        transform=ax.transAxes,
        color="green",
    )
    plt.annotate(
        "",
        xy=(-0.04, 1500),
        xytext=(0.3, 2500),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.6", color="green", lw=1.5),
    )

    # Arrow and text for second bars
    ax.text(
        0.5,
        0.4,
        "-9x",
        transform=ax.transAxes,
        color="green",
    )
    plt.annotate(
        "",  # No text
        xy=(1, 4000),  # Tip: Pointing to the bottom
        xytext=(1.3, 5500),  # Base: Starting at the top
        arrowprops=dict(
            arrowstyle="->",
            connectionstyle="arc3,rad=.6",  # Positive = curve up/left
            color="green",
            lw=1.5,
        ),
    )

    # Arrow and text for third bars
    ax.text(
        0.84,
        0.66,
        "-8x",
        transform=ax.transAxes,
        color="green",
    )
    plt.annotate(
        "",  # No text
        xy=(2, 8000),  # Tip: Pointing to the bottom
        xytext=(2.3, 9500),  # Base: Starting at the top
        arrowprops=dict(
            arrowstyle="->",
            connectionstyle="arc3,rad=.6",  # Positive = curve up/left
            color="green",
            lw=1.5,
        ),
    )

    # fig.tight_layout()
    # fig.subplots_adjust(left=0.24, right=0.95, top=0.9, bottom=0.12)
    plt.grid(True, which="major", linestyle="--", alpha=0.5)
    fig.subplots_adjust(left=0.24, right=0.95, top=0.83, bottom=0.21)
    fig.savefig(f"{filename}_depth.pdf", format="pdf")
    plt.close(fig)

    tex_fonts = {
        # Use LaTeX to write all text
        # "text.usetex": True,
        "font.family": "serif",
        # Font sizes
        "axes.labelsize": FONTSIZE * 1.3,
        "font.size": FONTSIZE * 1.2,
        "legend.fontsize": (FONTSIZE - 2) * 1.3,
        "xtick.labelsize": (FONTSIZE - 1) * 1.3,
        "ytick.labelsize": (FONTSIZE - 1) * 1.3,
        "axes.titlesize": 10,
        # Line and marker styles
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "lines.markeredgewidth": 1.5,
        "lines.markeredgecolor": "black",
        # Error bar cap size
        "errorbar.capsize": 3,
    }
    plt.rcParams.update(tex_fonts)

    # Calculate the maximum value across all bars for the upper y-limit
    max_overall_val = max(max(general_over_vals), max(custom_over_vals), max(sabre_over_vals))

    # Add a small padding to the upper y-limit for better visualization
    upper_ylim = 66000  # max_overall_val*1.01 # 1.015

    # fig = plt.figure(figsize=(HEIGHT_FIGSIZE*2.5, WIDTH_FIGSIZE*0.92))
    fig = plt.figure(figsize=(HEIGHT_FIGSIZE * 2.5, WIDTH_FIGSIZE * 0.5))

    gs = gridspec.GridSpec(2, 1, height_ratios=[12, 20], hspace=0.1)  # Swapped height_ratios

    # Top subplot (for values above the break, e.g., 60,000 to max)
    ax_top = plt.subplot(gs[0])  # Now gs[0] for the top part
    bars_ideal = ax_top.bar(
        x - width, general_over_vals, width, label="Ideal", color="lightcoral", hatch="//", edgecolor="black"
    )
    ax_top.bar(x, custom_over_vals, width, label="Chipmunq", color=pastel_blue, hatch="/", edgecolor="black")
    ax_top.bar(x + width, sabre_over_vals, width, label="LightSABRE", color=pastel_orange, hatch="o", edgecolor="black")

    # Limit for upper
    ax_top.set_ylim(66000, upper_ylim)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([])
    ax_top.tick_params(axis="y", length=5)
    plt.grid(True, which="major", linestyle="--", alpha=0.5)

    # Bottom subplot (for values below the break, e.g., 0 to 40,000)
    ax_bottom = plt.subplot(gs[1])  # Now gs[1] for the bottom part
    ax_bottom.bar(x - width, general_over_vals, width, label="Ideal", color="lightcoral", hatch="/", edgecolor="black")
    bars_gates_ours = ax_bottom.bar(
        x, custom_over_vals, width, label="Chipmunq", color=pastel_blue, hatch="/", edgecolor="black"
    )
    bars_gates_sabre = ax_bottom.bar(
        x + width, sabre_over_vals, width, label="LightSABRE", color=pastel_orange, hatch="o", edgecolor="black"
    )

    # ax_top.bar_label(bars_gates_ours, labels = overhead_gates_ours, padding = 9, rotation = 90)
    # ax_bottom.bar_label(bars_gates_ours, labels = overhead_gates_ours, padding = 9, rotation = 90)
    # ax_top.bar_label(bars_gates_sabre, labels = overhead_gates_sabre, padding = 9, rotation = 90)
    # ax_bottom.bar_label(bars_gates_sabre, labels = overhead_gates_sabre, padding = 9, rotation = 90)

    # Limit for bottom
    ax_bottom.set_ylim(0, 40000)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(section_titles)
    ax_bottom.tick_params(axis="y", length=5)

    ax_bottom.spines["top"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)

    ax_top.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax_bottom.tick_params(axis="x", which="both", top=False)

    # Size of horizontal lines of the split
    d = 0.015

    kwargs = dict(transform=ax_bottom.transAxes, color="k", clip_on=False, linewidth=1.5)
    # Bottom-left split
    ax_bottom.plot((-d, +d), (1, 1), **kwargs)
    # Bottom-right split
    ax_bottom.plot((1 - d, 1 + d), (1, 1), **kwargs)

    kwargs.update(transform=ax_top.transAxes)
    # Top-left split
    ax_top.plot((-d, +d), (0, 0), **kwargs)
    # Top-right split
    ax_top.plot((1 - d, 1 + d), (0, 0), **kwargs)

    # Arrow and text for first bars
    ax_bottom.text(
        0.16,
        0.4,
        "-10x",
        transform=ax_bottom.transAxes,
        color="green",
    )
    plt.annotate(
        "",
        xy=(-0.04, 4000),
        xytext=(0.3, 12000),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.6", color="green", lw=1.5),
    )

    # Arrow and text for second bars
    ax_bottom.text(
        0.39,
        0.8,
        "-12x",
        transform=ax_bottom.transAxes,
        color="green",
    )
    plt.annotate(
        "",
        xy=(0.95, 15000),
        xytext=(1.25, 38000),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.4", color="green", lw=1.5),
    )

    # Arrow and text for third bars
    ax_bottom.text(
        0.72,
        1.3,
        "-12x",
        transform=ax_bottom.transAxes,
        color="green",
    )
    plt.annotate(
        "",
        xy=(1.95, 30000),
        xytext=(2.25, 59000),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.4", color="green", lw=1.5),
    )

    fig.text(0.025, 0.5, "#2q gates", va="center", rotation="vertical", fontsize=FONTSIZE * 1.5)
    fig.text(0.46, 0.055, "Circuit size", va="center", rotation="horizontal", fontsize=FONTSIZE * 1.5)
    ax_top.text(
        0.24,
        1.6,
        "Lower is better ↓",
        transform=ax_top.transAxes,
        # fontsize=FONTSIZE,
        fontweight="bold",
        color=plot_lib_color,
        va="top",
        ha="left",
    )

    ax_top.text(
        -0.075,
        1.3,
        "c) Compilation overhead on #2q gates",
        transform=ax_top.transAxes,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # ax_top.legend(loc='upper left')

    # fig.tight_layout()
    # fig.subplots_adjust(left=0.24, right=0.95, top=0.95, bottom=0.07)
    # fig.subplots_adjust(left=0.24, right=0.95, top=0.9, bottom=0.12)
    plt.grid(True, which="major", linestyle="--", alpha=0.5)
    fig.subplots_adjust(left=0.24, right=0.95, top=0.83, bottom=0.21)
    fig.savefig(f"{filename}_overhead.pdf", format="pdf")
    plt.close(fig)

    legend_fig = plt.figure(figsize=(3, 2))
    legend = legend_fig.legend(
        handles=[bars_ideal, bars_gates_ours, bars_gates_sabre], loc="center", frameon=False, ncols=3, columnspacing=1.5
    )
    legend_fig.savefig(filename + "legend.pdf", bbox_inches="tight", format="pdf")
    plt.close(legend_fig)


def run_exp_statistics(reproduce: bool = False) -> None:

    # Backend configuration
    num_inter_chiplet_connections = 8
    ps_inter = 1e-4
    n_patches = [1, 3, 6]

    custom_depth = {}
    custom_overhead = {}
    sabre_depth = {}
    sabre_overhead = {}
    depth_overall = {}
    gate_overall = {}

    if reproduce:
        for ks in [2]:  # [1, 2, 3, 4]
            for np in n_patches:
                if np not in custom_depth:
                    custom_depth[np] = {}
                    custom_overhead[np] = {}
                    sabre_depth[np] = {}
                    sabre_overhead[np] = {}

                    depth_overall[np] = {}
                    gate_overall[np] = {}

                # Generate circuit
                circuit, partitions = get_tqec_cnot_rotated(distance_scale=ks, n1=np, n2=0)

                backend = BackendChipletV2(
                    size=(np * 2, np * 2, 15, 8),
                    n_inter=num_inter_chiplet_connections,
                    connectivity="nn",
                    topology="rotated_grid",
                    inter_chiplet_noise=ps_inter,
                    inter_chiplet_amplification=1,
                    inter_chiplet_noise_type="constant",
                    num_defective_qubits=0,
                )

                # Stim to qiskit
                stim_code_circuit = StimCodeCircuit(stim_circuit=circuit)

                # Custom transpilation
                print("Custom")
                custom_circuit = custom_partitioned_transpilation(
                    stim_code_circuit.qc, backend, pre_defined_partitions=partitions
                )

                # Sabre transpilation
                print("Sabre")
                sabre_circuit = sabre_transpilation(stim_code_circuit.qc, backend)

                def num_2q_gates(circuit):
                    ops = circuit.count_ops()
                    two_qubit_gate_names = ["cx", "cz", "swap"]
                    return sum(ops.get(g, 0) for g in two_qubit_gate_names)

                custom_depth[np][ks] = custom_circuit.depth() - (stim_code_circuit.qc).depth()
                custom_overhead[np][ks] = num_2q_gates(custom_circuit) - num_2q_gates(stim_code_circuit.qc)

                sabre_depth[np][ks] = sabre_circuit.depth() - (stim_code_circuit.qc).depth()
                sabre_overhead[np][ks] = num_2q_gates(sabre_circuit) - num_2q_gates(stim_code_circuit.qc)

                depth_overall[np][ks] = (stim_code_circuit.qc).depth()
                gate_overall[np][ks] = num_2q_gates(stim_code_circuit.qc)

        # Save data
        output_dir = Path("experiments/evaluation/scalability")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "custom_depth.pkl", "wb") as f:
            pickle.dump(custom_depth, f)
        with open(output_dir / "custom_overhead.pkl", "wb") as f:
            pickle.dump(custom_overhead, f)
        with open(output_dir / "sabre_depth.pkl", "wb") as f:
            pickle.dump(sabre_depth, f)
        with open(output_dir / "sabre_overhead.pkl", "wb") as f:
            pickle.dump(sabre_overhead, f)
        with open(output_dir / "depth_overall.pkl", "wb") as f:
            pickle.dump(depth_overall, f)
        with open(output_dir / "gate_overall.pkl", "wb") as f:
            pickle.dump(gate_overall, f)

    # Load files
    with open("experiments/evaluation/scalability/custom_depth.pkl", "rb") as f:
        custom_depth = pickle.load(f)
    with open("experiments/evaluation/scalability/custom_overhead.pkl", "rb") as f:
        custom_overhead = pickle.load(f)
    with open("experiments/evaluation/scalability/sabre_depth.pkl", "rb") as f:
        sabre_depth = pickle.load(f)
    with open("experiments/evaluation/scalability/sabre_overhead.pkl", "rb") as f:
        sabre_overhead = pickle.load(f)
    with open("experiments/evaluation/scalability/depth_overall.pkl", "rb") as f:
        depth_overall = pickle.load(f)
    with open("experiments/evaluation/scalability/gate_overall.pkl", "rb") as f:
        gate_overall = pickle.load(f)

    plot_combined_split(
        custom_depth,
        custom_overhead,
        sabre_depth,
        sabre_overhead,
        depth_overall,
        gate_overall,
        "experiments/evaluation/scalability/cnot_scaling_overhead_split",
    )


if __name__ == "__main__":
    run_exp_statistics(reproduce=True)
