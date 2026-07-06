from __future__ import annotations

import os
import sys



sys.path.append(os.path.join(os.getcwd(), "."))

from experiments.exp_utils.circuit_generator import get_tqec_cnot_rotated, generate_gross_code
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.utils import *
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit

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

# Plotting
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from pathlib import Path


def plot_combined_split(
    custom_depth, custom_overhead, sabre_depth, sabre_overhead, mech_depth, mech_overhead, depth_overall, gate_overall, filename: str = ""
):

    pastel_blue = "#A7D9ED"
    pastel_orange = "#F7C6A2"
    pastel_green = "#B5D8B0"

    tex_fonts = {
        "font.family": "serif",
        "axes.labelsize": FONTSIZE * 1.5,
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

    plt.rcParams.update(tex_fonts)

    ks = list(next(iter(custom_depth.values())).keys())[0]
    np_values = sorted(custom_depth.keys())

    section_titles = ["Surface CNOT", "6x Surface CNOT", "Gross Code"]

    # Extract values
    custom_depth_vals = [depth_overall[np][ks] + custom_depth[np][ks] for np in np_values]
    sabre_depth_vals = [depth_overall[np][ks] + sabre_depth[np][ks] for np in np_values]
    mech_depth_vals = [depth_overall[np][ks] + mech_depth[np][ks] for np in np_values]

    general_depth_vals = [depth_overall[np][ks] for np in np_values]

    custom_over_vals = [gate_overall[np][ks] + custom_overhead[np][ks] for np in np_values]
    sabre_over_vals = [gate_overall[np][ks] + sabre_overhead[np][ks] for np in np_values]
    mech_over_vals = [gate_overall[np][ks] + mech_overhead[np][ks] for np in np_values]
    general_over_vals = [gate_overall[np][ks] for np in np_values]

    overhead_ours = []
    overhead_sabre = []
    overhead_mech = []
    overhead_gates_ours = []
    overhead_gates_sabre = []
    overhead_gates_mech = []
    for i in range(3):
        print((custom_depth_vals[i] - general_depth_vals[i]) / (sabre_depth_vals[i] - general_depth_vals[i]))
        print(general_depth_vals[i])
        print((custom_over_vals[i] - general_over_vals[i]) / (sabre_over_vals[i] - general_over_vals[i]))
        print(general_over_vals[i])

        overhead_ours.append(f"{100 * ((custom_depth_vals[i] - general_depth_vals[i]) / general_depth_vals[i]):.0f}%")
        overhead_sabre.append(f"{100 * ((sabre_depth_vals[i] - general_depth_vals[i]) / general_depth_vals[i]):.0f}%")
        overhead_mech.append(f"{100 * ((mech_depth_vals[i] - general_depth_vals[i]) / general_depth_vals[i]):.0f}%")
        overhead_gates_ours.append(
            f"{100 * ((custom_over_vals[i] - general_over_vals[i]) / general_over_vals[i]):.0f}%"
        )
        overhead_gates_sabre.append(
            f"{100 * ((sabre_over_vals[i] - general_over_vals[i]) / general_over_vals[i]):.0f}%"
        )
        overhead_gates_mech.append(
            f"{100 * ((mech_over_vals[i] - general_over_vals[i]) / general_over_vals[i]):.0f}%"
        )

    x = np.arange(len(np_values))
    width = 0.19  # narrower now that there are 4 bars per group

    max_depth_val = max(
        max(general_depth_vals), max(custom_depth_vals), max(sabre_depth_vals), max(mech_depth_vals)
    )
    upper_ylim_depth = max_depth_val * 1.05

    fig = plt.figure(figsize=(HEIGHT_FIGSIZE * 2.5, WIDTH_FIGSIZE * 0.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[12, 20], hspace=0.1)

    # Top subplot: values from 40000 up to the max
    ax_top = plt.subplot(gs[0])
    bars_ideal_depth_top = ax_top.bar(
        x - 1.5 * width, general_depth_vals, width, label="Ideal", color="lightcoral", hatch="//", edgecolor="black"
    )
    bars_ours_top = ax_top.bar(
        x - 0.5 * width, custom_depth_vals, width, label="Chipmunq", color=pastel_blue, hatch="/", edgecolor="black"
    )
    bars_sabre_top = ax_top.bar(
        x + 0.5 * width, sabre_depth_vals, width, label="LightSABRE", color=pastel_orange, hatch="o", edgecolor="black"
    )
    bars_mech_top = ax_top.bar(
        x + 1.5 * width, mech_depth_vals, width, label="MECH", color=pastel_green, hatch="//", edgecolor="black"
    )

    ax_top.set_ylim(40000, upper_ylim_depth)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([])
    ax_top.tick_params(axis="y", length=5)
    plt.grid(True, which="major", linestyle="--", alpha=0.5)

    # Bottom subplot: values from 0 to 14000
    ax_bottom = plt.subplot(gs[1])
    bars_ideal_depth = ax_bottom.bar(
        x - 1.5 * width, general_depth_vals, width, label="Ideal", color="lightcoral", hatch="//", edgecolor="black"
    )
    bars_ours = ax_bottom.bar(
        x - 0.5 * width, custom_depth_vals, width, label="Chipmunq", color=pastel_blue, hatch="/", edgecolor="black"
    )
    bars_sabre = ax_bottom.bar(
        x + 0.5 * width, sabre_depth_vals, width, label="LightSABRE", color=pastel_orange, hatch="o", edgecolor="black"
    )
    bars_mech = ax_bottom.bar(
        x + 1.5 * width, mech_depth_vals, width, label="MECH", color=pastel_green, hatch="//", edgecolor="black"
    )

    # Style the timeout bars (last two MECH bars) on both axes
    for bars_mech_set in (bars_mech_top, bars_mech):
        tb = [bars_mech_set[-1], bars_mech_set[-2]]
        for timeout_bar in tb:
            timeout_bar.set_facecolor("white")
            timeout_bar.set_edgecolor("red")
            timeout_bar.set_linestyle("--")
            timeout_bar.set_hatch("xxx")
            timeout_bar.set_linewidth(2)
            timeout_bar.set_path_effects([])
            timeout_bar.set_edgecolor("#B2D8B2")
    ax_bottom.text(x[-1] + width + 0.12, 2000, "T/O", ha="center", va="bottom", color="red", fontweight="bold", fontsize=10)
    ax_bottom.text(x[-2] + width+0.12, 8000, "T/O", ha="center", va="bottom", color="red", fontweight="bold", fontsize=10)

    print(mech_depth_vals)

    ax_bottom.set_ylim(0, 14000)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(section_titles)
    ax_bottom.set_xlabel("Circuit type")
    ax_bottom.tick_params(axis="y", length=5)

    ax_bottom.spines["top"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax_bottom.tick_params(axis="x", which="both", top=False)

    fig.text(0.03, 0.5, "Circuit depth", va="center", rotation="vertical", fontsize=FONTSIZE * 1.5)

    # Diagonal break marks
    d = 0.015
    kwargs = dict(transform=ax_bottom.transAxes, color="k", clip_on=False, linewidth=1.5)
    ax_bottom.plot((-d, +d), (1, 1), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1, 1), **kwargs)

    kwargs.update(transform=ax_top.transAxes)
    ax_top.plot((-d, +d), (0, 0), **kwargs)
    ax_top.plot((1 - d, 1 + d), (0, 0), **kwargs)

    ax_top.text(-0.18, 1.15, "b) Compilation overhead on circuit depth", transform=ax_top.transAxes, fontweight="bold")
   
    ax_top.text(0.25, 1.4, "Lower is better ↓", transform=ax_top.transAxes, fontweight="bold", color=plot_lib_color)

    """
    # Arrows/labels all target values under 14000, so they live on ax_bottom
    ax_bottom.text(0.16, 0.22, "-5.5x", transform=ax_bottom.transAxes, color="green")
    ax_bottom.annotate(
        "",
        xy=(-0.04, 1500),
        xytext=(0.3, 2500),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.6", color="green", lw=1.5),
    )

    ax_bottom.text(0.5, 0.4, "-9x", transform=ax_bottom.transAxes, color="green")
    ax_bottom.annotate(
        "",
        xy=(1, 4000),
        xytext=(1.3, 5500),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.6", color="green", lw=1.5),
    )

    ax_bottom.text(0.84, 0.66, "-8x", transform=ax_bottom.transAxes, color="green")
    ax_bottom.annotate(
        "",
        xy=(2, 8000),
        xytext=(2.3, 9500),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.6", color="green", lw=1.5),
    )
    """

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
        "xtick.labelsize": (FONTSIZE - 1) * 1,
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
    max_overall_val = max(
        max(general_over_vals), max(custom_over_vals), max(sabre_over_vals), max(mech_over_vals)
    )

    # Add a small padding to the upper y-limit for better visualization
    upper_ylim = 124000  # max_overall_val*1.01 # 1.015

    # fig = plt.figure(figsize=(HEIGHT_FIGSIZE*2.5, WIDTH_FIGSIZE*0.92))
    fig = plt.figure(figsize=(HEIGHT_FIGSIZE * 2.5, WIDTH_FIGSIZE * 0.5))

    gs = gridspec.GridSpec(2, 1, height_ratios=[12, 20], hspace=0.1)  # Swapped height_ratios

    # Top subplot (for values above the break, e.g., 60,000 to max)
    ax_top = plt.subplot(gs[0])  # Now gs[0] for the top part
    bars_ideal = ax_top.bar(
        x - 1.5 * width, general_over_vals, width, label="Ideal", color="lightcoral", hatch="//", edgecolor="black"
    )
    bars_ours_top = ax_top.bar(
        x - 0.5 * width, custom_over_vals, width, label="Chipmunq", color=pastel_blue, hatch="/", edgecolor="black"
    )
    bars_sabre_top = ax_top.bar(
        x + 0.5 * width, sabre_over_vals, width, label="LightSABRE", color=pastel_orange, hatch="o", edgecolor="black"
    )
    bars_mech_top = ax_top.bar(
        x + 1.5 * width, mech_over_vals, width, label="MECH", color=pastel_green, hatch="//", edgecolor="black"
    )
    
    print(mech_over_vals)

    # Limit for upper
    ax_top.set_ylim(100000, upper_ylim)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([])
    ax_top.tick_params(axis="y", length=5)
    plt.grid(True, which="major", linestyle="--", alpha=0.5)

    # Bottom subplot (for values below the break, e.g., 0 to 40,000)
    ax_bottom = plt.subplot(gs[1])  # Now gs[1] for the bottom part
    bars_ideal_bottom = ax_bottom.bar(
        x - 1.5 * width, general_over_vals, width, label="Ideal", color="lightcoral", hatch="/", edgecolor="black"
    )
    bars_gates_ours = ax_bottom.bar(
        x - 0.5 * width, custom_over_vals, width, label="Chipmunq", color=pastel_blue, hatch="/", edgecolor="black"
    )
    bars_gates_sabre = ax_bottom.bar(
        x + 0.5 * width, sabre_over_vals, width, label="LightSABRE", color=pastel_orange, hatch="o", edgecolor="black"
    )
    bars_gates_mech = ax_bottom.bar(
        x + 1.5 * width, mech_over_vals, width, label="MECH", color=pastel_green, hatch="//", edgecolor="black"
    )
    tb = [bars_gates_mech[-1], bars_gates_mech[-2]]
    for timeout_bar in tb:
        timeout_bar.set_facecolor("white")
        # timeout_bar.set_edgecolor('#B2D8B2')
        # timeout_bar.set_facecolor('white')
        timeout_bar.set_edgecolor("red")
        timeout_bar.set_linestyle("--")
        timeout_bar.set_hatch("xxx")
        timeout_bar.set_linewidth(2)
        timeout_bar.set_path_effects([])
        timeout_bar.set_edgecolor("red")
        timeout_bar.set_edgecolor("#B2D8B2")
    ax_bottom.text(x[-1] + width + 0.12, 10000, "T/O", ha="center", va="bottom", color="red", fontweight="bold", fontsize=10)
    ax_bottom.text(x[-2] + width+0.12, 25000, "T/O", ha="center", va="bottom", color="red", fontweight="bold", fontsize=10)

    # ax_top.bar_label(bars_gates_ours, labels = overhead_gates_ours, padding = 9, rotation = 90)
    # ax_bottom.bar_label(bars_gates_ours, labels = overhead_gates_ours, padding = 9, rotation = 90)
    # ax_top.bar_label(bars_gates_sabre, labels = overhead_gates_sabre, padding = 9, rotation = 90)
    # ax_bottom.bar_label(bars_gates_sabre, labels = overhead_gates_sabre, padding = 9, rotation = 90)

    

    # Limit for bottom
    ax_bottom.set_ylim(0, 80000)
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
    """
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
    """

    fig.text(0.025, 0.5, "#2q gates", va="center", rotation="vertical", fontsize=FONTSIZE * 1.5)
    fig.text(0.46, 0.055, "Circuit type", va="center", rotation="horizontal", fontsize=FONTSIZE * 1.5)
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

    legend_fig = plt.figure(figsize=(4, 2))
    legend = legend_fig.legend(
        handles=[bars_ideal, bars_gates_ours, bars_gates_sabre, bars_gates_mech],
        loc="center",
        frameon=False,
        ncols=4,
        columnspacing=1.5,
    )
    legend_fig.savefig(filename + "legend.pdf", bbox_inches="tight", format="pdf")
    plt.close(legend_fig)


def run_exp_statistics(reproduce: bool = False) -> None:

    # Backend configuration
    num_inter_chiplet_connections = 8
    ps_inter = 1e-4
    n_patches = [6, 1]#[1, 3, 6]

    custom_depth = {}
    custom_overhead = {}
    sabre_depth = {}
    sabre_overhead = {}
    mech_depth = {}
    mech_overhead = {}
    mech_all = {}
    depth_overall = {}
    gate_overall = {}

    def num_2q_gates(circuit):
        ops = circuit.count_ops()
        two_qubit_gate_names = ["cx", "cz", "swap"]
        return sum(ops.get(g, 0) for g in two_qubit_gate_names)

    if reproduce:
        
        for ks in [2]:  # [1, 2, 3, 4]
            
            # Different number of patches for surface code
            for np in n_patches:
                if np not in custom_depth:
                    custom_depth[np] = {}
                    custom_overhead[np] = {}
                    sabre_depth[np] = {}
                    sabre_overhead[np] = {}
                    mech_depth[np] = {}
                    mech_overhead[np] = {}
                    mech_all[np] = {}

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

                # MECH compilation
                print("MECH")
                if np == 1:
                    monolithic_backend, qubit_num, data_qubit_num = generate_simple_backend(14, 14)
                    circuit_mech = transpile_circuit_MECH(stim_code_circuit.qc, monolithic_backend)
                    result_mech = calc_circuit_mech_stats(circuit_mech, stim_code_circuit.qc)

                    mech_depth[np][ks] = result_mech["depth_overhead"]
                    mech_overhead[np][ks] = result_mech["2q_gates_overhead"]
                    mech_all[np][ks] = result_mech
                else: 
                    mech_depth[np][ks] = -1
                    mech_overhead[np][ks] = -1
                    mech_all[np][ks] = -1

                custom_depth[np][ks] = custom_circuit.depth() - (stim_code_circuit.qc).depth()
                custom_overhead[np][ks] = num_2q_gates(custom_circuit) - num_2q_gates(stim_code_circuit.qc)

                sabre_depth[np][ks] = sabre_circuit.depth() - (stim_code_circuit.qc).depth()
                sabre_overhead[np][ks] = num_2q_gates(sabre_circuit) - num_2q_gates(stim_code_circuit.qc)


                depth_overall[np][ks] = (stim_code_circuit.qc).depth()
                gate_overall[np][ks] = num_2q_gates(stim_code_circuit.qc)

            

            # Gross Code with 12 logical qubits
            np = 12
            custom_depth[np] = {}
            custom_overhead[np] = {}
            sabre_depth[np] = {}
            sabre_overhead[np] = {}
            mech_depth[np] = {} 
            mech_overhead[np] = {}
            mech_all[np] = {}

            depth_overall[np] = {}
            gate_overall[np] = {}

            # Generate circuit
            circuit, partitions = generate_gross_code(num_qubits = 1)

            backend = BackendChipletV2(
                # 1 Chiplet with 288 qubits
                size = (1, 1, 12, 24),                     
                n_inter = 1, 
                # Long range connections                               
                connectivity="torus",
                # Grid layout
                topology="grid",
                long_range_offsets=[(1,0), (2,0), (3,0),
                                    (0,1), (0,2), (0,3)],
                num_defective_qubits=0
            )

            stim_code_circuit = StimCodeCircuit(stim_circuit=circuit)

            # Custom transpilation
            print("Custom")
            custom_circuit = custom_partitioned_transpilation(
                stim_code_circuit.qc, backend, pre_defined_partitions=partitions
            )

            # Sabre transpilation
            print("Sabre")
            sabre_circuit = sabre_transpilation(stim_code_circuit.qc, backend)

            # MECH transpilation
            print("MECH")
            """
            monolithic_backend, qubit_num, data_qubit_num = generate_simple_backend(14, 14)
            architecture = generate_qecc_synth_backend_from_mech(monolithic_backend)
            circuit_mech = transpile_circuit_MECH(stim_code_circuit.qc, monolithic_backend)
            result_mech = calc_circuit_mech_stats(circuit_mech, stim_code_circuit.qc)

            mech_depth[np][ks] = result_mech["depth_overhead"]
            mech_overhead[np][ks] = result_mech["2q_gates_overhead"]
            mech_all[np][ks] = result_mech
            """
            mech_depth[np][ks] = -1
            mech_overhead[np][ks] = -1
            mech_all[np][ks] = -1

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
        with open(output_dir / "mech_depth.pkl", "wb") as f:
            pickle.dump(mech_depth, f)
        with open(output_dir / "mech_overhead.pkl", "wb") as f:
            pickle.dump(mech_overhead, f)
        with open(output_dir / "mech_all.pkl", "wb") as f:
            pickle.dump(mech_all, f)
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
    with open("experiments/evaluation/scalability/mech_depth.pkl", "rb") as f:
        mech_depth = pickle.load(f)
    with open("experiments/evaluation/scalability/mech_overhead.pkl", "rb") as f:
        mech_overhead = pickle.load(f)
    with open("experiments/evaluation/scalability/depth_overall.pkl", "rb") as f:
        depth_overall = pickle.load(f)
    with open("experiments/evaluation/scalability/gate_overall.pkl", "rb") as f:
        gate_overall = pickle.load(f)

    plot_combined_split(
        custom_depth,
        custom_overhead,
        sabre_depth,
        sabre_overhead,
        mech_depth,
        mech_overhead,
        depth_overall,
        gate_overall,
        "experiments/evaluation/scalability/cnot_scaling_overhead_split",
    )


if __name__ == "__main__":
    run_exp_statistics(reproduce=False)
