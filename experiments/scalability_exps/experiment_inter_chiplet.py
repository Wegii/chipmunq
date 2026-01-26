# How does the number of inter-chiplet connections influence circuit routing?
# How does the error of inter-chiplet connections influence circuit routing?


from __future__ import annotations

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))
sys.path.append(os.path.join(os.getcwd(), "glue/eccentric_bench/"))
import time

# Custom utils
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.circuit_generator import get_tqec_cnot_rotated
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit
import numpy as np
import matplotlib.pyplot as plt
import pickle
from experiments.utils import *


def plot_combined(low_depth, low_overhead, high_depth, high_overhead, filename: str = ""):
    ks = list(next(iter(low_depth.values())).keys())[0]
    np_values = low_depth.keys()#sorted(low_depth.keys())

    section_titles = ["Full", "Half", "Limited"]

    # Extract values
    low_depth_vals = [low_depth[np][ks] for np in np_values]
    high_depth_vals = [high_depth[np][ks] for np in np_values]

    low_over_vals = [low_overhead[np][ks] for np in np_values]
    high_over_vals = [high_overhead[np][ks] for np in np_values]

    x = np.arange(len(np_values))
    width = 0.35

    tex_fonts = {
        # Use LaTeX to write all text
        # "text.usetex": True,
        "font.family": "serif",
        # Font sizes
        "axes.labelsize": FONTSIZE*1.5,
        "font.size": FONTSIZE*1.2,
        "legend.fontsize": (FONTSIZE - 2)*1.5,
        "xtick.labelsize": (FONTSIZE - 1)*1.5,
        "ytick.labelsize": (FONTSIZE - 1)*1.5,
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

    # Pastel colors
    pastel_blue = '#A7D9ED'
    pastel_orange = '#6476AD'#'#F7C6A2'

    # Create depth statistics
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.5, WIDTH_FIGSIZE*0.92))
    
    ax.bar(x - width/2, low_depth_vals, width,
           label = r"$p_{inter}$ = $1e^{-4}$",
           color = pastel_blue,
           hatch = '/', edgecolor = 'black')

    ax.bar(x + width/2, high_depth_vals, width,
           label = r"$p_{inter}$ = $1e^{-2}$",
           color = pastel_orange,
           hatch = 'o', edgecolor = 'black')

    ax.set_xticks(x)
    ax.set_xticklabels(section_titles)
    ax.set_xlabel("Circuit size")
    ax.set_ylabel("Circuit Depth Overhead")
    ax.legend()

    # Add annotation

    ax.text(
        -0.02, 1.02, "Effect of inter-chiplet noise",
        transform=ax.transAxes,
        fontweight="bold"
    )

    ax.text(
        0.3, 1.07, "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    #fig.tight_layout()
    fig.subplots_adjust(left=0.24, right=0.95, top=0.9, bottom=0.12)
    fig.savefig(f"{filename}_depth.pdf", format="pdf")
    plt.close(fig)


    # Create 2q gate overhead
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.5, WIDTH_FIGSIZE*0.92))


    ax.bar(x - width/2, low_over_vals, width,
           label = r"$p_{inter}$ = $1e^{-4}$",
           color = pastel_blue,
           hatch = '/', edgecolor = 'black')

    ax.bar(x + width/2, high_over_vals, width,
           label = r"$p_{inter}$ = $1e^{-2}$",
           color = pastel_orange,
           hatch = 'o', edgecolor = 'black')

    ax.set_xticks(x)
    ax.set_xticklabels(section_titles)
    ax.set_xlabel("Circuit size")
    ax.set_ylabel("2q Gate Overhead ")
    ax.legend()

    # Add annotation
    #ax.text(0.57, 1.04, "Lower is better ↓",
    #    transform=ax.transAxes,
    #    #fontsize=10,
    #    fontweight='bold',
    #    color = plot_lib_color,
    #    va='top',
    #    ha='left')
    
    ax.text(
        -0.02, 1.02, "Effect of inter-chiplet noise",
        transform=ax.transAxes,
        fontweight="bold"
    )

    ax.text(
        0.3, 1.07, "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    #fig.tight_layout()
    fig.subplots_adjust(left=0.24, right=0.95, top=0.9, bottom=0.12)
    fig.savefig(f"{filename}_overhead.pdf", format="pdf")
    plt.close(fig)

    

def run_exp_inter_chiplet():

    low_error_depth = {}
    low_error_overhead = {}
    high_error_depth = {}
    high_error_overhead = {}

    np = 4
    num_inter_chiplet_connections = [8, 4, 1]

    """
    for ks in [2]:#[1, 2, 3, 4]
        for ni in num_inter_chiplet_connections:

            if ni not in low_error_depth:
                low_error_depth[ni] = {}
                low_error_overhead[ni] = {}
                high_error_depth[ni] = {}
                high_error_overhead[ni] = {}

            # TODO: Calculate necessary backend given number of patches

            # Generate circuit
            circuit, partitions = get_tqec_cnot_rotated(distance_scale = ks,
                                                        n1 = np,
                                                        n2 = 0)

            backend_1e4 = BackendChipletV2(size = (np*2, np*2, 15, 8),
                            n_inter = ni,
                            connectivity = "nn",
                            topology = "rotated_grid",
                            inter_chiplet_noise = 1e-4,
                            inter_chiplet_amplification = 1,
                            inter_chiplet_noise_type = "random",
                            num_defective_qubits = 0,
                        )
            
            backend_1e2 = BackendChipletV2(size = (np*2, np*2, 15, 8),
                            n_inter = ni,
                            connectivity = "nn",
                            topology = "rotated_grid",
                            inter_chiplet_noise = 1e-2,
                            inter_chiplet_amplification = 1,
                            inter_chiplet_noise_type = "random",
                            num_defective_qubits = 0,
                        )

            # Stim to qiskit
            stim_code_circuit = StimCodeCircuit(stim_circuit = circuit)

            # Low error transpilation
            low_error_circuit = custom_cost_transpilation(stim_code_circuit.qc,
                                                          backend_1e4,
                                                          pre_defined_partitions=partitions,
                                                          routing_alpha = 0,##1e-4,
                                                          routing_beta = 0)

            # High error transpilation
            high_error_circuit = custom_cost_transpilation(stim_code_circuit.qc,
                                                          backend_1e2,
                                                          pre_defined_partitions=partitions,
                                                          routing_alpha = 1,##1e-4,
                                                          routing_beta = 1)

            def num_2q_gates(circuit):
                ops = circuit.count_ops()
                two_qubit_gate_names = ["cx", "cz", "swap"]
                return sum(ops.get(g, 0) for g in two_qubit_gate_names)

            low_error_depth[ni][ks] = low_error_circuit.depth() - (stim_code_circuit.qc).depth()
            low_error_overhead[ni][ks] = num_2q_gates(low_error_circuit) - num_2q_gates(stim_code_circuit.qc)

            high_error_depth[ni][ks] = high_error_circuit.depth() - (stim_code_circuit.qc).depth()
            high_error_overhead[ni][ks] = num_2q_gates(high_error_circuit) - num_2q_gates(stim_code_circuit.qc)


    with open(f"experiments/evaluation/inter_chiplet/low_error_depth.pkl", "wb") as f:
        pickle.dump(low_error_depth, f)
    with open(f"experiments/evaluation/inter_chiplet/low_error_overhead.pkl", "wb") as f:
        pickle.dump(low_error_overhead, f)
    with open(f"experiments/evaluation/inter_chiplet/high_error_depth.pkl", "wb") as f:
        pickle.dump(high_error_depth, f)
    with open(f"experiments/evaluation/inter_chiplet/high_error_overhead.pkl", "wb") as f:
        pickle.dump(high_error_overhead, f)
    """

    with open(f"experiments/evaluation/inter_chiplet/low_error_depth.pkl", "rb") as f:
        low_error_depth = pickle.load(f)
    with open(f"experiments/evaluation/inter_chiplet/low_error_overhead.pkl", "rb") as f:
        low_error_overhead = pickle.load(f)
    with open(f"experiments/evaluation/inter_chiplet/high_error_depth.pkl", "rb") as f:
        high_error_depth = pickle.load(f)
    with open(f"experiments/evaluation/inter_chiplet/high_error_overhead.pkl", "rb") as f:
        high_error_overhead = pickle.load(f)


    plot_combined(low_error_depth,
                  low_error_overhead,
                  high_error_depth,
                  high_error_overhead,
                  "experiments/evaluation/inter_chiplet/cnot_inter_chiplet_overhead")

            
if __name__ == "__main__":
    run_exp_inter_chiplet()