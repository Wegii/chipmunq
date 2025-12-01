# 3. How do defective qubits affect the resulting circuit?


from __future__ import annotations

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))
sys.path.append(os.path.join(os.getcwd(), "glue/eccentric_bench/"))
import time

# Custom utils
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.circuit_generator import QECMemory, QECCircuit, get_tqec_cnot_rotated
from qeccm.backends.backend_utils import plot_circuit_layout, plot_circuit_layout_utilization
from qeccm.src.reference_partitions import memory_d5
from experiments.exp_utils.simulation_utils import *
from qeccm.backends.backend_utils import plot_gate_map

from experiments.exp_utils.circuit_utils import stim_to_qiskit
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit
from glue.qiskit_qec.stim_tools import get_stim_circuits_with_detectors
from stim import Circuit as StimCircuit

from glue.eccentric_bench.noise import get_noise_model


import numpy as np
import matplotlib.pyplot as plt



def plot_combined(custom_depth, custom_overhead, filename: str = ""):
    ks = list(next(iter(custom_depth.values())).keys())[0]
    np_values = sorted(custom_depth.keys())

    section_titles = ["1", "2", "3"]

    # Extract values
    custom_depth_vals = [custom_depth[np][ks] for np in np_values]
    custom_over_vals = [custom_overhead[np][ks] for np in np_values]

    x = np.arange(len(np_values))
    width = 0.6   # one bar per section

    # Pastel colors
    pastel_blue   = "#5c79bd"
    pastel_orange = "#ef8d38"
    pastel_green  = "#7bb274"

    colors = [pastel_blue, pastel_orange, pastel_green]

    # ---------- DEPTH ----------
    fig, ax = plt.subplots(figsize=(5, 8))

    ax.bar(
        x, custom_depth_vals, width,
        color=colors,
        hatch='/',
        edgecolor='black'
    )

    ax.set_xticks(x)
    ax.set_xticklabels(section_titles)
    ax.set_xlabel("#Defective Qubits")
    ax.set_ylabel("Circuit Depth Overhead")

    # Annotation
    ax.text(
        -0.025, 1.05, "Lower is better ↓",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left"
    )

    fig.tight_layout()
    fig.savefig(f"{filename}_depth.png", dpi=300)
    plt.show()


    # ---------- 2Q GATE OVERHEAD ----------
    fig, ax = plt.subplots(figsize=(5, 8))

    ax.bar(
        x, custom_over_vals, width,
        color=colors,
        hatch='/',
        edgecolor='black'
    )

    ax.set_xticks(x)
    ax.set_xticklabels(section_titles)
    ax.set_ylabel("2q Gate Overhead")

    # Annotation
    ax.text(
        -0.025, 1.05, "Lower is better ↓",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left"
    )

    fig.tight_layout()
    fig.savefig(f"{filename}_overhead.png", dpi=300)
    plt.show()



def run_transpilation():

    # Backend configuration
    num_inter_chiplet_connections = 8
    ps_inter = 1e-4

    custom_depth = {}
    custom_overhead = {}

    np = 4
    defective_qubits = [1, 2, 3]
    for ks in [2]:#[1, 2, 3, 4]
        for df in defective_qubits:

            if df not in custom_depth:
                custom_overhead[df] = {}
                custom_depth[df] = {}
                
            # TODO: Calculate necessary backend given number of patches

            # Generate circuit
            circuit, partitions = get_tqec_cnot_rotated(distance_scale = ks,
                                                        n1 = np,
                                                        n2 = 0)

            backend = BackendChipletV2(size = (np*6, np*6, 15, 8),
                            n_inter = num_inter_chiplet_connections,
                            connectivity = "nn",
                            topology = "rotated_grid",
                            inter_chiplet_noise = ps_inter,
                            inter_chiplet_amplification = 1,
                            inter_chiplet_noise_type = "constant",
                            num_defective_qubits=df,
                        )

            # Stim to qiskit
            stim_code_circuit = StimCodeCircuit(stim_circuit = circuit)

            # Custom transpilation
            print("Custom")
            custom_circuit = custom_cost_transpilation(stim_code_circuit.qc,
                                                       backend,
                                                       pre_defined_partitions=partitions)


            def num_2q_gates(circuit):
                ops = circuit.count_ops()
                two_qubit_gate_names = ["cx", "cz", "swap"]
                return sum(ops.get(g, 0) for g in two_qubit_gate_names)

            custom_depth[df][ks] = custom_circuit.depth() - (stim_code_circuit.qc).depth()
            custom_overhead[df][ks] = num_2q_gates(custom_circuit) - num_2q_gates(stim_code_circuit.qc)


    plot_combined(custom_depth,
                  custom_overhead,
                  "experiments/evaluation/defective_qubits")

            
if __name__ == "__main__":
    run_transpilation()