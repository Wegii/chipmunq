# How do circuit depth and gate overhead scale as circuit size increases?


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



def plot_combined(custom_depth, custom_overhead, sabre_depth, sabre_overhead, filename: str = ""):
    ks = list(next(iter(custom_depth.values())).keys())[0]
    np_values = sorted(custom_depth.keys())

    section_titles = ["Small", "Medium", "Big"]

    # Extract values
    custom_depth_vals = [custom_depth[np][ks] for np in np_values]
    sabre_depth_vals = [sabre_depth[np][ks] for np in np_values]

    custom_over_vals = [custom_overhead[np][ks] for np in np_values]
    sabre_over_vals = [sabre_overhead[np][ks] for np in np_values]

    x = np.arange(len(np_values))
    width = 0.35

    # Pastel colors
    pastel_blue = "#5c79bd"
    pastel_orange = "#ef8d38"

    # Create depth statistics
    fig, ax = plt.subplots(figsize=(5, 8))

    ax.bar(x - width/2, custom_depth_vals, width,
           label="Custom", color=pastel_blue,
           hatch='/', edgecolor='black')

    ax.bar(x + width/2, sabre_depth_vals, width,
           label="SABRE", color=pastel_orange,
           hatch='o', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(section_titles)
    ax.set_xlabel("Circuit size")
    ax.set_ylabel("Circuit Depth Overhead")
    ax.legend()

    # Add annotation
    ax.text(-0.025, 1.05, 'Lower is better ↓',
            transform=ax.transAxes,
            fontsize=10,
            fontweight='bold',
            va='top',
            ha='left')

    fig.tight_layout()
    fig.savefig(f"{filename}_depth.png", dpi=300)
    plt.show()

    # Create 2q gate overhead
    fig, ax = plt.subplots(figsize=(5, 8))

    ax.bar(x - width/2, custom_over_vals, width,
           label="Custom", color=pastel_blue,
           hatch='/', edgecolor='black')

    ax.bar(x + width/2, sabre_over_vals, width,
           label="SABRE", color=pastel_orange,
           hatch='o', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(section_titles)
    ax.set_ylabel("2q Gate Overhead ")
    ax.legend()

    # Add annotation
    ax.text(-0.025, 1.05, 'Lower is better ↓',
            transform=ax.transAxes,
            fontsize=10,
            fontweight='bold',
            va='top',
            ha='left')

    fig.tight_layout()
    fig.savefig(f"{filename}_overhead.png", dpi=300)
    plt.show()

    

def run_exp_statistics():

    # Backend configuration
    num_inter_chiplet_connections = 8
    ps_inter = 1e-4

    custom_depth = {}
    custom_overhead = {}
    sabre_depth = {}
    sabre_overhead = {}

    n_patches = [1, 5, 10]
    for ks in [2]:#[1, 2, 3, 4]
        for np in n_patches:

            if np not in custom_depth:

                custom_depth[np] = {}
                custom_overhead[np] = {}
                sabre_depth[np] = {}
                sabre_overhead[np] = {}

            # TODO: Calculate necessary backend given number of patches

            # Generate circuit
            circuit, partitions = get_tqec_cnot_rotated(distance_scale = ks,
                                                        n1 = np,
                                                        n2 = 0)

            backend = BackendChipletV2(size = (np*2, np*2, 15, 8),
                            n_inter = num_inter_chiplet_connections,
                            connectivity = "nn",
                            topology = "rotated_grid",
                            inter_chiplet_noise = ps_inter,
                            inter_chiplet_amplification = 1,
                            inter_chiplet_noise_type = "constant",
                            num_defective_qubits=0,
                        )

            # Stim to qiskit
            stim_code_circuit = StimCodeCircuit(stim_circuit = circuit)

            # Custom transpilation
            print("Custom")
            custom_circuit = custom_cost_transpilation(stim_code_circuit.qc,
                                                       backend,
                                                       pre_defined_partitions=partitions)

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

    plot_combined(custom_depth,
                  custom_overhead,
                  sabre_depth,
                  sabre_overhead,
                  "experiments/evaluation/cnot_scaling_overhead")

            
if __name__ == "__main__":
    run_exp_statistics()