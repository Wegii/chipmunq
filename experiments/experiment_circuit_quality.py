# TODO: Number of gates and depth

from __future__ import annotations

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))
#sys.path.append(os.path.join(os.getcwd(), "../eccentric_bench/"))
sys.path.append(os.path.join(os.getcwd(), "../eccentric_bench/utils/"))
sys.path.append(os.path.join(os.getcwd(), "../eccentric_bench/external/qiskit_qec/src"))

# Enable debug logging for Qiskit
#import logging
#logging.basicConfig(level=logging.DEBUG)
import time

# Custom utils
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.circuit_generator import QECMemory, GenericCircuit
from experiments.exp_utils.circuit_statistics import QECCircuitStats

# Plotting
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def _transpile(circuit: qiskit.QuantumCircuit, backend: BackendChipletV2) -> tuple[float, float]:
    """Transpilation of circuit to backend using custom and sabre transpilation passes

    :param circuit: _description_
    :type circuit: qiskit.QuantumCircuit
    :param backend: _description_
    :type backend: BackendChipletV2
    :return: _description_
    :rtype: tuple[float, float]
    """

    custom_circuit = custom_partitioned_transpilation(circuit, backend)
    sabre_circuit = sabre_transpilation(circuit, backend)

    return custom_circuit, sabre_circuit


def plot_gate_statistics_bar_chart(custom_gates, sabre_gates):

    groups = {
        'Small': ['SABRE', 'C'],
        'Medium': ['SABRE', 'C'],
        'Big': ['SABRE', 'C']
    }

    base_palette = sns.color_palette("pastel", n_colors=2*3)
    colors = {}
    for c in range(3):#enumerate(3):
        colors[c] = base_palette[c+1]

    fig, ax = plt.subplots(figsize=(10, 6))
    #fig.suptitle(figure_title, fontsize=14, fontweight='bold')

    width = 0.35
    x = np.arange(len(groups))

    for i in range(len(custom_gates)):
        if i == 0:
            ax.bar(x[i] - width/2, sabre_gates[i], width, hatch='/', color=colors[0], edgecolor='black', label="SABRE")
            ax.bar(x[i] + width/2, custom_gates[i], width, hatch='o', color=colors[1], edgecolor='black', label="Custom")
        else:
            ax.bar(x[i] - width/2, sabre_gates[i], width, hatch='/', color=colors[0], edgecolor='black')
            ax.bar(x[i] + width/2, custom_gates[i], width, hatch='o', color=colors[1], edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(groups.keys(), fontsize=12)
    ax.set_ylabel('Number of two-qubit gates', fontsize=12)
    ax.set_xlabel('Backend Configuration', fontsize=12)
    #ax.set_yscale("log")
    ax.text(-0.025, 1.05, 'Lower is better ↓', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
    ax.legend(title="Method",
              fontsize=10,
              loc='center left',
              bbox_to_anchor=(1.02, 0.5))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"experiments/data/figures/statistics/circuit_statistics_gates.png", bbox_inches='tight')
    plt.close()


def plot_depth_statistics_bar_chart(custom_depth, sabre_depth):

    groups = {
        'Small': ['SABRE', 'C'],
        'Medium': ['SABRE', 'C'],
        'Big': ['SABRE', 'C']
    }

    base_palette = sns.color_palette("pastel", n_colors=2*3)
    colors = {}
    for c in range(3):#enumerate(3):
        colors[c] = base_palette[c+1]

    fig, ax = plt.subplots(figsize=(10, 6))
    #fig.suptitle(figure_title, fontsize=14, fontweight='bold')

    width = 0.35
    x = np.arange(len(groups))

    for i in range(len(custom_gates)):
        if i == 0:
            ax.bar(x[i] - width/2, sabre_depth[i], width, hatch='/', color=colors[0], edgecolor='black', label="SABRE")
            ax.bar(x[i] + width/2, custom_depth[i], width, hatch='o', color=colors[1], edgecolor='black', label="Custom")
        else:
            ax.bar(x[i] - width/2, sabre_depth[i], width, hatch='/', color=colors[0], edgecolor='black')
            ax.bar(x[i] + width/2, custom_depth[i], width, hatch='o', color=colors[1], edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(groups.keys(), fontsize=12)
    ax.set_ylabel('Depth', fontsize=12)
    ax.set_xlabel('Backend Configuration', fontsize=12)
    #ax.set_yscale("log")
    ax.text(-0.025, 1.05, 'Lower is better ↓', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
    ax.legend(title="Method",
              fontsize=10,
              loc='center left',
              bbox_to_anchor=(1.02, 0.5))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"experiments/data/figures/statistics/circuit_statistics_depth.png", bbox_inches='tight')
    plt.close()


def _get_circuit(type, num_patches, num_qubits):
    # Get circuit with non-interacting patches

    if type == "memory":
        #Note: Selects the maximum code distance given the number of qubits
        circuit_generator = QECMemory(num_qubits)

        # Note: This is a stim circuit!
        circuit = circuit_generator.generate_code_memory('surface', num_patches)

    elif type == "2-qubit surgery":
        pass
    elif type == "generic":
        num_qubits = num_qubits - 10
        circuit_generator = GenericCircuit(num_qubits)

        # Note: This is not a stim_circuit !
        circuit = circuit_generator.generate_circuit(num_patches)

    return circuit


if __name__ == "__main__":
    n_inter = 5

    small_backend = BackendChipletV2((2, 2, 10, 10), n_inter)
    medium_backend = BackendChipletV2((8, 8, 10, 10), n_inter)
    big_backend = BackendChipletV2((16, 16, 10, 10), n_inter)

    # Small backend
    small_generic_patch_circuit = _get_circuit("", 2*2)
    custom_small, sabre_small = _transpile(small_generic_patch_circuit, small_backend)
    custom_small_stats = QECCircuitStats(transpiled_circuit = custom_small, backend = small_backend)
    sabre_small_stats = QECCircuitStats(transpiled_circuit = sabre_small, backend = small_backend)
    #basic_small_stats = QECCircuitStats(transpiled_circuit = basic_small, backend = small_backend)


    
    # Medium backend
    medium_generic_patch_circuit = _get_circuit("", 8*8)
    custom_medium, sabre_medium = _transpile(medium_generic_patch_circuit, medium_backend)
    custom_medium_stats = QECCircuitStats(transpiled_circuit = custom_medium, backend = medium_backend)
    sabre_medium_stats = QECCircuitStats(transpiled_circuit = sabre_medium, backend = medium_backend)
    #basic_medium_stats = QECCircuitStats(transpiled_circuit = basic_medium, backend = medium_backend)

    # Big backend
    """
    big_generic_patch_circuit = _get_circuit("", 16*16)
    custom_big, sabre_big = _transpile(big_generic_patch_circuit, big_backend)
    custom_big_stats = QECCircuitStats(transpiled_circuit = custom_big, backend = big_backend)
    sabre_big_stats = QECCircuitStats(transpiled_circuit = sabre_big, backend = big_backend)
    #basic_big_stats = QECCircuitStats(transpiled_circuit = basic_big, backend = big_backend)
    """
    
    two_qubit_gates = []
    custom_gates = [custom_small_stats.get_num_two_gates(),
                    custom_medium_stats.get_num_two_gates(),]
                    #custom_big_stats.get_num_two_gates()]
    
    custom_depth = [custom_small_stats.get_depth(),
                    custom_medium_stats.get_depth(),]
                    #custom_big_stats.get_depth()]
    
    sabre_gates = [sabre_small_stats.get_num_two_gates(),
                   sabre_medium_stats.get_num_two_gates(),]
                   #sabre_big_stats.get_num_two_gates()]
    
    sabre_depth = [sabre_small_stats.get_depth(),
                   sabre_medium_stats.get_depth(),]
                   #sabre_big_stats.get_depth()]

    #print(custom_gates)
    print(custom_depth)
    #print(sabre_gates)
    print(sabre_depth)

    # Custom:
    #   Trivial: 1406
    #   Random: 12107 -> Much much worse
    # SABRE: 330

    #plot_gate_statistics_bar_chart(custom_gates, sabre_gates)
    #plot_depth_statistics_bar_chart(custom_depth, sabre_depth)
    