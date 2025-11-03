from __future__ import annotations

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))
#sys.path.append(os.path.join(os.getcwd(), "../eccentric_bench/"))
sys.path.append(os.path.join(os.getcwd(), "../eccentric_bench/utils/"))
sys.path.append(os.path.join(os.getcwd(), "../eccentric_bench/external/qiskit_qec/src"))
import time

# Custom utils
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.circuit_generator import QECMemory, GenericCircuit

# Plotting
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def _get_circuit(type, num_patches):
    # Get circuit with non-interacting patches

    if type == "surface":
        distance = 9
        # calculate num_qubits for distance
        num_qubits = 100

        circuit_generator = QECMemory(num_qubits)
        # Selects the maximum code distance given the number of qubits
        # Note: This is a stim circuit!
        circuit = circuit_generator.generate_code_memory('surface', num_patches)
    else:
        num_qubits = 10*10 - 10
        circuit_generator = GenericCircuit(num_qubits)
        # Note: This is not a stim_circuit !
        circuit = circuit_generator.generate_circuit(num_patches)

    return circuit


def _transpile(circuit: qiskit.QuantumCircuit, backend: BackendChipletV2) -> tuple[float, float]:
    """Time transpilation of circuit to backend using custom, accelerated and sabre transpilation passes

    :param circuit: _description_
    :type circuit: qiskit.QuantumCircuit
    :param backend: _description_
    :type backend: BackendChipletV2
    :return: _description_
    :rtype: tuple[float, float]
    """

    start_c = time.time()
    _ = custom_partitioned_transpilation(circuit, backend)
    end_c = time.time()
    c_time = end_c - start_c
    print("custom done")

    start_ca = time.time()
    _ = custom_accelerated_partitioned_transpilation(circuit, backend)
    end_ca = time.time()
    ca_time = end_ca - start_ca
    print("custom accelerated done")

    start = time.time()
    _ = sabre_transpilation(circuit, backend)
    end = time.time()
    s_time = end - start
    print("sabre done")

    return c_time, ca_time, s_time


def plot_execution_time_bar_chart(custom_timing, custom_accelerate_timing, sabre_timing):

    groups = {
        'Small': ['SABRE', 'C', 'CA'],
        'Medium': ['SABRE', 'C', 'CA'],
        'Big': ['SABRE', 'C', 'CA']
    }

    base_palette = sns.color_palette("pastel", n_colors=2*3)
    colors = {}
    for c in range(3):#enumerate(3):
        colors[c] = base_palette[c+1]

    fig, ax = plt.subplots(figsize=(10, 6))
    #fig.suptitle(figure_title, fontsize=14, fontweight='bold')

    width = 0.35
    x = np.arange(len(groups))

    for i in range(len(custom_timing)):
        if i == 0:
            ax.bar(x[i] - width, sabre_timing[i], width, hatch='/', color=colors[0], edgecolor='black', label="SABRE")
            ax.bar(x[i], custom_accelerate_timing[i], width, hatch='x', color=colors[2], edgecolor='black',
                   label="Custom (accelerate)")
            ax.bar(x[i] + width, custom_timing[i], width, hatch='o', color=colors[1], edgecolor='black', label="Custom")
        else:
            ax.bar(x[i] - width, sabre_timing[i], width, hatch='/', color=colors[0], edgecolor='black')
            ax.bar(x[i], custom_accelerate_timing[i], width, hatch='x', color=colors[2], edgecolor='black')
            ax.bar(x[i] + width, custom_timing[i], width, hatch='o', color=colors[1], edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(groups.keys(), fontsize=12)
    ax.set_ylabel('Runtime [s]', fontsize=12)
    ax.set_xlabel('Backend Configuration', fontsize=12)
    #ax.set_yscale("log")
    ax.text(-0.025, 1.05, 'Lower is better ↓', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
    ax.legend(title="Method",
              fontsize=10,
              loc='center left',
              bbox_to_anchor=(1.02, 0.5))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"experiments/data/figures/statistics/scaling_backend.png", bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    n_inter = 5

    small_backend = BackendChipletV2((2, 2, 10, 10), n_inter)
    medium_backend = BackendChipletV2((8, 8, 10, 10), n_inter)
    big_backend = BackendChipletV2((12, 12, 10, 10), n_inter)

    # Small backend
    small_generic_patch_circuit = _get_circuit("", 2*2)
    custom_time_small, custom_accelerate_time_small, sabre_time_small = _transpile(small_generic_patch_circuit,
                                                                                   small_backend)

    # Medium backend
    medium_generic_patch_circuit = _get_circuit("", 8*8)
    custom_time_medium, custom_accelerate_time_medium, sabre_time_medium = _transpile(medium_generic_patch_circuit,
                                                                                      medium_backend)

    # Big backend
    big_generic_patch_circuit = _get_circuit("", 12*12)
    custom_time_big, custom_accelerate_time_big, sabre_time_big = _transpile(big_generic_patch_circuit, big_backend)
    
    custom_timing = [custom_time_small, custom_time_medium, custom_time_big]
    custom_accelerate_timing = [custom_accelerate_time_small, custom_accelerate_time_medium, custom_accelerate_time_big]
    sabre_timing = [sabre_time_small, sabre_time_medium, sabre_time_big]
    print(custom_timing)
    print(custom_accelerate_timing)
    print(sabre_timing)

    plot_execution_time_bar_chart(custom_timing, custom_accelerate_timing, sabre_timing)
