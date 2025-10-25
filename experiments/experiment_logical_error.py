# TODO: Number of gates and depth

from __future__ import annotations

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))

# Custom utils
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.circuit_generator import QECMemory, GenericCircuit
from experiments.exp_utils.circuit_statistics import QECCircuitStats

# Plotting
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def _get_circuit(type, num_patches):
    # Get circuit with non-interacting patches

    distance = 9
    # calculate num_qubits for distance
    num_qubits = 100

    circuit_generator = QECMemory(num_qubits)
    # Selects the maximum code distance given the number of qubits
    # Note: This is a stim circuit!
    circuit = circuit_generator.generate_code_memory('surface', num_patches)

    return circuit


def _transpile(circuit: qiskit.QuantumCircuit, backend: BackendChipletV2) -> tuple[float, float]:
    """Time transpilation of circuit to backend using custom and sabre transpilation passes

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


if __name__ == "__main__":
    n_inter = 5

    small_backend = BackendChipletV2((2, 2, 10, 10), n_inter)
    medium_backend = BackendChipletV2((8, 8, 10, 10), n_inter)
    big_backend = BackendChipletV2((16, 16, 10, 10), n_inter)

    # Small backend
    small_generic_patch_circuit = _get_circuit("", 2*2)    
    custom_small, sabre_small = _transpile(small_generic_patch_circuit.qc, small_backend)

    print(custom_small.count_ops())
    print(sabre_small.count_ops())

    custom_small_stats = QECCircuitStats(transpiled_circuit = custom_small,
                                         stim_circuit = small_generic_patch_circuit,
                                         backend = small_backend)
    custom_small_error_rate = custom_small_stats.get_logical_error_rate(num_samples = 10000)

    sabre_small_stats = QECCircuitStats(transpiled_circuit = sabre_small,
                                         stim_circuit = small_generic_patch_circuit,
                                         backend = small_backend)

    sabre_small_error_rate = sabre_small_stats.get_logical_error_rate(num_samples = 10000)

    print(custom_small_error_rate)
    print(sabre_small_error_rate)


    """
    sabre_small_stats = QECCircuitStats(transpiled_circuit = sabre_small, backend = small_backend)
    #basic_small_stats = QECCircuitStats(transpiled_circuit = basic_small, backend = small_backend)

    # Medium backend
    medium_generic_patch_circuit = _get_circuit("", 8*8)
    custom_medium, sabre_medium = _transpile(medium_generic_patch_circuit, medium_backend)
    custom_medium_stats = QECCircuitStats(transpiled_circuit = custom_medium, backend = medium_backend)
    sabre_medium_stats = QECCircuitStats(transpiled_circuit = sabre_medium, backend = medium_backend)
    #basic_medium_stats = QECCircuitStats(transpiled_circuit = basic_medium, backend = medium_backend)

    # Big backend
    big_generic_patch_circuit = _get_circuit("", 16*16)
    custom_big, sabre_big = _transpile(big_generic_patch_circuit, big_backend)
    custom_big_stats = QECCircuitStats(transpiled_circuit = custom_big, backend = big_backend)
    sabre_big_stats = QECCircuitStats(transpiled_circuit = sabre_big, backend = big_backend)
    #basic_big_stats = QECCircuitStats(transpiled_circuit = basic_big, backend = big_backend)
    
    """
