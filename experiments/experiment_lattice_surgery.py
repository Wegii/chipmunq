from __future__ import annotations

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))

# Enable debug logging for Qiskit
#import logging
#logging.basicConfig(level=logging.DEBUG)
import time

# Custom utils
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.circuit_generator import QECCircuit
from experiments.exp_utils.circuit_statistics import LatticeSurgeryStats

# tqec
from tqec.utils.enums import Basis

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


def _get_circuit(type, distance_scale: int = 1):
    # Distance must be multiples of 3

    # Get circuit with non-interacting patches
    lattice_surgery_circuit = QECCircuit()

    if type == "cnot":
        qiskit_circuit, stim_circuit = lattice_surgery_circuit.single_cnot(distance_scale = distance_scale)
    elif type == "two_cnot":
        pass
    elif type == "steane":
        pass
  
    return qiskit_circuit, stim_circuit


if __name__ == "__main__":
    n_inter = 5

    small_backend = BackendChipletV2((2, 2, 10, 10), n_inter)
    medium_backend = BackendChipletV2((8, 8, 10, 10), n_inter)

    # Single CNOT circuit:
    single_cnot_qiskit_circuit, single_cnot_stim_circuit = _get_circuit(type = "cnot",
                                                                        distance_scale = 1)
    
    custom_circuit, sabre_circuit = _transpile(single_cnot_qiskit_circuit.qc, small_backend)

    #print(single_cnot_qiskit_circuit.qc.depth())
    #print(single_cnot_qiskit_circuit.qc)
    #print(single_cnot_stim_circuit)

    ls_stats = LatticeSurgeryStats(custom_circuit, single_cnot_qiskit_circuit, small_backend)
    print(single_cnot_stim_circuit)
    print(ls_stats.transpiled_stim_circuit)
    #print(ls_stats.test_logical_error_rate())

    #with open("experiments/data/tqec/tqec_to_stim_to_qiskit_to_stim.txt", "w") as f:
    #    print(ls_stats.transpiled_stim_circuit, file=f)
    
    #with open("experiments/data/tqec/tqec_to_stim.txt", "w") as f:
    #    print(single_cnot_stim_circuit, file=f)


    #ls_stats.get_logical_error_rate(Basis.Z)

