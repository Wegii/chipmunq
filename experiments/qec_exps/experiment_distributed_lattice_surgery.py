# How does the logical error rate of lattice surgery operations change when distributed to multiple chiplets?


from __future__ import annotations

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))
sys.path.append(os.path.join(os.getcwd(), "glue/eccentric_bench/"))

# Custom utils
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.circuit_generator import QECCircuit
from experiments.exp_utils.simulation_utils import *

from experiments.exp_utils.circuit_utils import stim_to_qiskit
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit
from glue.qiskit_qec.stim_tools import get_stim_circuits_with_detectors

from tqec.utils.noise_model import NoiseModel
import numpy as np
import matplotlib.pyplot as plt

def _get_stim_single_cnot():
    # Get circuit with non-interacting patches
    
    lattice_surgery_circuit = QECCircuit()
    qiskit_circuit, stim_circuit = lattice_surgery_circuit.single_cnot(distance_scale = 1)
  
    return stim_circuit


def simulate_simple_memory_patch() -> None:
    """Try to compile one single patch to the backend without deforming it"""
    backend = BackendChipletV2((2, 2, 5, 6), 5)

    circuit_generator = _get_stim_single_cnot()
    simulation_transpiled_circuit_stats = run_simulation_transpiled_circuit(circuit_generator, backend)

    plot_sinter_stats(simulation_transpiled_circuit_stats,
                      filename = f"experiments/evaluation/mapped_single_patch_no_deformation.png",
                      with_transpilation = True)




if __name__ == "__main__":
    simulate_simple_memory_patch()