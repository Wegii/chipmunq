from __future__ import annotations

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))
sys.path.append(os.path.join(os.getcwd(), "glue/eccentric_bench/"))

# Custom utils
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.circuit_generator import QECMemory, QECCircuit
from qeccm.backends.backend_utils import plot_circuit_layout
from experiments.exp_utils.simulation_utils import *

from experiments.exp_utils.circuit_utils import stim_to_qiskit
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit
from glue.qiskit_qec.stim_tools import get_stim_circuits_with_detectors

from tqec.utils.noise_model import NoiseModel
import numpy as np
import matplotlib.pyplot as plt

def _get_stim_memory_default():
    """Return stim unrotated surface code patch"""

    circuit_generator = QECMemory(None)
    # Selects the maximum code distance given the number of qubits
    # Note: This is a stim circuit!
    circuit = circuit_generator.generate_code_memory('surface', 1, 1)

    # qiskit_circuit = stim_to_qiskit(circuit)
    return circuit
    
def _get_stim_memory_rotated():
    """Return stim rotated surface code patch """
    
    circuit_generator = QECMemory(None)
    # Selects the maximum code distance given the number of qubits
    # Note: This is a stim circuit!
    circuit = circuit_generator.generate_code_memory('rotated_surface', 1, 1)

    return circuit


def _get_tqec_memory_rotated():
    """Return tqec rotated surface code patch 

    :return: _description_
    :rtype: _type_
    """
    circuit_generator = QECCircuit()
    _, stim_circuit = circuit_generator.single_memory_patch(distance_scale = 1)
  
    return stim_circuit


def simulate_simple_rotated_memory_patch_from_tqec() -> None:
    """Try to compile one single patch of rotated surface code to the backend without deforming it"""

    backend = BackendChipletV2((1, 2, 10, 10), 5, "nn", "rotated_grid")
    circuit = StimCodeCircuit(_get_tqec_memory_rotated()).qc
    with open("circuit_rotated.qc", "w") as f:
       print(circuit, file=f)

    _, custom_circuit, _, _ = transpile_stim_circuit(_get_tqec_memory_rotated(), backend)

    custom_circuit_stim = get_stim_circuits_with_detectors(custom_circuit)[0][0]
    normal_circuit_stim = get_stim_circuits_with_detectors(StimCodeCircuit(stim_circuit = 
                                                                           _get_tqec_memory_rotated()).qc)[0][0]
    
    with open("stim_circuit_rotated.stim", "w") as f:
       print(custom_circuit_stim, file=f)

    plot_circuit_layout(custom_circuit,
                        backend,
                        filename="data/backends/mapping/mapped_circuit_on_backend.png")


def simulate_simple_rotated_memory_patch_from_stim() -> None:
    """Try to compile one single patch of rotated surface code to the backend without deforming it"""

    backend = BackendChipletV2((1, 2, 10, 10), 5, "nn", "rotated_grid")
    circuit = StimCodeCircuit(_get_stim_memory_rotated()).qc
    with open("circuit_rotated.qc", "w") as f:
       print(circuit, file=f)

    _, custom_circuit, _, _ = transpile_stim_circuit(_get_stim_memory_rotated(), backend)

    custom_circuit_stim = get_stim_circuits_with_detectors(custom_circuit)[0][0]
    normal_circuit_stim = get_stim_circuits_with_detectors(StimCodeCircuit(stim_circuit = 
                                                                           _get_stim_memory_rotated()).qc)[0][0]
    
    with open("stim_circuit_rotated.stim", "w") as f:
       print(custom_circuit_stim, file=f)

    plot_circuit_layout(custom_circuit,
                        backend,
                        filename="data/backends/mapping/mapped_circuit_on_backend.png")

    """
    simulation_transpiled_circuit_stats = run_simulation_transpiled_circuit(circuit_generator, backend)

    plot_sinter_stats(simulation_transpiled_circuit_stats,
                      filename = f"experiments/evaluation/mapped_single_patch_rotated.png",
                      with_transpilation = True)
    """


def simulate_simple_memory_patch_from_stim():
    """Map a simple memory patch of unrotated surface code

    :yield: _description_
    :rtype: _type_
    """
    #backend_normal = BackendChipletV2((2, 2, 5, 5), 5)
    backend = BackendChipletV2((2, 2, 10, 5), 5)
    target = backend.target
    coupling_map_backend = target.build_coupling_map()

    # Create memory patch
    circuit = StimCodeCircuit(stim_circuit = _get_stim_memory_default())

    _, custom_circuit, _, sabre_circuit = transpile_stim_circuit(_get_stim_memory_default(), backend)

    deformed_circuit_stim = get_stim_circuits_with_detectors(custom_circuit)[0][0]
    normal_circuit_stim = get_stim_circuits_with_detectors(circuit.qc)[0][0]
    
    #print(normal_circuit_stim)
    with open("stim_circuit_deformed.stim", "w") as f:
        print(deformed_circuit_stim, file=f)

    with open("stim_circuit.stim", "w") as f:
        print(normal_circuit_stim, file=f)

    # Code distance to consider
    ks = [1]

    circuits = {
        # TODO: add observable to measure
        k: (
            #_transpile_to_tqec(lattice_surgery_circuit.single_cnot(distance_scale = k)[1], backend)#[1]
            [deformed_circuit_stim, normal_circuit_stim]
        )
        for k in ks
    }

    # Noise level
    ps = list(np.logspace(-4, -1, 10))
    # TODO: Change to noise model that takes remote gates into consideration
    tqec_noise_model = NoiseModel.si1000

    noisy_circuit = tqec_noise_model(0.003).noisy_circuit(deformed_circuit_stim)
    with open("stim_circuit_noisy.stim", "w") as f:
        print(noisy_circuit, file=f)
    
    # Transpilation
    ts = ["default", "deformed"]

    def _get_sinter_task():
        # Construct sinter task for multiple code distances and noise levels
        yield from (
            sinter.Task(
                circuit=circuit,
                json_metadata={"d": 2 * k + 1, "r": 2 * k + 1, "p": p, "transpilation": t},
            )
            for circuit, k, p, t in (
                (tqec_noise_model(p).noisy_circuit(circuit[0 if t == "deformed" else 1]), k, p, t)
                
                for k, circuit in circuits.items()
                for p in ps
                for t in ts
            )
        )

    stats = run_sinter_simulation(_get_sinter_task, ks, ps)
    plot_sinter_stats(stats,
                    filename = f"experiments/evaluation/mapped_single_patch_unrotated.png",
                    with_transpilation = True)



if __name__ == "__main__":
    # simulate_simple_memory_patch_from_stim()
    
    # simulate_simple_rotated_memory_patch_from_stim()

    simulate_simple_rotated_memory_patch_from_tqec()