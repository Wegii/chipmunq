from __future__ import annotations

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))
sys.path.append(os.path.join(os.getcwd(), "glue/eccentric_bench/"))

# Custom utils
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.circuit_generator import QECMemory, QECCircuit
from qeccm.backends.backend_utils import plot_circuit_layout, plot_circuit_layout_utilization
from qeccm.src.reference_partitions import memory_d5
from experiments.exp_utils.simulation_utils import *

from experiments.exp_utils.circuit_utils import stim_to_qiskit
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit
from glue.qiskit_qec.stim_tools import get_stim_circuits_with_detectors

from tqec.utils.noise_model import NoiseModel
from tqec.computation.block_graph import BlockGraph
from tqec.utils.enums import Basis
from tqec.utils.position import Position3D
import numpy as np
import matplotlib.pyplot as plt


def _run_simulation(circuit: list) -> None:
    """_summary_

    Circuit should be [compiled_circuit_stim, default_circuit_stim]

    :param circuit: _description_
    :type circuit: list
    :yield: _description_
    :rtype: _type_
    """

    # Code distance to consider
    ks = [2]

    circuits = {
        # TODO: add observable to measure
        k: (
            #_transpile_to_tqec(lattice_surgery_circuit.single_cnot(distance_scale = k)[1], backend)#[1]
            circuit
        )
        for k in ks
    }

    # Noise level
    ps = list(np.logspace(-4, -1, 10))
    # TODO: Change to noise model that takes remote gates into consideration
    tqec_noise_model = NoiseModel.si1000

    # Transpilation
    ts = ["default", "deformed"]
    ts = ["default", "compiled"]

    def _get_sinter_task():
        # Construct sinter task for multiple code distances and noise levels
        yield from (
            sinter.Task(
                circuit=circuit,
                json_metadata={"d": 2 * k + 1, "r": 2 * k + 1, "p": p, "transpilation": t},
            )
            for circuit, k, p, t in (
                (tqec_noise_model(p).noisy_circuit(circuit[0 if t == "compiled" else 1]), k, p, t)
                
                for k, circuit in circuits.items()
                for p in ps
                for t in ts
            )
        )

    stats = run_sinter_simulation(_get_sinter_task, ks, ps)

    return stats
    

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
    _, stim_circuit = circuit_generator.single_memory_patch(distance_scale = 2)
  
    return stim_circuit


def _get_tqec_full_memory_rotated():
    
    circuit_generator = QECCircuit()
    stim_circuit = circuit_generator.full_memory_patch(distance_scale = 2)

    with open("stim_circuit_full_memory.stim", "w") as f:
        print(stim_circuit, file=f)
  
    return stim_circuit


def _get_tqec_multiple_memory_rotated():
    circuit_generator = QECCircuit()
    stim_circuit = circuit_generator.multiple_memory_patch(num_x1=2, num_x2=2, distance_scale = 2)

    with open("stim_circuit_rotated_multiple.stim", "w") as f:
        print(stim_circuit, file=f)
  
    return stim_circuit


def _get_tqec_cnot_rotated():
    circuit_generator = QECCircuit()
    stim_circuit, partitions = circuit_generator.single_cnot_full_memory(distance_scale = 2)

    with open("stim_circuit_cnot.stim", "w") as f:
        print(stim_circuit, file=f)
  
    return stim_circuit, partitions


def simulate_single_cnot_from_tqec() -> None:
    #backend = BackendChipletV2((2, 2, 15, 8), 8, "nn", "rotated_grid")
    backend = BackendChipletV2((2, 2, 15, 8), 4, "nn", "rotated_grid")

    circuit, partitions = _get_tqec_cnot_rotated()
    _, custom_circuit, _, _ = transpile_stim_circuit(circuit,
                                                     backend,
                                                     pre_defined_partitions=partitions)
    print("Finished compilation")
    custom_circuit_stim = get_stim_circuits_with_detectors(custom_circuit)[0][0]
    normal_circuit_stim = get_stim_circuits_with_detectors(StimCodeCircuit(circuit).qc)[0][0]
    
    plot_circuit_layout(custom_circuit,
                        backend,
                        filename="data/backends/mapping/mapped_circuit_on_backend.png")
    
    plot_circuit_layout_utilization(custom_circuit,
                                    backend,
                                    filename="experiments/evaluation/single_cnot_rotated_layout_utilization.png")

    #with open("stim_circuit_rotated_single_cnot_compiled.stim", "w") as f:
    #    print(custom_circuit, file=f)

    print("Starting simulation")
    stats = _run_simulation([custom_circuit_stim, normal_circuit_stim])
    plot_sinter_stats(stats,
                filename = f"experiments/evaluation/single_cnot_rotated.png",
                with_transpilation = True)


def simulate_multi_rotated_memory_patch_from_tqec() -> None:
    """Try to compile one single patch of rotated surface code to the backend without deforming it"""

    backend = BackendChipletV2((1, 2, 12, 15), 5, "nn", "rotated_grid")
    circuit = StimCodeCircuit(_get_tqec_multiple_memory_rotated()).qc
    #with open("multi_circuit_rotated.qc", "w") as f:
    #   print(circuit, file=f)
    
    _, custom_circuit, _, _ = transpile_stim_circuit(_get_tqec_multiple_memory_rotated(), backend)


    custom_circuit_stim = get_stim_circuits_with_detectors(custom_circuit)[0][0]
    normal_circuit_stim = get_stim_circuits_with_detectors(StimCodeCircuit(stim_circuit = 
                                                                          _get_tqec_multiple_memory_rotated()).qc)[0][0]
    
    with open("stim_circuit_rotated_multi_compiled.stim", "w") as f:
        print(custom_circuit_stim, file=f)

    plot_circuit_layout(custom_circuit,
                        backend,
                        filename="data/backends/mapping/mapped_circuit_on_backend.png")
    
    #stats = _run_simulation([custom_circuit_stim, normal_circuit_stim])

    #plot_sinter_stats(stats,
    #            filename = f"experiments/evaluation/multi_patch_rotated.png",
    #            with_transpilation = True)


def simulate_simple_rotated_full_memory_patch_from_tqec() -> None:
    """Try to compile one single patch of rotated surface code to the backend without deforming it"""

    backend = BackendChipletV2((1, 2, 12, 12), 5, "nn", "rotated_grid")
    circuit = StimCodeCircuit(_get_tqec_full_memory_rotated()).qc


    _, custom_circuit, _, _ = transpile_stim_circuit(_get_tqec_memory_rotated(),
                                                     backend,
                                                     pre_defined_partitions=memory_d5)

    custom_circuit_stim = get_stim_circuits_with_detectors(custom_circuit)[0][0]
    normal_circuit_stim = get_stim_circuits_with_detectors(StimCodeCircuit(stim_circuit = 
                                                                          _get_tqec_memory_rotated()).qc)[0][0]
    
    with open("stim_circuit_rotated_compiled.stim", "w") as f:
       print(custom_circuit_stim, file=f)

    plot_circuit_layout(custom_circuit,
                        backend,
                        filename="data/backends/mapping/mapped_circuit_on_backend.png")

    #stats = _run_simulation([custom_circuit_stim, normal_circuit_stim])

    #plot_sinter_stats(stats,
    #            filename = f"experiments/evaluation/multi_full_patch_rotated.png",
    #            with_transpilation = True)


def simulate_simple_rotated_memory_patch_from_tqec() -> None:
    """Try to compile one single patch of rotated surface code to the backend without deforming it"""

    backend = BackendChipletV2((1, 2, 12, 12), 5, "nn", "rotated_grid")
    circuit = StimCodeCircuit(_get_tqec_memory_rotated()).qc
    #with open("circuit_rotated.qc", "w") as f:
    #   print(circuit, file=f)

    _, custom_circuit, _, _ = transpile_stim_circuit(_get_tqec_memory_rotated(), backend)

    custom_circuit_stim = get_stim_circuits_with_detectors(custom_circuit)[0][0]
    normal_circuit_stim = get_stim_circuits_with_detectors(StimCodeCircuit(stim_circuit = 
                                                                          _get_tqec_memory_rotated()).qc)[0][0]
    
    with open("stim_circuit_rotated_compiled.stim", "w") as f:
       print(custom_circuit_stim, file=f)

    plot_circuit_layout(custom_circuit,
                        backend,
                        filename="data/backends/mapping/mapped_circuit_on_backend.png")
    
    #stats = _run_simulation([custom_circuit_stim, normal_circuit_stim])

    #plot_sinter_stats(stats,
    #            filename = f"experiments/evaluation/single_patch_rotated.png",
    #            with_transpilation = True)


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
    #_run_simulation([custom_circuit_stim, normal_circuit_stim])


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

    #_run_simulation([deformed_circuit_stim, normal_circuit_stim])


if __name__ == "__main__":
    # simulate_simple_memory_patch_from_stim()
    
    # simulate_simple_rotated_memory_patch_from_stim()

    # simulate_simple_rotated_memory_patch_from_tqec()

    # simulate_simple_rotated_full_memory_patch_from_tqec()

    # simulate_multi_rotated_memory_patch_from_tqec()

    simulate_single_cnot_from_tqec()