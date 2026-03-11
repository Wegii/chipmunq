import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))

# Backend
from qeccm.backends.BackendChipletV2 import BackendChipletV2
from qeccm.backends.backend_utils import plot_circuit_layout

# Transpilation
import qiskit
from qiskit.transpiler import StagedPassManager
# Custom transpiler plugin
from qeccm.src.mar import PartitionedMapRoutePlugin
from glue.qiskit_qec.stim_tools import get_stim_circuits_with_detectors
from experiments.exp_utils.circuit_noise import get_noise_model

import stim
# TQEC noise model
from tqec.utils.noise_model import NoiseModel


def check_tqec_noise_model():
    """ Test the noise model from tqec.
    
    Check what kind of constraints and what structure a stim circuit must adhere to.
    """

    # Circuit that should fail, since one instruction uses some qubits multiple times
    stim_incorrect_circuit = stim.Circuit('''
            SWAP 14 15 13 14 12 13 11 12 10 11 15 16 14 15 13 14
            ''')
    
    stim_correct_circuit = stim.Circuit('''
            SWAP 14 15
            TICK
            SWAP 13 14
            TICK
            SWAP 12 13
            TICK
            SWAP 11 12
            TICK
            SWAP 10 11
            TICK
            SWAP 15 16
            TICK
            SWAP 14 15
            TICK
            SWAP 13 14
            ''')
    
    p = 1e-4
    tqec_noise_model = NoiseModel.uniform_depolarizing

    # This should throw an error
    try:
        _ = tqec_noise_model(p).noisy_circuit(stim_incorrect_circuit)
        print("Noisy circuit constructed successfully")
    except ValueError: print("Operation collisions")

    try:
        _ = tqec_noise_model(p).noisy_circuit(stim_correct_circuit)
        print("Noisy circuit constructed successfully")
    except ValueError as e: print(e)


def check_remote_connection_noise() -> None:
    """Test noise addition to remote connections between chiplets

    Inter-chiplet connections are treated seperately from intra-chiplet connections, since these have a much higher
    noise level.
    """

    chiplet_backend = BackendChipletV2((1, 2, 6, 6), 6, topology="rotated_grid")

    # Place patches on two QPUs and utilize inter-chiplet connections
    # Circuit with two CAT states
    qc_patch = qiskit.QuantumCircuit(50)

    # 1. CAT state
    for i in range(24):
        qc_patch.cx(i, i + 1)  # Chain of CNOTs

    # 2. CAT state
    for i in range(25, 49):
        qc_patch.cx(i, i + 1)

    # Utilize inter-chiplet connection
    qc_patch.cx(5, 34)


    mar_pmsp = PartitionedMapRoutePlugin()
    # Construct hypergraph from circui[t
    init_pm = mar_pmsp._generate_initial_pass()
    # Perform partition and mapping
    partitioning_pm = mar_pmsp._generate_layout_pass(chiplet_backend)
    # Perform routing
    routing_pm = mar_pmsp._generate_routing_pass(chiplet_backend)

    staged_pm = StagedPassManager(stages=["init", "layout", "routing"], init=init_pm, layout=partitioning_pm,
                                  routing=routing_pm)

    # Compile circuit to backend
    routed_circuit = staged_pm.run(qc_patch)
    plot_circuit_layout(routed_circuit, chiplet_backend, filename="data/backends/mapping/mapped_circuit_on_backend.png")

    # Convert to stim
    stim_routed_circuit = get_stim_circuits_with_detectors(routed_circuit)[0][0]

    with open("stim_circuit_transpiled.stim", "w") as f:
        print(stim_routed_circuit, file=f)
    
    # TODO: add noise
    p = 0.5
    inter_chiplet_noise = chiplet_backend.get_inter_chiplet()
    print(inter_chiplet_noise)
    
    noise_model = get_noise_model("si1000",
                                  None,
                                  p,
                                  None,
                                  remote = inter_chiplet_noise)
    noisy_stim_routed_circuit = noise_model.noisy_circuit(stim_routed_circuit)

    with open("stim_circuit_transpiled_noisy.stim", "w") as f:
        print(noisy_stim_routed_circuit, file=f)





if __name__ == "__main__":
    # check_tqec_noise_model()

    check_remote_connection_noise()