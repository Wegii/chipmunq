import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))

from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.circuit_generator import get_tqec_cnot_rotated
from qeccm.backends.backend_utils import plot_circuit_layout, plot_circuit_layout_utilization
from experiments.exp_utils.simulation_utils import *
from qeccm.backends.backend_utils import plot_gate_map

from qiskit import QuantumCircuit


def get_circuit(distance_scale: int, p_icc, amp_icc, num_defective: int = 0) -> QuantumCircuit:

    
    _, backend = get_backend(p_icc = p_icc,
                        amp_icc = amp_icc, 
                        d = distance_scale,
                        num_defective = num_defective)

    circuit, partitions = get_tqec_cnot_rotated(distance_scale = distance_scale,
                                            n1 = 3,
                                            n2 = 0)
    
    # Transpile circuit to backend
    _, routed_circuit_qiskit, _, _ = transpile_stim_circuit(circuit,
                                                    backend,
                                                    pre_defined_partitions = partitions,
                                                    routing_type = "cost",
                                                    routing_alpha = 0,
                                                    routing_beta = 0)
    
    plot_circuit_layout(routed_circuit_qiskit,
                        backend,
                        filename=f"tests/data/figures/qpu_placement/layout_{distance_scale}.png")

    return routed_circuit_qiskit, backend


def get_backend(p_icc: float, amp_icc: float, d: int = -1, num_defective: int = 0) -> BackendChipletV2:    
    if d == 1:
        chiplet_size = (4, 4, 11, 6)
        nic = 5
    elif d == 2:
        chiplet_size = (6, 6, 15, 8)
        nic = 7
    elif d == 3:
        chiplet_size = (6, 6, 19, 10)
        nic = 9
    elif d == 4:
        chiplet_size = (6, 6, 23, 12)
        nic = 11

    backend = BackendChipletV2(size=chiplet_size,
                                n_inter = nic,
                                connectivity = "nn",
                                topology = "rotated_grid",
                                inter_chiplet_noise = p_icc,
                                inter_chiplet_amplification = amp_icc,
                                inter_chiplet_noise_type = "constant",
                                num_defective_qubits=num_defective,
                                )

    plot_gate_map(backend = backend,
                    filename = f"tests/data/figures/qpu_placement/{chiplet_size}.png")
    return nic, backend


def grid_defective_QPU_assignment() -> None:
    # Placement of partitions to a grid of QPUs which contain defects


    routed_circuit, chiplet_backend = get_circuit(distance_scale=1,
                                                  p_icc = 1e-4,
                                                  amp_icc = 1,
                                                  num_defective = 1)

    plot_circuit_layout_utilization(routed_circuit,
                                chiplet_backend,
                                filename="tests/data/figures/qpu_placement/layout_utilization.png")
    
    pass


def grid_QPU_assignment() -> None:
    # Placement of partitions to a grid of QPUs
    
    pass



if __name__ == "__main__":

    # grid_QPU_assignment()

    grid_defective_QPU_assignment()
