import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))

# Circuits
from experiments.exp_utils.circuit_generator import QECMemory
# Backend
from qeccm.backends.BackendChipletV2 import BackendChipletV2
from qeccm.backends.backend_utils import plot_gate_map, plot_circuit_layout
# Qiskit Transpiler
import qiskit
from qiskit.transpiler import StagedPassManager
from qiskit import QuantumCircuit, QuantumRegister

# Custom transpiler plugin
from qeccm.src.mar import PartitionedMapRoutePlugin

# TODO: Test partitioning based on ill-defined circuits (without any patches) and well-defined circuits (with patches)


def test_partition_ill_defined():
    """Ill-defined quantum circuit with no patches in the circuit """
    pass


def test_partition_well_defined():
    """Well-defined quantum circuit with clear patches in the circuit """

    # Circuit with two GHZ states
    qc_patch = qiskit.QuantumCircuit(25)

    # 1. GHZ state
    qc_patch.h(0)
    for i in range(11):
        qc_patch.cx(i, i + 1)  # Chain of CNOTs

    # 2. GHZ state
    qc_patch.h(12)
    for i in range(12, 24):
        qc_patch.cx(i, i + 1)

    # Connect state
    qc_patch.cx(6, 18)
    qc_patch.cx(0, 12)
    qc_patch.cx(11, 24)

    print(qc_patch)

    chiplet_backend = BackendChipletV2((2, 2, 3, 4), 1)

    mar_pmsp = PartitionedMapRoutePlugin()
    # Construct hypergraph from circuit
    init_pm = mar_pmsp._generate_initial_pass()
    # Perform partition and mapping
    partitioning_pm = mar_pmsp._generate_layout_pass(chiplet_backend)

    staged_pm = StagedPassManager(stages=["init", "layout"], init=init_pm, layout=partitioning_pm)
    mapped_circuit = staged_pm.run(qc_patch)


    print(mapped_circuit)

    
    # print mapping
    plot_circuit_layout(mapped_circuit, chiplet_backend, filename="data/backends/mapping/mapped_circuit_on_backend.png")


def test_partition_on_chip() -> None:
    """Test if multiple patches are mapped to one chip, if they fit

    This tests the patches to qpu mapping
    """

    from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit

    num_qubits = 25
    circuit_generator = QECMemory(num_qubits)
    # Selects the maximum code distance given the number of qubits
    # Note: This is a stim circuit!
    circuit = circuit_generator.generate_code_memory('surface', 1, 1)
    distance_3_stim_circuit = StimCodeCircuit(stim_circuit = circuit)
    
    # Duplicate the surface code path
    new_qr = QuantumRegister(75, 'q')
    duplicated_distance_3_stim_circuit_qr = QuantumCircuit(new_qr)
    duplicated_distance_3_stim_circuit_qr.compose(distance_3_stim_circuit.qc, qubits=list(range(25)), inplace=True)
    duplicated_distance_3_stim_circuit_qr.compose(distance_3_stim_circuit.qc, qubits=list(range(25, 50)), inplace=True)
    duplicated_distance_3_stim_circuit_qr.compose(distance_3_stim_circuit.qc, qubits=list(range(50, 75)), inplace=True)

    #for i in range(20, 24):
    #    duplicated_distance_3_stim_circuit_qr.cz(i, i+25)
    duplicated_distance_3_stim_circuit_qr.cz(24, 25)
    #duplicated_distance_3_stim_circuit_qr.cz(49, 70)
    #duplicated_distance_3_stim_circuit_qr.cz(48, 69)

    # Generic chiplet
    chiplet_backend = BackendChipletV2((2, 2, 8, 12), 1)

    mar_pmsp = PartitionedMapRoutePlugin()
    init_pm = mar_pmsp._generate_initial_pass()
    partitioning_pm = mar_pmsp._generate_layout_pass(chiplet_backend)
    staged_pm = StagedPassManager(stages=["init", "layout"], init=init_pm, layout=partitioning_pm)
    mapped_circuit = staged_pm.run(duplicated_distance_3_stim_circuit_qr)

    plot_circuit_layout(mapped_circuit, chiplet_backend, filename="data/backends/mapping/on_chip_partitioning.png")
        

if __name__ == "__main__":
    # test_partition_well_defined()

    test_partition_on_chip()