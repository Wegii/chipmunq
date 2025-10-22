import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))

# Circuits
from experiments.utils.circuit_generator import QECMemory
# Backend
from qeccm.backends.BackendChipletV2 import BackendChipletV2
from qeccm.backends.backend_utils import plot_gate_map, plot_circuit_layout
# Qiskit Transpiler
from qiskit.transpiler import StagedPassManager
import qiskit
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


if __name__ == "__main__":
    test_partition_well_defined()