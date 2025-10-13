import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))

# Circuits
from experiments.circuit_generator import QECMemory
# Backend
from qeccm.backends.backend import GenericChipletBackend
# Qiskit Transpiler
from qiskit.transpiler import StagedPassManager
# Custom transpiler plugin
from qeccm.src.mar import PartitionedMapRoutePlugin


def test_surface_memory_circuit_to_hypergraph_partitioning_mapping():
    """Test partitioning and mapping of hypergraph"""

    # Minimum number of qubits for distance 3 surface code
    num_qubits = 26 

    # Generate surface code memory circuit
    circuit_generator = QECMemory(num_qubits)
    surface_memory_circuit = circuit_generator.generate_code_memory('surface')

    # Initialize backend to map to
    chiplet_backend = GenericChipletBackend((2, 5, 5), 1)
    chiplet_backend.build_backend()
    
    mar_pmsp = PartitionedMapRoutePlugin()
    # Construct hypergraph from circuit
    init_pm = mar_pmsp._generate_initial_pass()

    # Perform partition and mapping
    partitioning_pm = mar_pmsp._generate_layout_pass(chiplet_backend)

    staged_pm = StagedPassManager(stages=["init", "layout"], init=init_pm, layout=partitioning_pm)
    _ = staged_pm.run(surface_memory_circuit)

    #     mar.mapper.visualize_mapping(filename = "data/backends/surface_memory_mapped_backend.png")


if __name__ == "__main__":
    test_surface_memory_circuit_to_hypergraph_partitioning_mapping()

    # test_surface_memory_circuit_to_hypergraph_partition_calculation()