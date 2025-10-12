import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))

from experiments.circuit_generator import QECMemory
from qeccm.circuit.hypergraph_circuit import HypergraphCircuit
from qeccm.src.mar import PartitionedMapRoute
from qeccm.backends.backend import GenericChipletBackend
# Qiskit
from qiskit.converters import circuit_to_dag



def test_surface_memory_circuit_to_hypergraph_partitioning_mapping():
    """Test mapping of hypergraph"""

    # Minimum number of qubits for distance 3 surface code
    num_qubits = 26 

    # Generate surface code memory circuit
    circuit_generator = QECMemory(num_qubits)
    surface_memory_circuit = circuit_generator.generate_code_memory('surface')

    # Convert circuit to DAG circuit. In the qiskit transpilation passes, 
    surface_memory_circuit_dag = circuit_to_dag(surface_memory_circuit)
    # Construct hypergraph from circuit
    hgc = HypergraphCircuit(surface_memory_circuit_dag)

    # Initialize backend to map to
    chiplet_backend = GenericChipletBackend((2, 5, 5), 1)
    chiplet_backend.build_backend()

    # Partition
    mar = PartitionedMapRoute(hgc)
    # Set number of partitions
    k = 2
    # Perform mapping
    mar.perform_mapping(chiplet_backend, kp=k)    
    
    # Visualize the mapping 
    mar.mapper.visualize_mapping(filename = "data/backends/surface_memory_mapped_backend.png")


if __name__ == "__main__":
    test_surface_memory_circuit_to_hypergraph_partitioning_mapping()