import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))

# Circuits
from experiments.circuit_generator import QECMemory
# Hypergraph
from qeccm.circuit.hypergraph_circuit import HypergraphCircuit
# Qiskit
from qiskit.converters import circuit_to_dag
# Partitioning
from qeccm.src.mar import PartitionedMapRoute


def test_surface_memory_circuit_to_hypergraph_partition_calculation():
    """Test calculation of partions given circuit and backend"""

    # Minimum number of qubits for distance 3 surface code
    num_qubits = 26 

    # Generate surface code memory circuit
    circuit_generator = QECMemory(num_qubits)
    surface_memory_circuit = circuit_generator.generate_code_memory('surface')

    # Convert circuit to DAG circuit. In the qiskit transpilation passes, 
    surface_memory_circuit_dag = circuit_to_dag(surface_memory_circuit)
    # Construct hypergraph from circuit
    hgc = HypergraphCircuit(surface_memory_circuit_dag)

    # Partition
    mar = PartitionedMapRoute(hgc)
    k = mar.kahypar_partitioner.calculate_partitions()

    

def test_surface_memory_circuit_to_hypergraph_partitioning():
    """Test partitioning of hypergraph"""

    # Minimum number of qubits for distance 3 surface code
    num_qubits = 26 

    # Generate surface code memory circuit
    circuit_generator = QECMemory(num_qubits)
    surface_memory_circuit = circuit_generator.generate_code_memory('surface')

    # Convert circuit to DAG circuit. In the qiskit transpilation passes, 
    surface_memory_circuit_dag = circuit_to_dag(surface_memory_circuit)
    # Construct hypergraph from circuit
    hgc = HypergraphCircuit(surface_memory_circuit_dag)

    # Partition
    mar = PartitionedMapRoute(hgc)
    # Set number of partitions
    k = 2
    mar.perform_mapping(kp=k)

    # Draw partitioned graph
    #mar.partitioned_hgc.draw()
    

if __name__ == "__main__":
    test_surface_memory_circuit_to_hypergraph_partitioning()

    # test_surface_memory_circuit_to_hypergraph_partition_calculation()