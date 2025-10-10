import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))

# Circuits
from experiments.circuit_generator import QECMemory
# Hypergraph
from qeccm.circuit.hypergraph_circuit import HypergraphCircuit
# Qiskit
from qiskit.converters import circuit_to_dag


def test_surface_memory_circuit_to_hypergraph():
    # Minimum number of qubits for distance 3 surface code
    num_qubits = 26 

    # Generate surface code memory circuit
    circuit_generator = QECMemory(num_qubits)
    surface_memory_circuit = circuit_generator.generate_code_memory('surface')

    # Convert circuit to DAG circuit
    surface_memory_circuit_dag = circuit_to_dag(surface_memory_circuit)
    # Construct hypergraph from circuit
    hgc = HypergraphCircuit(surface_memory_circuit_dag)
    # Draw hypergraph
    hgc.draw_hg("data/circuits/surface_memory_hg.png")
    

if __name__ == "__main__":
    test_surface_memory_circuit_to_hypergraph()