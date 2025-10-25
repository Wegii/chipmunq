import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))

# Circuits
from experiments.exp_utils.circuit_generator import QECMemory
# Hypergraph
from qeccm.circuit.hypergraph_circuit import HypergraphCircuit
# Qiskit
from qiskit.converters import circuit_to_dag
# Custom transpiler plugin
from qeccm.src.mar import PartitionedMapRoutePlugin


def test_surface_memory_circuit_to_hypergraph():
    # Minimum number of qubits for distance 3 surface code
    num_qubits = 26 

    # Generate surface code memory circuit
    circuit_generator = QECMemory(num_qubits)
    surface_memory_circuit = (circuit_generator.generate_code_memory('surface')).qc
    
    mar_pmsp = PartitionedMapRoutePlugin()
    # Construct hypergraph from circuit
    init_pm = mar_pmsp._generate_initial_pass()
    _ = init_pm.run(surface_memory_circuit)
    

if __name__ == "__main__":
    test_surface_memory_circuit_to_hypergraph()