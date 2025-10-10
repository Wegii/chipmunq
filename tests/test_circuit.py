import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))

# Circuits
from experiments.circuit_generator import QECMemory

# Qiskit DAG
from qiskit.converters import circuit_to_dag
from qiskit.visualization.dag_visualization import dag_drawer


def test_surface_memory_circuit_generation():
    # Minimum number of qubits for distance 3 surface code
    num_qubits = 26

    # Generate surface code memory circuit
    circuit_generator = QECMemory(num_qubits)
    surface_memory_circuit = circuit_generator.generate_code_memory('surface')

    # Draw circuit to file
    circuit_generator.draw_circuit(surface_memory_circuit, "data/circuits/surface_memory.png")

    # Draw circuit as DAG to file
    dag = circuit_to_dag(surface_memory_circuit)
    dag_drawer(dag, filename="data/circuits/surface_memory_dag.png")


if __name__ == "__main__":
    test_surface_memory_circuit_generation()
