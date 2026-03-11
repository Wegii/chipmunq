import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))

# Circuits
from experiments.exp_utils.circuit_generator import QECCircuit, QECMemory

# Qiskit DAG
from qiskit.converters import circuit_to_dag
from qiskit.visualization.dag_visualization import dag_drawer


def test_surface_memory_circuit_generation():
    # Minimum number of qubits for distance 3 surface code
    num_qubits = 26

    # Generate surface code memory circuit
    circuit_generator = QECMemory(num_qubits)
    surface_memory_circuit = (circuit_generator.generate_code_memory('surface')).qc

    # Draw circuit to file
    circuit_generator.draw_circuit(surface_memory_circuit, "data/circuits/surface_memory.png")

    # Draw circuit as DAG to file
    dag = circuit_to_dag(surface_memory_circuit)
    dag_drawer(dag, filename="data/circuits/surface_memory_dag.png")


def test_logical_circuit_generation():
    # Minimum number of qubits for distance 3 surface code
    num_qubits = 26

    # Generate surface code memory circuit
    circuit_generator = QECCircuit()
    circ = circuit_generator.simple_circuit()

    #print(cir)
    print(dict(circ.count_ops()))
    print("Number of qubits:", circ.num_qubits)
    #circ.draw(output="mpl", fold=-1, filename="data/circuits/QECCircuit/test.png")
    #circ.draw(output="mpl", filename="data/circuits/QECCircuit/test.png")


def _output_to_file(content: str, filename: str = "") -> None:
    """Write a string to a file.

    If file exists, it will be overwritten.

    :param content: Content to write to file
    :type content: str
    :param filename: File to write to, defaults to ""
    :type filename: str, optional
    """

    if filename != "":
        with open(filename, "w") as f:
            print(content, file=f)


def test_extended_cnot():
    ks = 1
    circuit_generator = QECCircuit()
    circuit = circuit_generator.single_cnot_full_memory_extended(distance_scale=ks)
    
    _output_to_file(circuit, "tests/data/circuits/extended_cnot.stim")


if __name__ == "__main__":
    #test_surface_memory_circuit_generation()

    #test_logical_circuit_generation()

    test_extended_cnot()