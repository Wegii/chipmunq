import sys
import os
import logging

import qiskit
# Eccentric_bench for generating memory circuits
sys.path.append(os.path.join(os.getcwd(), "../eccentric_bench/"))
sys.path.append(os.path.join(os.getcwd(), "../eccentric_bench/external/qiskit_qec/src"))
from codes import get_code, get_max_d


class QECMemory():
    """QECC memory circuits
    
    Supported codes:
        - Color Code
        - Surface Code
        - Steane Code
        - BB Codes
    """

    # QECC Memory circuits
    def __init__(self, nq: int, gate_set: list = []):
        self.num_qubits = nq

        self.gate_set = gate_set 

    def draw_circuit(self, circuit: qiskit.QuantumCircuit, filename: str = "") -> None:
        renderer = "mpl"

        if filename == "":
            circuit.draw(output = renderer)
        else:
            circuit.draw(output = renderer, filename = filename)

    def _check_circuit_gateset(self, circuit) -> qiskit.QuantumCircuit:

        # qiskit transpilation step with the gateset
        self.gate_set
        return circuit

    def _generate_code_from_eccentric_bench(self, codename: str) -> qiskit.QuantumCircuit:
        """Generate QECC memory circuit using eccentric_bench library

        :param codename: Name of QEC code to generate
        :type codename: str
        :return: QECC memory circuit
        :rtype: qiskit.QuantumCircuit
        """
        d = get_max_d("surface", self.num_qubits)
        print(d)

        if d < 3:
            logging.error(
                f"Code distance too small! {codename} with distance {d} and {self.num_qubits} qubits: Execution not possible")
            exit(1)

        # Generate code
        cycles = 1#d
        return get_code(codename, d, cycles).qc
    
    def _generate_distributed_code_from_eccentric_bench(self, codename: str) -> qiskit.QuantumCircuit:
        # Simply generate multiple patches of the same code, and concatenate the generated code circuits

        # Generate code

        # Stitch circuits together
        pass

    def generate_code_memory(self, codename: str) -> qiskit.QuantumCircuit:
        """Generate QECC memory circuit

        Currently used as wrapper around the circuit generation function from eccentric_bench. Extend this function if
        some unsupported QEC code should be generated.

        :param codename: Name of QEC code to generate
        :type codename: str
        :return: QECC memory circuit
        :rtype: qiskit.QuantumCircuit
        """
        multi = False

        if multi:
            qecc_mem = self._generate_code_from_eccentric_bench(codename)
        else:
            qecc_mem = self._generate_code_from_eccentric_bench(codename)

        #TODO: Check if circuit only consists of allowed gates, i. e. it is important that the circuit only consists of
        # e. g. two qubit gates as this will influence the hypergraph construction
        # Note: The allowed gateset also needs to be taken into account in the local routing algorithm.
        qecc_mem_transpiled = self._check_circuit_gateset(qecc_mem)

        return qecc_mem_transpiled
