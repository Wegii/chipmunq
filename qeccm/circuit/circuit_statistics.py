from qiskit import QuantumCircuit


class QECCircuitStats():
    """_summary_
    """

    def __init__(self, circuit: QuantumCircuit):
        self.circ = circuit

    def get_num_qubits(self):
        # Number of qubits
        return self.circ.num_qubits

    def get_num_gates(self):
        # Sum of all gates
        return self.get_num_two_gates() + self.get_num_single_gates()
        
    def get_num_remote_gates(self):
        # Get number of remote gates
        # TODO: Check backend how these are defined
        
        pass

    def get_num_two_gates(self):
        # Get number of two-qubit gates
        pass

    def get_num_single_gates(self):
        # Get number of single-qubit gates
        pass
    