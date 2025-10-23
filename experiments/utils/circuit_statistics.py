from qiskit import QuantumCircuit

# Typing
from eccentric_bench.external.qiskit_qec.src.qiskit_qec.circuits.stim_code_circuit import StimCodeCircuit
from qiskit.providers import BackendV2

# eccentric_bench
from glue.eccentric_bench.backends import QubitTracking
from glue.eccentric_bench.noise import get_noise_model
from glue.eccentric_bench.decoders import decode

# TODO: fix this import
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../eccentric_bench/external/qiskit_qec/src/"))
from qiskit_qec.utils import get_stim_circuits



class QECCircuitStats():
    """_summary_
    """

    def __init__(self, transpiled_circuit: QuantumCircuit, stim_circuit: StimCodeCircuit = None,
                backend: BackendV2 = None):

        if stim_circuit != None:
            # To perform simulation
            self.stim_circuit = stim_circuit

            # Transpiled stim circuit
            detectors, logicals = self.stim_circuit.stim_detectors()
            self.transpiled_stim_circuit = get_stim_circuits(transpiled_circuit, detectors=detectors, logicals=logicals)[0][0]

        if backend != None:
            self.backend = backend

        # For calculating gate statistics. Note: stim_circuit.qc contains the circuit before the transpiler passes,
        # while transpiled_circuit is the transpiled version
        self.circ = transpiled_circuit


    def get_logical_error_rate(self):


        error_type = ""
        error_prob = ""
        code_name = ""
        decoder = ""
        backend_name = ""

        qt = QubitTracking(self.backend, self.stim_circuit.qc)
        print("After GET STIM CIRCUIT")

        noise_model = get_noise_model(error_type, qt, error_prob, self.backend)

        print("After get_noise_model")
        stim_circuit = noise_model.noisy_circuit(stim_circuit)

        print("After adding noise")
        print("before decoding")

        error_occured = decode(code_name, stim_circuit, 1, decoder, backend_name, error_type)
        print("After decoding")

        #if error_occured == None:
        #    exit(1)
        #pass

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
    