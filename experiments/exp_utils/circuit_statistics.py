import os
import sys
#sys.path.append(os.path.join(os.getcwd(), "../eccentric_bench/"))
#sys.path.append(os.path.join(os.getcwd(), "../eccentric_bench/external/qiskit_qec/src/"))
#sys.path.append(os.path.join(os.getcwd(), "../eccentric_bench/external/qiskit_qec/"))
#from qiskit_qec.utils import get_stim_circuits
sys.path.append(os.path.join(os.getcwd(), "glue/eccentric_bench/"))

from qiskit import QuantumCircuit

# Typing
from collections.abc import Callable, Iterable, Iterator
from qiskit.providers import BackendV2

# Qiskit to stim translation
from glue.qiskit_qec.stim_tools import get_stim_circuits_with_detectors

# eccentric_bench
from glue.eccentric_bench.backends import QubitTracking
#from glue.eccentric_bench.noise import get_noise_model
#from glue.eccentric_bench.decoders import decode


class QECCircuitStats():
    """_summary_
    """

    def __init__(self, transpiled_circuit: QuantumCircuit, stim_circuit = None,
                backend: BackendV2 = None):

        if stim_circuit != None:
            # To perform simulation
            self.stim_circuit = stim_circuit

            # Transpiled stim circuit
            detectors, logicals = self.stim_circuit.stim_detectors()
            self.transpiled_stim_circuit = get_stim_circuits(transpiled_circuit,
                                                             detectors=detectors,
                                                             logicals=logicals)[0][0]

        if backend != None:
            self.backend = backend

        # For calculating gate statistics. Note: stim_circuit.qc contains the circuit before the transpiler passes,
        # while transpiled_circuit is the transpiled version
        self.circ = transpiled_circuit


    def get_logical_error_rate(self, num_samples):

        error_type = "modsi1000"
        error_prob = 5e-3
        code_name = "surface"
        decoder = "bposd"
        backend_name = "custom_chiplet"

        qt = QubitTracking(self.backend, self.circ)
        print("After GET STIM CIRCUIT")

        noise_model = get_noise_model(error_type, qt, error_prob, self.backend)

        print("After get_noise_model")
        stim_circuit = noise_model.noisy_circuit(self.transpiled_stim_circuit)

        print("After adding noise")
        print("before decoding")

        error_occured = decode(code_name, stim_circuit, num_samples, decoder, backend_name, error_type)
        print("After decoding")

        logical_error_rate = error_occured / num_samples

        return logical_error_rate


    def get_num_qubits(self):
        """Number of qubits in circuit.

        :return: Number of qubits in circuit
        :rtype: _type_
        """

        # Number of qubits
        return self.circ.num_qubits

    def get_num_gates(self):
        """Sum of two-and single-qubit gates

        :return: _description_
        :rtype: _type_
        """

        # Sum of all gates
        return self.get_num_two_gates() + self.get_num_single_gates()
        
    def get_num_remote_gates(self):
        # Get number of remote gates
        # TODO: Check backend how these are defined
        
        pass

    def get_num_two_gates(self) -> int:
        """Calculate number of two-qubit gates.

        :return: Number of two-qubit gates
        :rtype: int
        """

        return sum(1 for instr, qargs, cargs in self.circ.data if len(qargs) == 2)


    def get_num_single_gates(self):
        # Get number of single-qubit gates
        pass

    def get_depth(self) -> int:
        """Calculate depth of circuit.

        :return: Depth of circuit
        :rtype: int
        """

        return self.circ.depth()
    


from multiprocessing import cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
import sinter

from tqec.gallery.cnot import cnot
from tqec import NoiseModel
from tqec.simulation.plotting.inset import plot_observable_as_inset
from tqec.simulation.simulation import start_simulation_using_sinter
from tqec.utils.enums import Basis

class SingleLatticeSurgeryStats():
    """Statistics of a single circuit with Lattice Surgery
    """

    def __init__(self, transpiled_circuit: QuantumCircuit, stim_circuit = None,
                backend: BackendV2 = None):

        if stim_circuit != None:
            # To perform simulation
            self.stim_circuit = stim_circuit

            # Transpiled stim circuit
            self.transpiled_stim_circuit = get_stim_circuits_with_detectors(transpiled_circuit)[0][0]

        if backend != None:
            self.backend = backend

        # For calculating gate statistics. Note: stim_circuit.qc contains the circuit before the transpiler passes,
        # while transpiled_circuit is the transpiled version
        self.circ = transpiled_circuit

    def test_logical_error_rate(self) -> float:
        """Run simple decoding task for calculating logical error rate

        :return: _description_
        :rtype: float
        """
        
        error_type = "modsi1000"
        error_prob = 5e-2
        code_name = "surface"
        decoder = "mwpm"
        backend_name = "custom_chiplet"
        num_samples = 1_000_000

        qt = QubitTracking(self.backend, self.circ)
        noise_model = get_noise_model(error_type, qt, error_prob, self.backend)
        stim_circuit = noise_model.noisy_circuit(self.transpiled_stim_circuit)

        error_occured = decode(code_name, stim_circuit, num_samples, decoder, backend_name, error_type)

        logical_error_rate = error_occured / num_samples
        return logical_error_rate

