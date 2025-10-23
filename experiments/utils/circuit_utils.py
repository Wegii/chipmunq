from qiskit import QuantumCircuit

import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../eccentric_bench/external/qiskit_qec/src/"))
from qiskit_qec.circuits.stim_code_circuit import StimCodeCircuit


def stim_to_qiskit(stim_circuit) -> QuantumCircuit:
    """ Convert stim circuit to qiskit

    wrapper around stim_code_circuit
    
    Note:
        - not all stim gates can be represented by a qiskit QuantumCircuit. These gates are simply not used
    """

    # TODO: extract detectors and return them

    # Convert stim circuit to qiskit
    stim_code = StimCodeCircuit(stim_circuit = stim_circuit)
    
    return stim_code.qc


def qiskit_to_stim():
    """Convert qiskit circuit to stim circuit

    Wrapper around qiskit_qec.utils.get_stim_circuits

    
    References:
        - https://qiskit-community.github.io/qiskit-qec/stubs/qiskit_qec.utils.get_stim_circuits.html
    """

    # TODO: See https://github.com/aswierkowska/eccentric_bench/blob/main/main.py how to add the detectors again

    pass


def qiskit_to_pyzx():
    pass

def pyzx_to_tqec():
    # IMplement necessary steps to go from pyzx to tqec

    # This utilizes topologiq library

    # Note: topologiq wants to use quite old version of libraries. This can potentially be ignored

    pass


