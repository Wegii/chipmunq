from qiskit import QuantumCircuit

from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit


def stim_to_qiskit(stim_circuit) -> QuantumCircuit:
    """Convert stim circuit to qiskit

    wrapper around stim_code_circuit

    Note:
        - not all stim gates can be represented by a qiskit QuantumCircuit. These gates are simply not used
    """
    # TODO: extract detectors and return them

    # Convert stim circuit to qiskit
    stim_code = StimCodeCircuit(stim_circuit=stim_circuit)

    return stim_code
