import qiskit
import qiskit.circuit
from qiskit.compiler import transpile
from qiskit.transpiler import CouplingMap
import logging


def transpile_circuit_SABRE(circuit: qiskit.circuit, coupling_map: CouplingMap) -> qiskit.circuit:
    """Perform mapping and routing with qiskit-SABRE algorithm

    Args:
        circuit (qiskit.circuit): Circuit to perform algorithm on
        coupling_map (qiskit.CouplingMap): CouplingMap of the backend to transpile to

    Returns:
        qiskit.circuit: Transpiled circuit
    """

    logging.info("Starting algorithm: qiskit-SABRE")
    transpiled_circuit = transpile(circuit, coupling_map=coupling_map, optimization_level=2)
    logging.info("Finished algorithm!")

    return transpiled_circuit
