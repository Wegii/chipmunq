# Qiskit integration
import qiskit
from qiskit.transpiler.basepasses import TransformationPass

# Implementation
from qecc_mapping.qeccm.src.mar import *
from qecc_mapping.qeccm.circuit.hypergraph_circuit import HypergraphCircuit


class PartitionedMapRoute(TransformationPass):
    """ Map input circuit onto a backend topology via insertion of SWAPs.
    
    References:
    SAD
    """

    def __init__(self):
        super().__init__()

    def run(self, circuit: qiskit.QuantumCircuit, type: str) -> qiskit.QuantumCircuit:
        if type == "basic":
            qc = _basic_mar(circuit)
        elif type == "partitioned":
            qc = _partitioned_mar(circuit)

        return qc

    def _partitioned_mar(circuit: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    
        hg_circuit = HypergraphCircuit(circuit)

        mar = PartitionedMapRoute()

        m_circuit = mar.perform_mapping(hg_circuit)
        mr_circuit = mar.perform_routing(m_circuit)

        return mr_circuit


    def _basic_mar(circuit: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
        # Random mapping of logical to physical qubits
        # Simplest routing

        mar = BasicMapRoute()

        m_circuit = mar.perform_mapping(circuit)
        mr_circuit = mar.perform_routing(m_circuit)
        
        return mr_circuit