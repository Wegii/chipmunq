# Qiskit integration
from qecc_mapping.qeccm.circuit.hypergraph_circuit import HypergraphCircuit

# Implementation
from qecc_mapping.qeccm.src.mar import *
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass


class PartitionedMapRoutePass(TransformationPass):
    """Map input circuit onto a backend topology via insertion of SWAPs.

    The `PartitionedMapRoutePass` pass acts both as a layout stage and a routing stage.

    Tasks:
        - TODO: Find out how to implement this the best way. Probably rewrite the mar class. This class then returns the
                multiple stages

    References:
    https://quantum.cloud.ibm.com/docs/en/guides/create-transpiler-plugin

    """

    def __init__(self):
        super().__init__()

        # TODO: The transpiler passes get a DAG instead of a circuit
        # See: https://quantum.cloud.ibm.com/docs/en/guides/DAG-representation

    def run(self, circuit: DAGCircuit, type: str) -> DAGCircuit:

        # TODO: fix input parameters to adhere to qiskit passmanager
        if type == "basic":
            qc = self._basic_mar(circuit)
        elif type == "partitioned":
            qc = self._partitioned_mar(circuit)

        return qc

    def _partitioned_mar(circuit: DAGCircuit) -> DAGCircuit:

        # TODO: rewrite this in terms of passes

        hg_circuit = HypergraphCircuit(circuit)

        mar = PartitionedMapRoute()

        m_circuit = mar.perform_mapping(hg_circuit)
        mr_circuit = mar.perform_routing(m_circuit)

        return mr_circuit

    def _basic_mar(circuit: DAGCircuit) -> DAGCircuit:
        # Random mapping of logical to physical qubits
        # Simplest routing

        mar = BasicMapRoute()

        m_circuit = mar.perform_mapping(circuit)
        mr_circuit = mar.perform_routing(m_circuit)

        return mr_circuit
