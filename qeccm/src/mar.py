import abc
from qecc_mapping.qeccm.circuit.hypergraph_circuit import HypergraphCircuit
from qecc_mapping.qeccm.src.partitioners import *
import qiskit


class GenericMapRoute(metaclass=abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def perform_mapping(self):
        raise NotImplementedError

    @abc.abstractmethod
    def perform_routing(self):
        raise NotImplementedError


class BasicMapRoute(GenericMapRoute):
    def __init__(self):
        pass

    def perform_mapping(self, circuit: qiskit.QuantumCircuit) -> None:
        # Random mapping

        pass

    def perform_routing(self):
        # SABRE
        pass


class PartitionedMapRoute(GenericMapRoute):

    def __init__(self):
        pass

    def perform_mapping(self, hgc: HypergraphCircuit):

        # The hypergraph circuit has multiple edges, since multigraph=True
        # Remove these duplicates, since these are not needed in the partitioning
        # In the local mapping these can be again quite interesting

        kahypar_partitioner = KaHyParPartitioning()

        # Calculate number of partitions based on circuit and backend
        kp = kahypar_partitioner.calculate_partitions()

        # Partitioned mapping

    def perform_routing(self):
        # SABRE
        pass