import abc
from qeccm.circuit.hypergraph_circuit import HypergraphCircuit
from qeccm.circuit.partitioners import KaHyParPartitioning
import qiskit


class GenericMapRoute(abc.ABC):

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

    # Not all qubits have the same connectivity
    # Only certain qubits are directly connected to the other chiplets
    # Each partitioning this needs to have enough qubits that can be connected to other partitions

    def __init__(self, hgc):
        self.hgc = hgc
        self.partitioned_hgc = None

        self.kahypar_partitioner = KaHyParPartitioning(hgc)

    def perform_mapping(self, kp: int = None) -> None:

        # The hypergraph circuit has multiple edges, since multigraph=True
        # Remove these duplicates, since these are not needed in the partitioning
        # In the local mapping these can be again quite interesting

        if kp is None:
            # Calculate number of partitions based on circuit and backend
            self.kahypar_partitioner.k = self.kahypar_partitioner.calculate_partitions()
        else:
            self.kahypar_partitioner.k = kp

        # Partitioned mapping
        self.kahypar_partitioner.run()

        # TODO: generate hypergraph from partition indices


    def perform_routing(self):
        # SABRE
        pass

    def draw_partitioned_hg(self):
        self.partitioned_hgc
        pass