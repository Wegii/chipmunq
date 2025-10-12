import abc

import qiskit
import qiskit.dagcircuit
from qiskit.providers import BackendV2

from qeccm.src.partitioners import KaHyParPartitioning
from qeccm.src.mapper import CongestionMapper, RandomMapper


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

    def __init__(self, hgc: qiskit.dagcircuit):
        self.hgc = hgc
        self.partitioned_hgc = None

        self.mapping = None

        self.kahypar_partitioner = KaHyParPartitioning(hgc)
        self.mapper = RandomMapper()

    def perform_mapping(self, backend: BackendV2, kp: int = None) -> None:

        # The hypergraph circuit has multiple edges, since multigraph=True
        # Remove these duplicates, since these are not needed in the partitioning
        # In the local mapping these can be again quite interesting

        if kp is None:
            # Calculate number of partitions based on circuit and backend
            self.kahypar_partitioner.k = self.kahypar_partitioner.calculate_partitions()
        else:
            self.kahypar_partitioner.k = kp

        # Partitioned mapping
        self.partitioned_hgc = self.kahypar_partitioner.run()

        # Perform mapping
        self.mapping = self.mapper.perform_mapping(backend, self.partitioned_hgc)


    def perform_routing(self):
        # SABRE
        pass