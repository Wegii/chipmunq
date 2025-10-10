# Partitioning
import mtkahypar as kahypar


# Not all qubits have the same connectivity
# Only certain qubits are directly connected to the other chiplets
# Each partitioning this needs to have enough qubits that can be connected to other partitions

class GenericHypergraphPartitioning():
    

    def __init__(self):
        pass

    def calculate_partitions(self, circuit, backend) -> float:
        # TODO: function that calculates (given a backend and circuit) into how many cuts it is necessary to partition the circuit

        pass


class KaHyParPartitioning(GenericHypergraphPartitioning):
    """ Hypergraph partitioning based on multilevel hypergraph partitioning framework KaHyPar

    See implementation details in `<https://kahypar.org/>`_ `<https://github.com/kahypar/kahypar>`_
    """

    def __init__(self):
        super().__init__()

    pass