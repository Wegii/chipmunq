# Partitioning
import kahypar as kahypar


# Not all qubits have the same connectivity
# Only certain qubits are directly connected to the other chiplets
# Each partitioning this needs to have enough qubits that can be connected to other partitions

class GenericHypergraphPartitioning():
    pass


class KaHyParPartitioning(GenericHypergraphPartitioning):
    """ Hypergraph partitioning based on multilevel hypergraph partitioning framework KaHyPar

    See implementation details in `<https://kahypar.org/>`_ `<https://github.com/kahypar/kahypar>`_
    """
    pass