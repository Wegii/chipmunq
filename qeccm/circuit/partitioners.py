import logging

# Partitioning
import multiprocessing
import mtkahypar as mtkahypar
import kahypar as kahypar
from qeccm.circuit.hypergraph_circuit import HypergraphCircuit
from qeccm.backends.backend import GenericChipletBackend


class GenericHypergraphPartitioning():
    def __init__(self):
        pass

    def run(self, circuit, backend) -> float:
        # TODO: function that calculates (given a backend and circuit) into how many cuts it is necessary to partition the circuit

        pass


class KaHyParPartitioning(GenericHypergraphPartitioning):
    """ Hypergraph partitioning based on multilevel hypergraph partitioning framework KaHyPar

    See implementation details in `<https://kahypar.org/>`_ `<https://github.com/kahypar/mt-kahypar>`_
    """

    def __init__(self, hgc: HypergraphCircuit):
        super().__init__()

        self.hypergraph = hgc
        self.k = -1
        self.partition_id = None

        # Initialize kahypar
        self.khp_context = kahypar.Context()
        # TODO: change objective
        self.khp_context.loadINIconfiguration("qeccm/circuit/kahypar_config.ini")

    def run(self):
        try:
            assert self.k > 0
        except AssertionError:
            logging.warning(f"Number of partitions not set. Setting to 10")
            # TODO: Fix the default setting to 10
            self.k = 10

        # Usage from 
        # https://github.com/CQCL/pytket-dqc/blob/main/src/pytket_dqc/allocators/hypergraph_partitioning.py#L149
        
        # Size of each partition
        # TODO: Calculate from backend
        partition_sizes = [15, 15]

        num_vertices = self.hypergraph.get_num_vertices() 
        num_hyperedges = self.hypergraph.get_num_edges() - 1
        # TODO: check if correct format
        index_vector, edge_vector = self.hypergraph.hg_to_kahypar()

        # For now, all hyperedges are assumed to have the same weight
        hyperedge_weights = [1 for i in range(0, num_hyperedges)]
        # Qubit vertices are given weight 1
        # Potentially vertices with high connectivity should get higher weight to connect these together
        vertex_weights = [1 for i in range(0, num_vertices)]

        self.khp_context.setK(self.k)
        self.khp_context.setCustomTargetBlockWeights(partition_sizes)
        self.khp_context.suppressOutput(True)
        self.khp_context.setSeed(42)

        # Convert hypergraph to kahypar format
        kahypar_hg = kahypar.Hypergraph(
            num_vertices,
            num_hyperedges,
            index_vector,
            edge_vector,
            self.k,
            hyperedge_weights,
            vertex_weights,
            )

        # Partition hypergraph
        kahypar.partition(kahypar_hg, self.khp_context)

        # Partition id for each vertice
        self.partition_id = [kahypar_hg.blockID(i) for i in range(kahypar_hg.numNodes())]

    def cost_analysis(self):
        """Calculate statistics of partitioned graph
        """

        # Output metrics
        print("Partition Stats:")
        print("Imbalance = " + str(self.partitioned_hgc.imbalance(self.khp_context)))
        print("km1       = " + str(self.partitioned_hgc.km1()))
        print("cut       = " + str(self.partitioned_hgc.cut()))
        print("Block Weights:")
        for i in self.partitioned_hgc.blocks():
            print("Weight of Block " + str(i) + " = " + str(self.partitioned_hgc.block_weight(i)))

    def calculate_partitions(self, backend: GenericChipletBackend) -> int:

        # self.hypergraph
        # backend
        pass
