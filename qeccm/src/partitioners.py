import logging

# Qiskit transpiler
from qiskit.transpiler.basepasses import AnalysisPass

# Partitioning
import multiprocessing
import mtkahypar as mtkahypar
import kahypar as kahypar
from qeccm.circuit.hypergraph_circuit import HypergraphCircuit, PartitionedHyperGraph
from qeccm.backends.backend import GenericChipletBackend


class GenericHypergraphPartitioning(AnalysisPass):
    def __init__(self):
        super().__init__()

    def run(self, circuit, backend) -> float:
        # TODO: function that calculates (given a backend and circuit) into how many cuts it is necessary to partition the circuit

        pass


class KaHyParPartitioning(GenericHypergraphPartitioning):
    """ Hypergraph partitioning based on multilevel hypergraph partitioning framework KaHyPar

    See implementation details in `<https://kahypar.org/>`_ `<https://github.com/kahypar/mt-kahypar>`_
    """


    def __init__(self, kp: int):
        """KaHyPar partitioning initializer"""
        super().__init__()

        
        #self.hypergraph = hgc
        self.k = kp
        self.partition_id = None

        # Initialize kahypar
        self.khp_context = kahypar.Context()
        # TODO: change objective
        self.khp_context.loadINIconfiguration("qeccm/src/kahypar_config.ini")

    def run(self, dag):
        try:
            assert self.k > 0
        except AssertionError:
            logging.warning(f"Number of partitions not set. Calculating optimal parameter.")

            # TODO: Calculate from coupling_map
            self.k = self.calculate_partitions()

        # Size of each partition
        # TODO: this needs to be calculated from the backend
        partition_sizes = [15, 15]

        hgc = self.property_set['hyper_dag']
        #print(hgc)
        #print(self.property_set)

        (index_vector, edge_vector) = self.property_set['hyper_dag_kahypar'] #self.hypergraph.hg_to_kahypar()
        num_vertices = hgc.get_num_vertices() 
        num_hyperedges = len(index_vector)-1

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

        self.property_set["partitioned_hyper_dag"] = PartitionedHyperGraph(kahypar_hg)

        # TODO: drawing
        self.property_set["partitioned_hyper_dag"].draw_phg(filename="data/circuits/surface_memory_phg.png")

        return dag

    def calculate_partitions(self, backend: GenericChipletBackend) -> int:

        # TODO: this needs to be imlpemented
        # Calculate the optimal partitions given circuit size and available backend (number of e. g. chiplets, to which
        # the circuit is partitioned and distributed)

        # self.hypergraph
        # backend
        pass

    def cost_analysis(self):
        """Calculate statistics of partitioned graph
        """

        # Output metrics
        pass