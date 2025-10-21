import logging

# Qiskit transpiler
from qiskit.transpiler.basepasses import AnalysisPass

# Partitioning
import multiprocessing
import mtkahypar as mtkahypar
import kahypar as kahypar
from qeccm.circuit.hypergraph_circuit import HypergraphCircuit, PartitionedHyperGraph
from qeccm.backends.BackendChipletV2 import BackendChipletV2


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


    def __init__(self, backend: BackendChipletV2, kp: int = None):
        """KaHyPar partitioning initializer"""
        super().__init__()

        self.backend = backend
        self.kp = kp

        # Initialize kahypar
        self.khp_context = kahypar.Context()
        # TODO: change objective
        self.khp_context.loadINIconfiguration("qeccm/src/kahypar_config.ini")

    def run(self, dag):
        
        if self.kp == None:
            self.kp = self.calculate_partitions()

        # TODO: Calculate into how many partitions the circuit should be split.
        #       This is based on a number of things:
        #           - Backend size of each QPU (are all QPUs the same size, and are all qubits working?)
        #           - For lattice surgery, detect the patch size and how many can be placed on the backend
        #           


        # Size of each partition (number of qubits it can hold)
        # TODO: this needs to be calculated from the backend!!!
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

        self.khp_context.setK(self.kp)
        self.khp_context.setCustomTargetBlockWeights(partition_sizes)
        self.khp_context.suppressOutput(True)
        self.khp_context.setSeed(42)

        # Convert hypergraph to kahypar format
        kahypar_hg = kahypar.Hypergraph(
            num_vertices,
            num_hyperedges,
            index_vector,
            edge_vector,
            self.kp,
            hyperedge_weights,
            vertex_weights,
            )

        # Partition hypergraph
        kahypar.partition(kahypar_hg, self.khp_context)

        self.property_set["partitioned_hyper_dag"] = PartitionedHyperGraph(kahypar_hg)

        # TODO: drawing
        #self.property_set["partitioned_hyper_dag"].draw_phg(filename="data/circuits/surface_memory_phg.png")


        # TODO: Do some visualization, so see if for lattice surgery, it is possible to lay out the partitions without any
        #       edges intersecting each other. If there are intersecting edges, this is a big problem for the routing, 
        #       since these connections need to be routed through a whole other qpu.
        #       Goal: We do not want any intersection of edges of the partitioned graph

        return dag

    def calculate_partitions(self, backend: BackendChipletV2, dag) -> int:

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