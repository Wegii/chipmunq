from __future__ import annotations

# Qiskit transpiler
from qiskit.transpiler.basepasses import AnalysisPass

# Hypergrap
import kahypar as kahypar
from qeccm.circuit.hypergraph_circuit import HyperGraph, HypergraphCircuit, PartitionedHyperGraph
import networkx as nx
from networkx.algorithms import community

from qiskit.dagcircuit import DAGCircuit
from qeccm.backends.BackendChipletV2 import BackendChipletV2
from typing import List, Tuple
import numpy as np


class GenericHypergraphPartitioning(AnalysisPass):
    def __init__(self):
        super().__init__()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        # TODO: function that calculates (given a backend and circuit) into how many cuts it is necessary to partition the circuit
        pass


class KaHyParPartitioning(GenericHypergraphPartitioning):
    """ Hypergraph partitioning based on multilevel hypergraph partitioning framework KaHyPar

    See implementation details in `<https://kahypar.org/>`_ `<https://github.com/kahypar/mt-kahypar>`_
    """


    def __init__(self, backend: BackendChipletV2):
        """KaHyPar partitioning initializer"""
        super().__init__()

        self.backend = backend

        # Method for calculating the number of partitions
        #self._calculate_partitions_method = "full"
        self._calculate_partitions_method = "patch-aware"

        # Initialize KaHyPar
        self.khp_context = kahypar.Context()

        # TODO: change objective
        self.khp_context.loadINIconfiguration("qeccm/src/kahypar_config.ini")

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Partition a DAGCircuit into a optimal (calculated) number of partitions

        :param dag: _description_
        :type dag: DAGCircuit
        :return: _description_
        :rtype: _type_
        """

        # Get hypergraph representation of dag
        hgc = self.property_set['hyper_dag']
        
        # Calculate number of partitions
        self.kp, partition_sizes = self.calculate_number_partitions(dag, hgc)


        
        if self.kp > 1:
            # Partition graph into calculated number of partitions
            kahypar_hg = self.perform_partitioning(partition_sizes)

            # Calculate mapping of partition to QPU. This is necessary, since KaHyPar assumes an all-to-all chiplet 
            # topology. Depending on the backend chiplet_topology, we do not have and all-to-all connection.
            partition_to_qpu = self.partition_to_qpu_mapping(kahypar_hg)

            self.property_set["partitioned_hyper_dag"] = PartitionedHyperGraph(kahypar_hg)
            self.property_set["partition_to_qpu"] = partition_to_qpu
        else:
            # Explicit Partitioning not needed 
            # TODO: 
            num_vertices = None
            partition_to_qpu = None
            self.property_set["partitioned_hyper_dag"] = PartitionedHyperGraph(num_nodes = num_vertices)
            self.property_set["partition_to_qpu"] = partition_to_qpu






        # TODO: Do some visualization, so see if for lattice surgery, it is possible to lay out the partitions without any
        #       edges intersecting each other. If there are intersecting edges, this is a big problem for the routing, 
        #       since these connections need to be routed through a whole other qpu.
        #       Goal: We do not want any intersection of edges of the partitioned graph

        return dag

    def perform_partitioning(self, partition_sizes: List[int]) -> kahypar.Hypergraph:
        """_summary_

        :param partition_sizes: _description_
        :type partition_sizes: List[int]
        :return: _description_
        :rtype: kahypar.Hypergraph
        """

        hgc = self.property_set['hyper_dag']

        # Translate vertices and edges from general hypergraph to KaHyPar specific format
        (index_vector, edge_vector) = self.property_set['hyper_dag_kahypar']
        num_vertices = hgc.get_num_vertices() 
        num_hyperedges = len(index_vector) - 1

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
        print("starting partitioning")
        kahypar.partition(kahypar_hg, self.khp_context)
        print("partitioning found")

        return kahypar_hg

    def partition_to_qpu_mapping(self, kahypar_hg: kahypar.Hypergraph) -> dict:
        """Assign each partition to a QPU, based on the interactions with the other partitions.
        
        https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html
        TODO: On which QPU does a partition need to be placed? The QPUs do not have connections to all other QPUs, so
              this can easily become a huge bottleneck!

        :param kahypar_hg: _description_
        :type kahypar_hg: kahypar.Hypergraph
        :return: _description_
        :rtype: List[int]
        """

        # get backend
        backend = self.backend

        # Get number of chips
        num_chips = backend.get_num_chips

        # Number of blocks after partitioning
        num_blocks = kahypar_hg.numBlocks()

        block_to_chip = {}
        # Simple one-to-one mapping of partition to chiplet
        for i, b in enumerate(range(num_blocks)):
            block_to_chip[b] = i

        # TODO: Implement improved version of this mapping!
    
        return block_to_chip

    def calculate_number_partitions(self, dag: DAGCircuit, hgc: HyperGraph) -> Tuple[int, List[int]]:
        """Calculate number of partitions

        This is based on:
            - Backend size of each QPU (are all QPUs the same size, and are all qubits working?)
            - For lattice surgery, detect the patch size and how many can be placed on the backend

        Different methods:
            - full: Fill single chips as much as possible (for lattice surgery potentially not ideal)
            - patch-aware: Keep patches together and distribute


        References:
            - https://networkx.org/documentation/stable/reference/algorithms/community.html
        
        :param dag: _description_
        :type dag: DAGCircuit
        :return: _description_
        :rtype: Tuple[int, List[int]]
        """

        num_qubits_chiplet = self.backend.get_chip_size()
        num_qubits_circuit = dag.num_qubits()

        print("Calculate optimal k")
        # Perform partitioning, if circuit does not fit on on chiplet
        if num_qubits_chiplet < num_qubits_circuit:
            if self._calculate_partitions_method == "full":
                # Fill chiplet as much as possible
                k = int(np.ceil(num_qubits_circuit / num_qubits_chiplet))
                
            elif self._calculate_partitions_method == "patch-aware":

                # Approach to keep patches together:
                #   - Try to find patches in the circuit
                #   - How many patches can be place on a single chiplet? Calculate
                #   - Distribute the patches to all chiplets:
                #       - Simply distributed patches if more chiplets than patches
                #       - If more patches than chiplets, try to have as many as possible good patches, and some bad ones.
                #         TODO: Find out a better way how to handle this

                # Rustworkx to NetworkX
                G_rx = hgc._hg
                G_nx = nx.Graph()

                # Add nodes
                for node_index, node_data in enumerate(G_rx.nodes()):
                    G_nx.add_node(node_index, data=node_data)

                # Add edges
                for u, v, _ in G_rx.weighted_edge_list():
                    G_nx.add_edge(u, v)

                # Iteratively compute Kernighan–Lin bipartition graphs. In each iterations, the graph or already
                # partitioned sub-graph is split into two subgraph while minimizing edge-cut.
                # Stop if all patches can be mapped to the chiplets. The total number of communities is then used as
                # parameter k for the graph partitioning
                # The idea here is to find communities, which should be similar to surface code patches
                # TODO: This corresponds to multilevel partitioning / hierarchical clustering
                bipartite_community_detection = []
                bipartite_community_detection.append(G_nx)
                bipartite_communities = []
                #print(G_nx)
                while True:
                    G_iter = bipartite_community_detection.pop(0)

                    g1_nodes, g2_nodes = nx.algorithms.community.kernighan_lin_bisection(G_iter, max_iter=10)
                    g1 = G_nx.subgraph(g1_nodes).copy()
                    g2 = G_nx.subgraph(g2_nodes).copy()

                    if len(g1) > num_qubits_chiplet:
                        bipartite_community_detection.append(g1)
                    else:
                        bipartite_communities.append(g1_nodes)
                    if len(g2) > num_qubits_chiplet:
                        bipartite_community_detection.append(g2)
                    else:
                        bipartite_communities.append(g2_nodes)

                    if bipartite_community_detection == []:
                        break

                #print(bipartite_communities)
                k = len(bipartite_communities)

                # k can be of maximum size backend_num_chiplets
                if k > self.backend.get_num_chips():
                    k = self.backend.get_num_chips()

                #core_numbers = nx.core_number(G_nx)
                #print(core_numbers)
                #max_core = max(core_numbers.values())
                #print(core_numbers.values())
                #print(max_core)
                #for k in range(1, max_core + 1):
                #    subg = nx.k_core(G_nx, k)
                #    print(f"{k}-core has {len(subg.nodes())} nodes")
                    
                # Community detection
                #communities = community.louvain_communities(G_nx, resolution=0.2, seed=42)
                #print("Detected communities:")
                #for i, c in enumerate(communities):
                #    print(f"  Community {i}: {c}")


                # TODO: Visualzation, if we found the clusters
            else:
                pass
        else:
            k = 1

        print("Optimal k found")
        # Set size of each partition as number of qubits on a chiplet
        partition_sizes = [num_qubits_chiplet for c in range(k)]

        return k, partition_sizes

    def cost_analysis(self):
        """Calculate statistics of partitioned graph
        """

        # Output metrics
        pass