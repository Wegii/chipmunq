from __future__ import annotations

# Hypergraph
import kahypar

# Plotting utils
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Typing
from qiskit.dagcircuit import DAGCircuit

# Qiskit transpiler
from qiskit.transpiler.basepasses import AnalysisPass

from qeccm.backends.BackendChipletV2 import BackendChipletV2
from qeccm.circuit.hypergraph_circuit import HyperGraph, PartitionedHyperGraph


class GenericHypergraphPartitioning(AnalysisPass):
    def __init__(self):
        super().__init__()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        # TODO: function that calculates (given a backend and circuit) into how many cuts it is necessary to
        # partition the circuit
        raise NotImplementedError


class KaHyParPartitioning(GenericHypergraphPartitioning):
    """Hypergraph partitioning based on multilevel hypergraph partitioning framework KaHyPar

    References:
        - https://kahypar.org/
        - https://github.com/kahypar/mt-kahypar
    """

    def __init__(self, backend: BackendChipletV2, partitions=None):
        """KaHyPar partitioning initializer"""
        super().__init__()

        self.backend = backend

        # Method for calculating the number of partitions
        # self._calculate_partitions_method = "full"
        self._calculate_partitions_method = "patch-aware"
        # self._calculate_partitions_method = "patch-splitting-aware"

        # Initialize KaHyPar
        self.khp_context = kahypar.Context()

        # TODO: change objective
        self.khp_context.loadINIconfiguration("qeccm/src/kahypar_config.ini")

        # Option to pass partitions. This way, no partitioning needs to be performed
        self.partitions = partitions

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Partition a DAGCircuit into a optimal (calculated) number of partitions

        :param dag: _description_
        :type dag: DAGCircuit
        :return: _description_
        :rtype: _type_
        """
        # Utilize pre-defined partitions
        if self.partitions != None:
            self.property_set["partitioned_hyper_dag"] = PartitionedHyperGraph(
                partitioned_hgc=None, hgc=self.property_set["hyper_dag_kahypar"], partitions=self.partitions, dag=dag
            )

            # Store the pre-defined partitions for all other passes to access
            self.property_set["pre_defined_partitions"] = self.partitions
        else:
            # Get hypergraph representation of dag
            hgc = self.property_set["hyper_dag"]

            # Calculate number of partitions
            self.kp, partition_sizes = self.calculate_number_partitions(dag, hgc)

            if self.kp > 1:
                # Partition graph into calculated number of partitions
                kahypar_hg = self.perform_partitioning(partition_sizes)

                # Hypergraph
                self.property_set["partitioned_hyper_dag"] = PartitionedHyperGraph(
                    partitioned_hgc=kahypar_hg, hgc=self.property_set["hyper_dag_kahypar"]
                )
            else:
                # Explicit Partitioning not needed
                (index_vector, edge_vector) = self.property_set["hyper_dag_kahypar"]
                num_vertices = len(index_vector)

                self.property_set["partitioned_hyper_dag"] = PartitionedHyperGraph(
                    num_nodes=num_vertices, hgc=self.property_set["hyper_dag_kahypar"]
                )
                # self.property_set["partition_to_qpu"] = partition_to_qpu

            # TODO: Do some visualization, so see if for lattice surgery, it is possible to lay out the partitions without
            #       any edges intersecting each other. If there are intersecting edges, this is a big problem for the
            #       routing, since these connections need to be routed through a whole other qpu.
            #       Goal: We do not want any intersection of edges of the partitioned graph

        return dag

    def perform_partitioning(self, partition_sizes: list[int]) -> kahypar.Hypergraph:
        """_summary_

        :param partition_sizes: _description_
        :type partition_sizes: List[int]
        :return: _description_
        :rtype: kahypar.Hypergraph
        """
        # Get vertices and edges in KaHyPar specific format
        (index_vector, edge_vector) = self.property_set["hyper_dag_kahypar"]

        num_vertices = max(max(edge_vector) + 1, len(set(edge_vector)) - 1)
        num_hyperedges = len(index_vector) - 1

        # For now, all hyperedges are assumed to have the same weight
        hyperedge_weights = [1 for i in range(num_hyperedges)]
        # Qubit vertices are given weight 1
        # Potentially vertices with high connectivity should get higher weight to connect these together
        vertex_weights = [1 for i in range(num_vertices)]

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

        print("Partitioning succeeded")

        return kahypar_hg

    def calculate_number_partitions(self, dag: DAGCircuit, hgc: HyperGraph) -> tuple[int, list[int]]:
        """Calculate number of partitions

        This is based on:
            - Backend size of each QPU (are all QPUs the same size, and are all qubits working?)
            - For lattice surgery, detect the patch size and how many can be placed on the backend

        Different methods:
            - full: Fill single chips as much as possible (for lattice surgery potentially not ideal)
            - patch-aware: Keep patches together and distribute


        References:
            - https://networkx.org/documentation/stable/reference/algorithms/community.html
            - https://link.springer.com/article/10.1007/s11227-025-06918-3

        :param dag: _description_
        :type dag: DAGCircuit
        :return: _description_
        :rtype: Tuple[int, List[int]]
        """
        num_qubits_chiplet = self.backend.get_chip_size()
        num_qubits_circuit = dag.num_qubits()

        print("Calculate optimal k")
        # Perform partitioning, if circuit does not fit on on chiplet
        if self._calculate_partitions_method == "full":
            # Fill chiplet as much as possible
            k = int(np.ceil(num_qubits_circuit / num_qubits_chiplet))

        elif self._calculate_partitions_method == "patch-aware":
            # Try to find all higly connected patches in a circuit

            (index_vector, edge_vector) = self.property_set["hyper_dag_kahypar"]

            # Create multigraph given index and edge vectors
            H = nx.MultiGraph()

            num_hyperedges = len(index_vector) - 1

            # TODO: Incorporate number of paths from one to another node. Patches should have many parallel edges,
            #       which should allow us to extract the patches

            for h in range(num_hyperedges):
                start = index_vector[h]
                end = index_vector[h + 1]

                # Nodes in hyperedge h
                nodes = edge_vector[start:end]

                # Hyperedge node label
                H.add_node(h)

                # Connect hyperedge to its member nodes
                h_node = h
                for u in nodes:
                    if u != h:
                        H.add_node(u)
                        H.add_edge(h_node, u)

            plt.figure(figsize=(6, 6))
            # Draw the graph
            nx.draw(H, with_labels=True, node_size=100)
            plt.savefig("data/backends/mapping/hx_graph_of_circuit.png", dpi=300)
            plt.close()
            print("Printed graph")

            # Girvan–Newman algorithm because this method produces a contractiontree that approximates the optimal
            # solution in terms of spatial cost.
            comp = nx.community.girvan_newman(H)
            communities = tuple(sorted(c) for c in next(comp))

            # communities = nx.community.greedy_modularity_communities(H, cutoff=5)

            print(f"Found {len(communities)} communities")
            print(communities)
            k = len(communities)
            # print()

        elif self._calculate_partitions_method == "patch-splitting-aware":
            # Approach to keep patches together:
            #   - Try to find patches in the circuit
            #   - How many patches can be place on a single chiplet? Calculate
            #   - Distribute the patches to all chiplets:
            #       - Simply distributed patches if more chiplets than patches
            #       - If more patches than chiplets, try to have as many as possible good patches, and some bad ones.
            #         TODO: Find out a better way how to handle this

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
            # print(G_nx)
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

            # print(bipartite_communities)
            k = len(bipartite_communities)

            # k can be of maximum size backend_num_chiplets
            k = min(k, self.backend.get_num_chips())
        else:
            pass

        # k = 1#5#3 # 3

        print(f"Optimal k found: {k}")
        # Set size of each partition as number of qubits on a chiplet
        partition_sizes = [num_qubits_chiplet for c in range(k)]
        # partition_sizes = [58, 58, 58, 5, 5]

        """
        (index_vector, edge_vector) = self.property_set['hyper_dag_kahypar']
        print(edge_vector)
        print(sorted(set(edge_vector)))
        
        num_vertices = len(set(edge_vector))-1#max(max(edge_vector) + 1, len(set(edge_vector))-1)
        print(num_vertices)
        num_hyperedges = len(index_vector) - 1
        print(num_hyperedges)
        """

        return k, partition_sizes
