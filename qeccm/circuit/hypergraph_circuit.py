# Circuit verification
# Hashmap
from collections import defaultdict

import hypernetx as hnx
import kahypar

# Visualization
import matplotlib.pyplot as plt
import networkx as nx

# from qecc_mapping.experiments.utils.circuit_statistics import *
# Hypergraph
import rustworkx as rx

# Qiskit Transpiler
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import AnalysisPass
from rustworkx.visualization import graphviz_draw

from qeccm.circuit.circuit_verification import check_valid_1q_2q_gates


class PartitionedHyperGraph:
    """Partitioned Hypergraph after partitioning a HyperGraph object

    Visualization of hypergraph given kahypar partitioning using hypernetx. Translates partitioning format to dict
    structure to then construct a hypergraph

    Tasks:
        - TODO: Potentially also use the hypergraph from hypernetx as hypergraph object, instead of using the rustworkx
                pygraph.
    """

    def __init__(self, partitioned_hgc: kahypar.Hypergraph = None, num_nodes=None, hgc=None, partitions=None, dag=None):

        # """Generate hypernetx hypergraph given a partitioned kahypar hypergraph

        #:param partitioned_hgc: Hypergraph after partitioning. If None, assume that only one block exists
        #:type partitioned_hgc: kahypar.Hypergraph
        # """

        if partitioned_hgc is not None:
            # Generate dictionary for each block as key containing all nodes
            num_blocks = partitioned_hgc.numBlocks()
            block_to_nodes = {"b:" + str(b): [] for b in range(num_blocks)}

            # Assign each block all nodes
            (index_vector, edge_vector) = hgc
            available_nodes = set(edge_vector)
            for node in range(partitioned_hgc.numNodes()):
                # Note: kahypar might assign more nodes to a block, than we actually want to consider. Thus, filter
                # out nodes that do not belong to the hypergraph at all
                if node in available_nodes:
                    block_to_nodes["b:" + str(partitioned_hgc.blockID(node))].append(node)

            # Construct collapsed hypergraph
            # All blocks are collapsed to singular nodes, while edges between blocks are kept
            ch = nx.MultiGraph()
            # Create a node for each block
            for b in range(partitioned_hgc.numBlocks()):
                ch.add_node(b)

            # Iterate through all hyperedges
            num_hyperedges = len(index_vector) - 1
            for h in range(num_hyperedges):
                # Get the nodes inside hyperedge h
                start = index_vector[h]
                end = index_vector[h + 1]
                hyperedge_nodes = edge_vector[start:end]

                # Determine the blocks these nodes belong to
                blocks = set(partitioned_hgc.blockID(v) for v in hyperedge_nodes)

                # If multiple blocks appear in one hyperedge, connect them
                blocks = list(blocks)
                for i in range(len(blocks)):
                    for j in range(i + 1, len(blocks)):
                        ch.add_edge(blocks[i], blocks[j])
        elif partitions != None:
            block_to_nodes = {"b:" + str(b): [] for b in range(len(partitions))}
            # block_to_nodes = {b: [] for b in range(len(partitions))}

            for i, partition in enumerate(partitions):
                block_to_nodes["b:" + str(i)].extend(partition["indices"][:])
                # block_to_nodes[i].extend(partition['indices'][:])

            (index_vector, edge_vector) = hgc
            available_nodes = set(edge_vector)
            # Construct collapsed hypergraph
            # All blocks are collapsed to singular nodes, while edges between blocks are kept
            ch = nx.MultiGraph()
            # Create a node for each partition
            for b in range(len(partitions)):
                ch.add_node(b)

            # Precompute qubit to partition mapping
            qubit_to_partition = {}
            for p_index, part in enumerate(partitions):
                for qubit in part["indices"]:
                    qubit_to_partition[qubit] = p_index

            interactions = {i: set() for i in range(len(partitions))}
            # Iterate over all 2-qubit gates in the DAG
            for node in dag.two_qubit_ops():
                q_indices = [q._index for q in node.qargs]
                q0, q1 = q_indices

                # Find the partitions these qubits belong to
                p0 = qubit_to_partition[q0]
                p1 = qubit_to_partition[q1]

                # Record direct interactions between adjacent partitions
                if p0 != p1:
                    interactions[p0].add(p1)
                    interactions[p1].add(p0)

            interactions = {k: sorted(v) for k, v in interactions.items()}

            for p, others in interactions.items():
                for q in others:
                    ch.add_edge(p, q)

        else:
            # Implementation in case only one partition is available

            # Generate dictionary with one block
            block_to_nodes = {"b:" + str(0): []}

            (index_vector, edge_vector) = hgc
            all_nodes = set(edge_vector)

            # Assign all nodes to this block
            for node in all_nodes:
                block_to_nodes["b:" + str(0)].append(node)

            # Create a node for this block
            ch = nx.MultiGraph()
            ch.add_node(0)

        # Construct hypergraph from partitioned hypergraph
        # print(block_to_nodes)
        self._phg = hnx.Hypergraph(block_to_nodes)

        plt.figure(figsize=(6, 6))
        nx.draw(ch, with_labels=True, node_size=600)
        plt.savefig("data/backends/mapping/hx_contracted_graph_of_circuit.png", dpi=300)
        plt.close()

        self._collapsed_phg = ch

        self._btn = block_to_nodes
        # Save kahypar hypergraph
        self._kahypar_hgc = partitioned_hgc

        # print("Found partitions:")
        # print(block_to_nodes)

    def draw_phg(self, graph: hnx.Hypergraph, filename: str = "") -> None:
        """Draw partitioned hypergraph

        :param filename: Path to write figure to, defaults to ""
        :type filename: str, optional
        """
        # TODO: Add some options (visualization) for plotting the graph more nicely
        hnx.draw(graph)

        if filename != "":
            plt.savefig(fname=filename)
        plt.close()

    def contract_partitioning(self) -> None:
        # Create a contracted hypergraph given the partitions

        # Start with the hypergraph and all nodes

        # Contract all nodes of one group
        pass


class HyperGraph:
    """Hypergraph build upon rustworx graph."""

    def __init__(self, multigraph=False):
        self._hg = rx.PyGraph(multigraph=multigraph)
        # Mapping of qubit id to graph id
        self.node_idx = defaultdict(int)

    def add_hyperedge(self, root: int, targets: list) -> None:
        if root not in self.node_idx:
            node_idx = self._hg.add_node(str(root))
            self.node_idx[root] = node_idx

        # Add edges to all target nodes
        for t in targets:
            # Add target node if it does not exists yet
            if t not in self.node_idx:
                node_idx = self._hg.add_node(str(t))
                self.node_idx[t] = node_idx

            self._hg.add_edge(self.node_idx[root], self.node_idx[t], None)

    def get_num_edges(self):
        return self._hg.num_edges()

    def get_num_vertices(self):
        return self._hg.num_nodes()


class HypergraphCircuit(AnalysisPass):
    """Quantum circuit represented as hypergraph.

    TODO: Description

    References:
    [1] Felix Burt, Kuan-Cheng Chen, Kin Leung, "Generalised Circuit Partitioning for Distributed Quantum Computing"
    `arXiv:2408.01424 <https://arxiv.org/abs/2408.01424>`

    [2] Pablo Andres-Martinez, Tim Forrer, Daniel Mills, Jun-Yi Wu, Luciana Henaut, Kentaro Yamamoto, Mio Murao,
    Ross Duncan, "Distributing circuits over heterogeneous, modular quantum computing network architectures".
    `arXiv:2305.14148 <https://arxiv.org/abs/2305.14148>`
    """

    def __init__(self):
        """Hypergraph initializer"""
        super().__init__()

    def run(self, dag: DAGCircuit) -> None:
        # print("Start circuit to hg transformation")
        # Circuit to hypergraph
        # self._qc_to_hypergraph(dag)
        # Translate hypergraph to Kahypar.
        # self.hg_to_kahypar()

        # Fast conversion of dag to KaHyPar CSR format. Note: hyper_dag property is not available, since no real
        # hypergraph is constructed.
        self._dag_to_kahypar(dag)

        # TODO: possibility to create hypergraph (for visualization only) from KaHyPar CSR format

    def _qc_to_hypergraph(self, dag: DAGCircuit) -> None:
        """Create hypergraph given circuit as DAG.

        It is possible to e.g. group gates together (these will then become hyperedges). For now, do not consider such
        grouping mechanism.
        """
        # Use multigraph option for e. g. statistics. For partitioning, duplicate edges can be removed
        hgc = HyperGraph(multigraph=True)

        # Check if circuit consists of 1q and 2q gates only
        if check_valid_1q_2q_gates(dag):
            raise Exception("Circuit contains >2q gates!")

        control_map = defaultdict(list)
        # Serial iteration over DAG using serial_layers. It is also possible to group all gates together (theoretical
        # possible to run in parallel) using multigraph_layers.
        for layer in dag.serial_layers():
            subdag = layer["graph"]

            # Iterate over two-qubit gates
            for gate in subdag.two_qubit_ops():
                qc, qt = gate.qargs
                control_map[qc._index].append(qt._index)

        for control, targets in control_map.items():
            hgc.add_hyperedge(control, targets)

        self.property_set["hyper_dag"] = hgc

        # TODO: modify the drawing
        self.draw_hg("data/circuits/surface_memory_hg.png")

    def hg_to_kahypar(self):
        """Translate hypergraph to kahypar format

        The hypergraph is converted into  a format that is similar to the CSR (Compressed Sparse Row) format. The
        edge_vector list defines all vertices of a hyperedge. The idx_vector marks where each hyperedge starts in the
        edge_vector list

        Reference:
        - https://github.com/kahypar/kahypar/blob/master/python/module.cpp

        :return: _description_
        :rtype: _type_
        """
        hgc = self.property_set["hyper_dag"]

        # Construct edge_vector and index_vector

        edge_vector = []
        idx_vector = []
        pos = 0
        # Iterate over all vertices
        for vertice in hgc.node_idx:
            root_node = hgc.node_idx[vertice]

            # Get all edges going out from this vertice
            out_edges = hgc._hg.out_edges(root_node)
            out_edges_target = [n[1] for n in out_edges]
            # Need to be in ascending order
            out_edges_target.sort()

            idx_vector.append(pos)
            edge_vector.extend([root_node] + out_edges_target)
            pos += len(out_edges_target) + 1

            # print(f"{root_node}: {out_edges_target}")

        # Set property for later usage
        self.property_set["hyper_dag_kahypar"] = (idx_vector, edge_vector)

        # TODO: Remove the return statement and only use the property set from above
        return idx_vector, edge_vector

    def _dag_to_kahypar(self, dag: DAGCircuit) -> None:
        """Fast conversion from dag to KaHyPar CSR format

        :param dag: _description_
        :type dag: DAGCircuit
        """
        edge_vector = []
        num_qubits = len(dag.qubits)
        # print(num_qubits)
        connectivity = [set() for _ in range(num_qubits)]

        # Build connectivity map from all 2-qubit gates
        for node in dag.two_qubit_ops():
            q_indices = [q._index for q in node.qargs]
            q0, q1 = q_indices
            connectivity[q0].add(q1)
            connectivity[q1].add(q0)

        # Build edge_vector list
        idx_vector = []
        pos = 0
        edge_vector = []
        for i, connected in enumerate(connectivity):
            edge = sorted([i] + list(connected))

            # Do not add a edge if the vertice only interacts with itself
            if len(edge) > 1:
                edge_vector.extend(edge)
                idx_vector.append(pos)
                pos += len(edge)
            # else:
            # print(i)

        self.property_set["hyper_dag_kahypar"] = (idx_vector, edge_vector)

        # print(idx_vector)
        # print(edge_vector)

    def multigraph_to_singular(self):
        # Remove all duplicate edges added due to multigraph setting

        hgc = self.property_set["hyper_dag"]

        simple_g = rx.PyGraph(multigraph=False)

        # Copy nodes
        for node in hgc.node_indices():
            simple_g.add_node(hgc[node])

        # Copy edges (only one per unique pair)
        added_pairs = set()
        for u, v, data in hgc.edge_list():
            pair = tuple(sorted((u, v)))
            if pair not in added_pairs:
                simple_g.add_edge(u, v, data)
                added_pairs.add(pair)

        return simple_g

    def cost_analysis(self):
        # calculate gates (swap, two-qubits, etc.)

        # calc depth

        # calc num qubits

        # calculate_gates()
        pass

    def get_num_edges(self):
        hgc = self.property_set["hyper_dag"]
        return hgc.get_num_edges()

    def get_num_vertices(self):
        hgc = self.property_set["hyper_dag"]
        return hgc.get_num_vertices()

    def draw_hg(self, filename=""):
        """Draw hypergraph

        :param filename: Path to write figure to, defaults to ""
        :type filename: str, optional
        """

        def node_attr_fn(node):
            attr_dict = {
                "fontcolor": "white",
                "color": "darkcyan",
                "fill_color": "darkcyan",
                "style": "filled",
                "shape": "circle",
                "label": str(node),
                "width": ".5",
                "height": ".5",
                "rank": "same",
            }
            return attr_dict

        hgc = self.property_set["hyper_dag"]
        graphviz_draw(hgc._hg, filename=filename, node_attr_fn=node_attr_fn)
