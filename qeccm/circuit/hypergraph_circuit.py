# Circuit verification
from qeccm.circuit.circuit_verification import check_valid_1q_2q_gates
from qeccm.circuit.circuit_statistics import *

# Hypergraph
import rustworkx as rx
from rustworkx.visualization import graphviz_draw

# Qiskit DAG
import qiskit
from qiskit.dagcircuit import DAGCircuit

# Hashmap
from collections import defaultdict


class PartitionedHyperGraph:
    """Partitioned Hypergraph after partitioning a HyperGraph object

    TODO: Current idea is to use this class for visualization of a hypergraph after partitioning. For visualization, 
    it is possible to use a specific visualization library:
    - https://github.com/HGX-Team/hypergraphx/
    - https://github.com/pnnl/HyperNetX

    TODO: A possible idea is to plot all nodes and overlay the nodes corresponding to the same partition with a color
    """

    def __init__(self, partition_id: list):
        # This list contains the partition id for each node
        self.partition_id = partition_id

        pass

    def draw_phg(self):
        # TODO: Draw partitioned hypergraph
        pass
        

class HyperGraph:
    """Hypergraph build upon rustworx graph."""             

    def __init__(self, multigraph=False):
        self._hg = rx.PyGraph(multigraph=multigraph)
        self.node_idx = defaultdict(int)   

    def add_hyperedge(self, root: int, targets: list) -> None:
        if not root in self.node_idx:
            node_idx = self._hg.add_node(str(root))
            self.node_idx[root] = node_idx

        # Add edges to all target nodes
        for t in targets:
            # Add target node if it does not exists yet
            if not t in self.node_idx:
                node_idx = self._hg.add_node(str(t))
                self.node_idx[t] = node_idx

            self._hg.add_edge(self.node_idx[root], self.node_idx[t], None)

    def get_num_edges(self):
        return self._hg.num_edges()

    def get_num_vertices(self):
        return self._hg.num_nodes()


class HypergraphCircuit():
    """Quantum circuit represented as hypergraph.

    TODO: Description
    
    References:
    [1] Felix Burt, Kuan-Cheng Chen, Kin Leung, "Generalised Circuit Partitioning for Distributed Quantum Computing"
    `arXiv:2408.01424 <https://arxiv.org/abs/2408.01424>` 

    [2] Pablo Andres-Martinez, Tim Forrer, Daniel Mills, Jun-Yi Wu, Luciana Henaut, Kentaro Yamamoto, Mio Murao,
    Ross Duncan, "Distributing circuits over heterogeneous, modular quantum computing network architectures".
    `arXiv:2305.14148 <https://arxiv.org/abs/2305.14148>`
    """
    
    def __init__(self, circuit: DAGCircuit):

        # Quantum circuit
        self.circuit = circuit

        # Use multigraph option for e. g. statistics. For partitioning, duplicate edges can be removed
        self.hgc = HyperGraph(multigraph = True)

        # Circuit to hypergraph
        self._qc_to_hypergraph()

    def _qc_to_hypergraph(self) -> None:
        """ Create hypergraph given circuit as DAG.

        It is possible to e.g. group gates together (these will then become hyperedges). For now, do not consider such
        grouping mechanism.
        """

        # Check if circuit consists of 1q and 2q gates only
        if check_valid_1q_2q_gates(self.circuit):
            raise Exception("Circuit contains >2q gates!")

        control_map = defaultdict(list)
        # Serial iteration over DAG using serial_layers. It is also possible to group all gates together (theoretical
        # possible to run in parallel) using multigraph_layers.
        for layer in self.circuit.serial_layers():
            subdag = layer["graph"]

            # Iterate over two-qubit gates
            for gate in subdag.two_qubit_ops():
                qc, qt = gate.qargs
                control_map[qc._index].append(qt._index)

        for control, targets in control_map.items():
            self.hgc.add_hyperedge(control, targets)

    def hg_to_kahypar(self):
        """ Translate hypergraph to kahypar format

        The hypergraph is converted into  a format that is similar to the CSR (Compressed Sparse Row) format. The 
        edge_vector list defines all vertices of a hyperedge. The idx_vector marks where each hyperedge starts in the
        edge_vector list

        Reference:
        - https://github.com/kahypar/kahypar/blob/master/python/module.cpp
        - https://github.com/CQCL/pytket-dqc/blob/main/src/pytket_dqc/circuits/hypergraph.py#L474
        
        :return: _description_
        :rtype: _type_
        """
        
        # Construct edge_vector and index_vector
        
        edge_vector = []
        idx_vector = []
        pos = 0
        # Iterate over all vertices
        for vertice in self.hgc.node_idx:
            root_node = self.hgc.node_idx[vertice]

            # Get all edges going out from this vertice
            out_edges = self.hgc._hg.out_edges(root_node)
            out_edges_target = [n[1] for n in out_edges]
            # Need to be in ascending order
            out_edges_target.sort()

            idx_vector.append(pos)
            edge_vector.extend([root_node] + out_edges_target)
            pos += len(out_edges_target) + 1

            #print(f"{root_node}: {out_edges_target}")

        return idx_vector, edge_vector

    def multigraph_to_singular(self):
        # Remove all duplicate edges added due to multigraph setting

        simple_g = rx.PyGraph(multigraph=False)

        # Copy nodes
        for node in self.hgc.node_indices():
            simple_g.add_node(self.hgc[node])

        # Copy edges (only one per unique pair)
        added_pairs = set()
        for u, v, data in self.hgc.edge_list():
            pair = tuple(sorted((u, v)))
            if pair not in added_pairs:
                simple_g.add_edge(u, v, data)
                added_pairs.add(pair)

        return simple_g

    def cost_analysis(self):
        # calculate gates (swap, two-qubits, etc.)
        
        # calc depth
        
        # calc num qubits

        calculate_gates()

    def get_num_edges(self):
        return self.hgc.get_num_edges()

    def get_num_vertices(self):
        return self.hgc.get_num_vertices()

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
                "rank": "same"
            }
            return attr_dict
        
        graphviz_draw(self.hgc._hg, filename=filename, node_attr_fn=node_attr_fn)
