# Circuit verification
from qeccm.circuit.circuit_verification import check_valid_1q_2q_gates
from qeccm.circuit.circuit_statistics import *
# Hypergraph
import rustworkx as rx
from rustworkx.visualization import graphviz_draw
import kahypar
# Visualization
import matplotlib.pyplot as plt
import hypernetx as hnx
# Qiskit Transpiler
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import AnalysisPass
# Hashmap
from collections import defaultdict


class PartitionedHyperGraph:
    """Partitioned Hypergraph after partitioning a HyperGraph object

    Visualization of hypergraph given kahypar partitioning using hypernetx. Translates partitioning format to dict
    structure to then construct a hypergraph 

    Tasks:
        - TODO: Potentially also use the hypergraph from hypernetx as hypergraph object, instead of using the rustworkx
                pygraph. 
    """

    def __init__(self, partitioned_hgc: kahypar.Hypergraph):
        """Generate hypernetx hypergraph given a partitioned kahypar hypergraph

        :param partitioned_hgc: Hypergraph after partitioning
        :type partitioned_hgc: kahypar.Hypergraph
        """

        # Generate dictionary for each block as key containing all nodes
        num_blocks = partitioned_hgc.numBlocks()
        block_to_nodes = {"b:" + str(b): [] for b in range(num_blocks)}
        
        for node in range(partitioned_hgc.numNodes()):
            block_to_nodes["b:" + str(partitioned_hgc.blockID(node))].append(node)

        # Construct hypergraph from partitioned hypergraph
        self._phg = hnx.Hypergraph(block_to_nodes)
        self._btn = block_to_nodes

    def draw_phg(self, filename: str = "") -> None:
        """Draw partitioned hypergraph

        :param filename: Path to write figure to, defaults to ""
        :type filename: str, optional
        """

        # TODO: Add some options (visualization) for plotting the graph more nicely
        hnx.draw(self._phg)

        if filename != "":
            plt.savefig(fname=filename)
        plt.close()
        

class HyperGraph:
    """Hypergraph build upon rustworx graph."""             

    def __init__(self, multigraph=False):
        self._hg = rx.PyGraph(multigraph=multigraph)
        # Mapping of qubit id to graph id
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
        """Hypergraph initializer """

        super().__init__()

    def run(self, dag: DAGCircuit) -> None:
        # Circuit to hypergraph
        self._qc_to_hypergraph(dag)

        # TODO: Translate hypergraph to Kahypar. For now this is left out, since the mapper calls this functions.
        #       Potentially call this directly here, such that the mapper only needs to access the property the
        self.hg_to_kahypar()


    def _qc_to_hypergraph(self, dag: DAGCircuit) -> None:
        """ Create hypergraph given circuit as DAG.

        It is possible to e.g. group gates together (these will then become hyperedges). For now, do not consider such
        grouping mechanism.
        """
        # Use multigraph option for e. g. statistics. For partitioning, duplicate edges can be removed
        hgc = HyperGraph(multigraph = True)

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

        self.property_set['hyper_dag'] = hgc

        # TODO: modify the drawing
        self.draw_hg("data/circuits/surface_memory_hg.png")

    def hg_to_kahypar(self):
        """ Translate hypergraph to kahypar format

        The hypergraph is converted into  a format that is similar to the CSR (Compressed Sparse Row) format. The 
        edge_vector list defines all vertices of a hyperedge. The idx_vector marks where each hyperedge starts in the
        edge_vector list

        Reference:
        - https://github.com/kahypar/kahypar/blob/master/python/module.cpp
        
        :return: _description_
        :rtype: _type_
        """
        
        hgc = self.property_set['hyper_dag']

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

            #print(f"{root_node}: {out_edges_target}")

        # Set property for later usage
        self.property_set['hyper_dag_kahypar'] = (idx_vector, edge_vector)

        # TODO: Remove the return statement and only use the property set from above
        return idx_vector, edge_vector

    def multigraph_to_singular(self):
        # Remove all duplicate edges added due to multigraph setting

        hgc = self.property_set['hyper_dag']

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
        hgc = self.property_set['hyper_dag']
        return hgc.get_num_edges()

    def get_num_vertices(self):
        hgc = self.property_set['hyper_dag']
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
                "rank": "same"
            }
            return attr_dict
        
        hgc = self.property_set['hyper_dag']
        graphviz_draw(hgc._hg, filename=filename, node_attr_fn=node_attr_fn)
