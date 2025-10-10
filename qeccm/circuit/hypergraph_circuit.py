from qiskit import QuantumCircuit

# Circuit verification
from circuit.circuit_verification import check_valid_1q_2q_gates
from circuit.circuit_statistics import *

# Hypergraph
import rustworkx as rx
from itertools import count


class HyperGraph:
    """Hypergraph build upon rustworx graph

    Reference:
    [1] Felix Burt, Kuan-Cheng Chen, Kin Leung, "Generalised Circuit Partitioning for Distributed Quantum Computing"
    `arXiv:2408.01424 <https://arxiv.org/abs/2408.01424>` 
    """
    
     # running id for hyper‑edges
    _hid = count()                 

    def __init__(self, multigraph=False):

        self._g = rx.PyGraph(multigraph=multigraph)
        # vertex label -> rustworkx index
        self._idx_of = {}      

    def _index(self, v):
        """Return rustworkx index for vertex label, creating the node if absent."""
        if v not in self._idx_of:
            self._idx_of[v] = self._g.add_node(v)
        return self._idx_of[v]

    def add_hyperedge(self, roots, targets, payload=None):
        """Insert the hyper‑edge Roots ⟶ Targets and return its id."""
        hid   = next(self._hid)
        he_ix = self._g.add_node(("HE", hid, payload))   # tag helps identify later

        for r in roots:
            self._g.add_edge(self._index(r), he_ix, None)
        for t in targets:
            self._g.add_edge(he_ix, self._index(t), None)
        return hid


class HypergraphCircuit():
    """Quantum circuit represented as hypergraph.

    ASD
    
    References:
    [1] Felix Burt, Kuan-Cheng Chen, Kin Leung, "Generalised Circuit Partitioning for Distributed Quantum Computing"
    `arXiv:2408.01424 <https://arxiv.org/abs/2408.01424>` 

    [2] Pablo Andres-Martinez, Tim Forrer, Daniel Mills, Jun-Yi Wu, Luciana Henaut, Kentaro Yamamoto, Mio Murao,
    Ross Duncan, "Distributing circuits over heterogeneous, modular quantum computing network architectures".
    `arXiv:2305.14148 <https://arxiv.org/abs/2305.14148>`
    """
    
    def __init__(self, circuit: QuantumCircuit):

        # Quantum circuit
        self.circuit = circuit

        # Use multigraph option for e. g. statistics. For partitioning, duplicate edges can be removed
        self.hgc = HyperGraph(multigraph = True)

        # Circuit to hypergraph
        self.hgc = self._qc_to_hypergraph()

    def _qc_to_hypergraph(self):

        # Check if circuit consists of 1q and 2q gates only
        if check_valid_1q_2q_gates(self.circuit):
            raise Exception("Circuit contains >2q gates!")

        # It is possible to e.g. group gates together (these will then become hyperedges). For now, do not consider such
        # grouping mechanism.

        # Iterate over qubits
        for qubit_index, qubit in enumerate(self.circuit.qubits):
            # Each qubit can connect to other qubits (two-qubit gates)
            # 1-qubit gates
            # 2-qubit gates
            pass
        
        pass

    def hg_to_kahypar(self):
        # Translate hypergraph into graph format used by kahypar
        # See: https://github.com/CQCL/pytket-dqc/blob/main/src/pytket_dqc/circuits/hypergraph.py#L474
        pass

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
