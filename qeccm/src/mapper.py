import abc

from qiskit.providers import BackendV2
from qeccm.circuit.hypergraph_circuit import PartitionedHyperGraph
from qiskit.transpiler.basepasses import AnalysisPass


class GenericMapper(AnalysisPass):

    def __init__(self):
        # Backend for mapping
        self.backend = None
        # Dictionary with mapping of nodes
        self.mapping = {}

    def perform_mapping(self):
        raise NotImplementedError
    
    def visualize_mapping(self, filename) -> None:
        """Visualize mapping on backend"""

        self.backend.visualize_mapping(self.mapping, filename)
    

class RandomMapper(GenericMapper):
    """Random mapping to the backend

    The local chiplet mapping is similar to qiskit.transpiler.passes.TrivialLayout. In the future this could be replaced
    by a completely random mapping scheme.

    Idea:
    - Randomly map necessary qubits to the backend (currently one-to-one mapping)
    - Since most algorithms are heuristics, this can be used as some sort of baseline for comparison
    """

    def __init__(self):
        super().__init__()

    def _get_node_in_backend(self, block_idx: int) -> list:
        return self.backend.get_chiplet_at(block_idx)

    def run(self, backend: BackendV2, partitioned_hgc: PartitionedHyperGraph) -> dict:
        # TODO: fix the input and output parameters

        # Set backend in order to track used backend after mapping
        self.backend = backend

        # Construct dictionary with block as key and value as (random) mapping from node to backend node 
        block_node_map = {}

        # TODO: parallelization over distributed blocks
        for block in partitioned_hgc._btn.items():
            # Extract index from key
            block_idx = int(block[0][2:])

            # Iterate over nodes
            node_map = {}
            nodes_in_backend = self._get_node_in_backend(block_idx)
            for n, node in enumerate(block[1]):
                node_map[node] = nodes_in_backend[n]

            block_node_map[block_idx] = node_map

        self.mapping = block_node_map

        # TODO: perform mapping on DAG
        # TODO: this should then be returned DAGCircuit: A mapped DAG.
        # TODO: set the block_node_map as self.property_set[SOME_NAME] = block_node_map

        return block_node_map


class CongestionMapper(GenericMapper):
    """TODO: Short Description
    
    Idea:
    - 1. Map qubits with high connectivities close together -> Reason for congestion part, as we generally have less qubits
         with connections to other modules
    - 2. Qubits that need a lot of communication with other modules, are placed on nodes that have the connection to the
         other modules
    - For the 1. idea, have a look at qiskit.transpiler.passes.DenseLayout, since this could be similar

    Issues and Problems:
    - 
    """

    def __init__(self):
        super().__init__()


    def perform_mapping(self):
        pass