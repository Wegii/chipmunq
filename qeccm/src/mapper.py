import abc
import qiskit
from qiskit.providers import BackendV2
from qeccm.circuit.hypergraph_circuit import PartitionedHyperGraph


class GenericMapper(abc.ABC):

    def __init__(self):
        # Backend for mapping
        self.backend = None
        # Dictionary with mapping of nodes
        self.mapping = {}

    @abc.abstractmethod
    def perform_mapping(self):
        raise NotImplementedError
    
    def visualize_mapping(self, filename) -> None:
        """Visualize mapping on backend"""

        self.backend.visualize_mapping(self.mapping, filename)
    

class RandomMapper(GenericMapper):
    """Random mapping to the backend

    Idea:
    - Randomly map necessary qubits to the backend
    - Since most algorithms are heuristics, this can be used as some sort of baseline for comparison
    """

    def __init__(self):
        super().__init__()

    def _get_node_in_backend(self, block_idx: int) -> list:
        return self.backend.get_chiplet_at(block_idx)

    def perform_mapping(self, backend: BackendV2, partitioned_hgc: PartitionedHyperGraph) -> dict:

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
        return block_node_map


class CongestionMapper(GenericMapper):
    """TODO: Short Description
    
    Idea:
    - Map qubits with high connectivities close together -> Reason for congestion part, as we generally have less qubits
        with connections to other modules
    - Qubits that need a lot of communication with other modules, are placed on nodes that have the connection to the
        other modules

    Issues and Problems:
    - 
    """

    def __init__(self):
        super().__init__()


    def perform_mapping(self):
        pass