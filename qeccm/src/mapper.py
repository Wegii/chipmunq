import abc

from qiskit.providers import BackendV2
from qiskit.dagcircuit import DAGCircuit
from qeccm.circuit.hypergraph_circuit import PartitionedHyperGraph
from qiskit.transpiler.basepasses import TransformationPass


class GenericMapper(TransformationPass):

    def __init__(self):
        """ GenericMapper initializer """ 
        super().__init__()

        # Backend for mapping
        #self.backend = None
        # Dictionary with mapping of nodes
        #self.mapping = {}

    def perform_mapping(self):
        raise NotImplementedError
    
    def visualize_mapping(self, filename) -> None:
        """Visualize mapping on backend"""

        #self.backend.visualize_mapping(self.mapping, filename)        
        self.coupling_map.visualize_mapping(self.property_set["block_node_map"], filename)
    

class RandomMapper(GenericMapper):
    """Random mapping to the backend

    The local chiplet mapping is similar to qiskit.transpiler.passes.TrivialLayout. In the future this could be replaced
    by a completely random mapping scheme.

    Idea:
    - Randomly map necessary qubits to the backend (currently one-to-one mapping)
    - Since most algorithms are heuristics, this can be used as some sort of baseline for comparison
    """

    def __init__(self, coupling_map):
        """ RandomMapper initializer """        

        super().__init__()

        # Coupling map to map the dag to
        self.coupling_map = coupling_map

    def _get_node_in_backend(self, block_idx: int) -> list:
        #return self.backend.get_chiplet_at(block_idx)
        return self.coupling_map.get_chiplet_at(block_idx)

    #def run(self, backend: BackendV2, partitioned_hgc: PartitionedHyperGraph) -> dict:
    #def run(self, dag: DAGCircuit, partitioned_hgc: PartitionedHyperGraph) -> dict:
    def run(self, dag: DAGCircuit) -> None:

        # Construct dictionary with block as key and value as (random) mapping from node to backend node 
        block_node_map = {}

        partitioned_hgc = self.property_set["partitioned_hyper_dag"]

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

        # TODO: technically this should not be done, since we are in a transformation pass
        self.property_set["block_node_map"] = block_node_map

        # TODO: draw mapping
        self.visualize_mapping(filename = "data/backends/surface_memory_mapped_backend.png")

        # Perform mapping on dag
        # TODO: iterate over dag
        # TODO: each qubit gets it's mapped value given the block_node_map

        #for i in dag:
        #    pass
        # TODO: perform mapping on DAG



        return dag


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