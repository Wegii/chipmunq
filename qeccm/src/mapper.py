from qeccm.circuit.hypergraph_circuit import PartitionedHyperGraph

from qiskit.providers import BackendV2
from qiskit.dagcircuit import DAGCircuit
# Qiskit transpiler
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.layout import Layout
from qiskit.circuit import Qubit, QuantumRegister



class GenericMapper(AnalysisPass):

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
        self.backend.visualize_mapping(self.property_set["block_node_map"], filename)
    

class RandomMapper(GenericMapper):
    """Random mapping to the backend

    The local chiplet mapping is similar to qiskit.transpiler.passes.TrivialLayout. In the future this could be replaced
    by a completely random mapping scheme.

    Idea:
    - Randomly map necessary qubits to the backend (currently one-to-one mapping)
    - Since most algorithms are heuristics, this can be used as some sort of baseline for comparison
    """

    def __init__(self, backend):
        """ RandomMapper initializer """        

        super().__init__()

        # Coupling map to map the dag to
        self.coupling_map = backend.coupling_map
        self.backend = backend

    def _get_node_in_backend(self, block_idx: int) -> list:
        #return self.backend.get_chiplet_at(block_idx)
        return self.backend.get_chiplet_at(block_idx)

    def run(self, dag: DAGCircuit) -> None:

        # Construct dictionary with block as key and value as (random) mapping from node to backend node 
        block_node_map = {}
        full_node_map = {}

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
                # Continuous dict
                full_node_map[node] = nodes_in_backend[n]
            # Each block is also hashed
            block_node_map[block_idx] = node_map

        self.property_set["block_node_map"] = block_node_map

        # TODO: draw mapping
        self.visualize_mapping(filename = "data/backends/surface_memory_mapped_backend.png")


        # Generate layout for mapping
        # Generate a list of physical qubits, to which there exists no virtual mapping
        not_mapped_qubits = []
        
        # TODO: fix the range with minimum and maximum of backend
        for nmq in range(0, 100):
            # Check if nmq is mapped with 
            if not (nmq in full_node_map.values()):
                # value is already mapped
                not_mapped_qubits.append(nmq)

        layout = Layout()
        regs = dag.qubits + list(dag.qregs.values())
        hgc = self.property_set['hyper_dag']
        for reg in regs:
            if isinstance(reg, QuantumRegister):
                layout.add_register(reg)
                print(reg)
            else:
                # Map qubit id to graph id (since the partitioning works on the graph ids)
                qubit_to_node = hgc.node_idx.get(reg._index)
                if qubit_to_node is None:
                    # virtual qubit is not used, so simply use first free qubit
                    p_b = not_mapped_qubits.pop(0)

                    #print(f"mapping qubit {reg._index} to {p_b}")
                    layout.add(reg, p_b)
                else:
                    # Virtual qubit is used and mapped. Get the mapping from the mapping list 
                    p_b = full_node_map[qubit_to_node]
                    
                    #print(f"mapping qubit {reg._index} as {qubit_to_node} to {p_b}")
                    layout.add(reg, p_b)

        # Virtual to physical qubit mapping
        self.property_set["layout"] = layout
        

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