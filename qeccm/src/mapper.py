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
        # TODO: this needs to be replaced by plot_circuit_layout from backend_utils     
        #self.backend.visualize_mapping(self.property_set["block_node_map"], filename)
    

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

        # Big Issue: However, partitioning assumes full qubit connectivity inside and across the quantum processors to
        #            reduce the problem to a graph partitioning problem. on a higher level, this constrained is already
        #            known to a high level compiler (e. g. for lattice surgery). Thus, we should generally not get a 
        #            circuit that has to communicate with another node, to which no direct connection is.
        #            Note: This is not entirely true, since the ancilla qubits used in lattice surgery could become an
        #                  issue, if it is not possible to map these also to the same node!

        # TODO: On which QPU does a partition need to be placed? The QPUs do not have connections to all other QPUs, so
        #       this can easily become a huge bottleneck!
        # TODO: partition_to_qpu_mapping() -> Assign each partition a QPU, based on the interactions with the other
        #                                     partitions.

        # TODO: Parallelization over distributed blocks

        # Get blocks from partitioned hypergraph
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

        layout = Layout()
        regs = dag.qubits + list(dag.qregs.values())

        hgc = self.property_set['hyper_dag']
        for reg in regs:
            if isinstance(reg, QuantumRegister):
                layout.add_register(reg)
            else:
                # Map qubit id to graph id (since the partitioning works on the graph ids)
                qubit_to_node = hgc.node_idx.get(reg._index)

                # Add qubit mapping from partitioning
                if qubit_to_node is not None:
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
         other modules (in general on the border of the chiplet)
    - For the 1. idea, have a look at qiskit.transpiler.passes.DenseLayout, since this could be similar

    Issues and Problems:
    - 
    """

    def __init__(self):
        super().__init__()


    def perform_mapping(self):

        pass


class SABREMapper(GenericMapper):
    def __init__(self):
        super().__init__()

    def perform_mapping(self):
        # Iterate over all partitions
        # Perform SABRE (simply remove the remote gates) to get an initial layout

        # The routing for the local chips should be more or less optimal now. The routing pass will then have to do the
        # actual local routing (SABRE again) and the remote routing as BASIC SWAP
        pass