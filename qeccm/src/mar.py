import abc

# Qiskit transpiler
import qiskit
from qiskit.transpiler import PassManager, StagedPassManager, CouplingMap
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin
from qiskit.transpiler.passes import Unroll3qOrMore, ApplyLayout, TrivialLayout
from qiskit.transpiler.passes.layout.full_ancilla_allocation import FullAncillaAllocation
from qiskit.transpiler.passes.layout.enlarge_with_ancilla import EnlargeWithAncilla
# Custom passes
from qeccm.backends import BackendChipletV2
from qeccm.src.partitioners import KaHyParPartitioning
from qeccm.circuit.hypergraph_circuit import HypergraphCircuit
from qeccm.src.mapper import CongestionMapper, RandomMapper
from qeccm.src.router import BasicSwapRouter


class GenericMapRoute(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def perform_mapping(self):
        raise NotImplementedError

    @abc.abstractmethod
    def perform_routing(self):
        raise NotImplementedError


class BasicMapRoute(GenericMapRoute):
    def __init__(self):
        pass

    def perform_mapping(self, circuit: qiskit.QuantumCircuit) -> None:
        # Random mapping

        pass

    def perform_routing(self):
        # SABRE
        pass


#class PartitionedMapRoute(GenericMapRoute):
class PartitionedMapRoutePlugin(PassManagerStagePlugin):

    # Not all qubits have the same connectivity
    # Only certain qubits are directly connected to the other chiplets
    # Each partitioning this needs to have enough qubits that can be connected to other partitions

    def pass_manager(self, backend: BackendChipletV2, optimization_level: int | None = None
                     ) -> StagedPassManager:
        # TODO: generate stagedpassmanager with all stages

        init_pass = self._generate_initial_pass()
        layout_pass = self._generate_layout_pass(backend)
        routing_pass = self._generate_routing_pass(backend)

        staged_pm = StagedPassManager(stages=["init", "layout", "routing"], 
                                      init=init_pass, layout=layout_pass, routing=routing_pass)

        return staged_pm

    def _generate_initial_pass(self) -> PassManager:
        # The output of the init stage is an abstract circuit that contains only one- and two-qubit operations.

        # Construct hypergraph from circuit
        hgc_op = HypergraphCircuit()

        # Convert circuit to single- and two-qubit gates only
        conversion_op = Unroll3qOrMore()

        init_pm = PassManager([conversion_op, hgc_op])
        return init_pm

    #def _generate_layout_pass(self, backend: BackendV2, kp: int = None) -> PassManager:
    def _generate_layout_pass(self, backend: BackendChipletV2 = None) -> PassManager:
        # Consists of analysis and transformation passes

        # The hypergraph circuit has multiple edges, since multigraph=True
        # Remove these duplicates, since these are not needed in the partitioning
        # In the local mapping these can be again quite interesting
        
        # KaHyPar partitioning pass
        partition_op = KaHyParPartitioning(backend)

        # Mapping pass
        mapping_op = RandomMapper(backend)
        #mapping_op = TrivialLayout(pass_manager_config.coupling_map)

        # Extend the dag with ancillas and idling qubits
        #extension_op = [FullAncillaAllocation(pass_manager_config.coupling_map), EnlargeWithAncilla()]
        extension_op = [FullAncillaAllocation(backend.coupling_map), ]

        # Map application pass, which performs the mapping on the dag
        apply_mapping_op = ApplyLayout()

        # Combine partitioning and mapping into a single pass
        
        #layout_pm = PassManager([partition_op, mapping_op ] + extension_op + 
        #                        [apply_mapping_op] + [mapping_op, apply_mapping_op])
        #layout_pm = PassManager([partition_op, mapping_op ] + extension_op + 
        #                        [apply_mapping_op] )
        
        layout_pm = PassManager([partition_op, mapping_op ] + extension_op)

        return layout_pm

    def _generate_routing_pass(self, backend):
        # Consists of transformation passes

        # Note: it is necessary to perform the mapping_op twice. After the first mapping, we are only working on a
        # circuit of (potentially) size smaller than the backend. By calling the extension_op, ancilla qubits are 
        # initialized and added to the DAG. These ancilla qubits are added in an additional qubit register. This is a
        # problem, since this is not expected by the routing pass; thus, it fails!
        # In order to *merge* the normal and ancilla qubit register, simply perform the mapping operation again. This
        # generates a single qubit register with the correct mapping and size

        # SABRE
        #routing_op = qiskit.transpiler.passes.SabreSwap(
        #    coupling_map=CouplingMap(backend.coupling_map),
        #    heuristic='decay',
        #    seed=42
        #    )
        #routing_op = qiskit.transpiler.passes.BasicSwap(coupling_map=CouplingMap(backend.coupling_map))
        
        # Custom implementation
        routing_op = BasicSwapRouter(backend)

        router_pm = PassManager([EnlargeWithAncilla(), ApplyLayout(), routing_op])#, routing_op])
        
        return router_pm