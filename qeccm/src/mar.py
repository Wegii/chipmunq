import abc

# Qiskit transpiler
import qiskit
import qiskit.dagcircuit
from qiskit.providers import BackendV2
from qiskit.transpiler import PassManager, StagedPassManager
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.passes import Unroll3qOrMore, ApplyLayout, TrivialLayout
from qiskit.transpiler.passes.layout.full_ancilla_allocation import FullAncillaAllocation
from qiskit.transpiler.passes.layout.enlarge_with_ancilla import EnlargeWithAncilla
# Custom passes
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

    def pass_manager(self, pass_manager_config: PassManagerConfig, optimization_level: int | None = None
                     ) -> StagedPassManager:
        # TODO: generate stagedpassmanager with all stages

        init_pass = self._generate_initial_pass()
        layout_pass = self._generate_layout_pass(pass_manager_config)
        routing_pass = self._generate_routing_pass(pass_manager_config)

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
    def _generate_layout_pass(self, pass_manager_config: PassManagerConfig = None) -> PassManager:
        # Consists of analysis and transformation passes

        # The hypergraph circuit has multiple edges, since multigraph=True
        # Remove these duplicates, since these are not needed in the partitioning
        # In the local mapping these can be again quite interesting

        kp = 2
        #if kp is None:
        #    # Calculate number of partitions based on circuit and backend
        #    self.kahypar_partitioner.k = self.kahypar_partitioner.calculate_partitions()
        #else:
        #    self.kahypar_partitioner.k = kp

        # KaHyPar partitioning pass
        partition_op = KaHyParPartitioning(kp=kp)

        # Mapping pass
        mapping_op = RandomMapper(pass_manager_config)
        #mapping_op = TrivialLayout(pass_manager_config.coupling_map)

        # Extend the dag with ancillas and idling qubits
        extension_op = [FullAncillaAllocation(pass_manager_config.coupling_map), EnlargeWithAncilla()]

        # Map application pass, which performs the mapping on the dag
        apply_mapping_op = ApplyLayout()

        # Combine partitioning and mapping into a single pass
        # Note: it is necessary to perform the mapping_op twice. After the first mapping, we are only working on a
        # circuit of (potentially) size smaller than the backend. By calling the extension_op, ancilla qubits are 
        # initialized and added to the DAG. These ancilla qubits are added in an additional qubit register. This is a
        # problem, since this is not expected by the routing pass; thus, it fails!
        # In order to *merge* the normal and ancilla qubit register, simply perform the mapping operation again. This
        # generates a single qubit register with the correct mapping and size
        layout_pm = PassManager([partition_op, mapping_op ] + extension_op + 
                                [apply_mapping_op] + [mapping_op, apply_mapping_op])

        return layout_pm

    def _generate_routing_pass(self, pass_manager_config):
        # Consists of transformation passes

        routing_op = BasicSwapRouter(pass_manager_config)
        router_pm = PassManager([routing_op])
        
        return router_pm