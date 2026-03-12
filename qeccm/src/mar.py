import abc

# Qiskit transpiler
import qiskit
from qiskit.transpiler import CouplingMap, PassManager, StagedPassManager
from qiskit.transpiler.passes import ApplyLayout, Unroll3qOrMore
from qiskit.transpiler.passes.layout.enlarge_with_ancilla import EnlargeWithAncilla
from qiskit.transpiler.passes.layout.full_ancilla_allocation import FullAncillaAllocation
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin

# Custom passes
from qeccm.backends import BackendChipletV2
from qeccm.circuit.hypergraph_circuit import HypergraphCircuit
from qeccm.src.mapper import TrivialMapper
from qeccm.src.partitioners import KaHyParPartitioning
from qeccm.src.router import BasicSwapRouter, CostRouter, ParallelSwapRouter


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
        pass

    def perform_routing(self):
        pass


class PartitionedMapRoutePlugin(PassManagerStagePlugin):
    # Not all qubits have the same connectivity
    # Only certain qubits are directly connected to the other chiplets
    # Each partitioning this needs to have enough qubits that can be connected to other partitions

    def pass_manager(self, backend: BackendChipletV2, optimization_level: int | None = None) -> StagedPassManager:
        init_pass = self._generate_initial_pass()
        layout_pass = self._generate_layout_pass(backend)
        routing_pass = self._generate_routing_pass(backend)

        staged_pm = StagedPassManager(
            stages=["init", "layout", "routing"], init=init_pass, layout=layout_pass, routing=routing_pass
        )

        return staged_pm

    def _generate_initial_pass(self) -> PassManager:
        # The output of the init stage is an abstract circuit that contains only one- and two-qubit operations.

        # Construct hypergraph from circuit
        hgc_op = HypergraphCircuit()

        # Convert circuit to single- and two-qubit gates only
        conversion_op = Unroll3qOrMore()

        init_pm = PassManager([conversion_op, hgc_op])
        return init_pm

    # def _generate_layout_pass(self, backend: BackendV2, kp: int = None) -> PassManager:
    def _generate_layout_pass(
        self, backend: BackendChipletV2 = None, partitions=None, patch_initialization: str = ""
    ) -> PassManager:
        # Consists of analysis and transformation passes

        # The hypergraph circuit has multiple edges, since multigraph=True
        # Remove these duplicates, since these are not needed in the partitioning
        # In the local mapping these can be again quite interesting

        # KaHyPar partitioning pass
        partition_op = KaHyParPartitioning(backend, partitions)

        # Mapping pass
        mapping_op = TrivialMapper(backend, patch_initialization=patch_initialization)

        # Extend the dag with ancillas and idling qubits
        extension_op = [
            FullAncillaAllocation(backend.coupling_map),
        ]

        # Map application pass, which performs the mapping on the dag
        apply_mapping_op = ApplyLayout()

        # Combine partitioning and mapping into a single pass
        layout_pm = PassManager([partition_op, mapping_op] + extension_op)

        return layout_pm

    def _generate_routing_pass(
        self, backend, routing_type: str = "basic_optimized", alpha: float = 0.0, beta: float = 0.0
    ) -> PassManager:
        """_summary_

        :param backend: _description_
        :type backend: _type_
        :param routing_type: Type of routing method. Select from: basic, basic_parallel, cost, sabre. Default: "basic"
        :type routing_type: str, optional
        :return: _description_
        :rtype: PassManager
        """
        # Consists of transformation passes

        # Note: it is necessary to perform the mapping_op twice. After the first mapping, we are only working on a
        # circuit of (potentially) size smaller than the backend. By calling the extension_op, ancilla qubits are
        # initialized and added to the DAG. These ancilla qubits are added in an additional qubit register. This is a
        # problem, since this is not expected by the routing pass; thus, it fails!
        # In order to *merge* the normal and ancilla qubit register, simply perform the mapping operation again. This
        # generates a single qubit register with the correct mapping and size

        if routing_type == "basic":
            # Qiskit basic swap implementation
            routing_op = qiskit.transpiler.passes.BasicSwap(coupling_map=CouplingMap(backend.coupling_map))
        elif routing_type == "basic_optimized":
            # Optimized swap router
            routing_op = BasicSwapRouter(backend)
        elif routing_type == "basic_parallel":
            # Parallel implementation of the basic routing
            routing_op = ParallelSwapRouter(backend)
        elif routing_type == "cost":
            # Optimized implementation using an additional cost metric
            routing_op = CostRouter(backend, alpha=alpha, beta=beta)
        elif routing_type == "sabre":
            # Qiskit accelerate SABRE implementation
            routing_op = qiskit.transpiler.passes.SabreSwap(
                coupling_map=CouplingMap(backend.coupling_map), heuristic="decay", seed=42
            )

        router_pm = PassManager([EnlargeWithAncilla(), ApplyLayout(), routing_op])

        return router_pm
