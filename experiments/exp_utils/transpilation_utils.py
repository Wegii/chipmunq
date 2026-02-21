# Qiskit transpiler
import qiskit
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager, StagedPassManager
from qiskit.transpiler.passes import ApplyLayout, TrivialLayout, Unroll3qOrMore
from qiskit.transpiler.passes.layout.enlarge_with_ancilla import EnlargeWithAncilla
from qiskit.transpiler.passes.layout.full_ancilla_allocation import FullAncillaAllocation

from qeccm.backends.BackendChipletV2 import BackendChipletV2

# Custom implementation
from qeccm.src.mar import PartitionedMapRoutePlugin


def custom_partitioned_transpilation(
    circuit: QuantumCircuit, backend: BackendChipletV2, pre_defined_partitions: list = None
) -> qiskit.QuantumCircuit:
    """Transpile circuit to a chiplet backend using custom mapping and routing.

    :param circuit: _description_
    :type circuit: QuantumCircuit
    :param backend: _description_
    :type backend: BackendChipletV2
    :return: _description_
    :rtype: qiskit.QuantumCircuit
    """
    mar_pmsp = PartitionedMapRoutePlugin()
    # Pass to construct hypergraph from circuit
    init_pm = mar_pmsp._generate_initial_pass()
    # Pass to perform partition and mapping
    partitioning_pm = mar_pmsp._generate_layout_pass(backend, partitions=pre_defined_partitions)
    # Pass to perform routing
    routing_pm = mar_pmsp._generate_routing_pass(backend)
    # Construct pass manager with all passes
    staged_pm = StagedPassManager(
        stages=["init", "layout", "routing"], init=init_pm, layout=partitioning_pm, routing=routing_pm
    )
    # staged_pm = StagedPassManager(stages=["init", "layout"], init=init_pm, layout=partitioning_pm)
    # Run passes
    routed_circuit = staged_pm.run(circuit)

    return routed_circuit


def custom_cost_transpilation(
    circuit: QuantumCircuit,
    backend: BackendChipletV2,
    pre_defined_partitions: list = None,
    routing_alpha: float = 0.0,
    routing_beta: float = 0.0,
    patch_initialization: str = "",
) -> QuantumCircuit:

    # Initialize transpilation plugin in order to run the different passes
    mar_pmsp = PartitionedMapRoutePlugin()
    # Pass to construct hypergraph from circuit
    init_pm = mar_pmsp._generate_initial_pass()
    # Pass to perform partition and mapping
    partitioning_pm = mar_pmsp._generate_layout_pass(
        backend, partitions=pre_defined_partitions, patch_initialization=patch_initialization
    )
    # Perform routing utilizing cost routing
    routing_pm = mar_pmsp._generate_routing_pass(backend, routing_type="cost", alpha=routing_alpha, beta=routing_beta)
    # Construct pass manager with all passes
    staged_pm = StagedPassManager(
        stages=["init", "layout", "routing"], init=init_pm, layout=partitioning_pm, routing=routing_pm
    )

    # Run passes
    routed_circuit = staged_pm.run(circuit)

    return routed_circuit


def custom_accelerated_partitioned_transpilation(
    circuit: QuantumCircuit, backend: BackendChipletV2
) -> qiskit.QuantumCircuit:
    """Transpile circuit to a chiplet backend using custom mapping and accelerated routing.

    :param circuit: _description_
    :type circuit: QuantumCircuit
    :param backend: _description_
    :type backend: BackendChipletV2
    :return: _description_
    :rtype: qiskit.QuantumCircuit
    """
    # Custom implementation of basic routing

    from qeccm.src.router import AcceleratedBasicSwapRouter

    mar_pmsp = PartitionedMapRoutePlugin()
    # Pass to construct hypergraph from circuit
    init_pm = mar_pmsp._generate_initial_pass()
    # Pass to perform partition and mapping
    partitioning_pm = mar_pmsp._generate_layout_pass(backend)
    # Pass to perform routing
    # Pre-routing pass
    routing_pm = PassManager([EnlargeWithAncilla(), ApplyLayout(), AcceleratedBasicSwapRouter(backend)])

    # Construct pass manager with all passes
    staged_pm = StagedPassManager(
        stages=["init", "layout", "routing"], init=init_pm, layout=partitioning_pm, routing=routing_pm
    )
    # Run passes
    routed_circuit = staged_pm.run(circuit)

    return routed_circuit


def basicswap_transpilation(circuit: QuantumCircuit, backend: BackendChipletV2) -> qiskit.QuantumCircuit:
    """Transpile circuit to a chiplet backend using basic swap mapping and routing.

    :param circuit: _description_
    :type circuit: QuantumCircuit
    :param backend: _description_
    :type backend: BackendChipletV2
    :return: _description_
    :rtype: qiskit.QuantumCircuit
    """
    init_pm = PassManager([Unroll3qOrMore()])

    layout_pm = PassManager([TrivialLayout(backend.coupling_map), FullAncillaAllocation(backend.coupling_map)])

    routing_op = qiskit.transpiler.passes.BasicSwap(coupling_map=CouplingMap(backend.coupling_map))
    router_pm = PassManager([EnlargeWithAncilla(), ApplyLayout(), routing_op])
    staged_pm = StagedPassManager(
        stages=["init", "layout", "routing"], init=init_pm, layout=layout_pm, routing=router_pm
    )

    return staged_pm.run(circuit)


def sabre_transpilation(circuit: QuantumCircuit, backend: BackendChipletV2) -> qiskit.QuantumCircuit:
    """Transpile circuit to a chiplet backend using SABRE mapping and routing.

    :param circuit: _description_
    :type circuit: QuantumCircuit
    :param backend: _description_
    :type backend: BackendChipletV2
    :return: _description_
    :rtype: qiskit.QuantumCircuit
    """
    init_pm = PassManager([Unroll3qOrMore()])

    # TODO: Change this to SABRE layout
    layout_pm = PassManager([TrivialLayout(backend.coupling_map), FullAncillaAllocation(backend.coupling_map)])

    routing_op = qiskit.transpiler.passes.SabreSwap(
        coupling_map=CouplingMap(backend.coupling_map), heuristic="decay", seed=42
    )
    router_pm = PassManager([EnlargeWithAncilla(), ApplyLayout(), routing_op])
    staged_pm = StagedPassManager(
        stages=["init", "layout", "routing"], init=init_pm, layout=layout_pm, routing=router_pm
    )

    return staged_pm.run(circuit)
