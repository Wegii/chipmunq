# Consider a surface code memory patch
# Map one to one of qubits to hardware -> This should then technically no longer need mapping
# Use SABRE to map and route -> How does this now look like?
from __future__ import annotations

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))

from qeccm.backends.BackendChipletV2 import BackendChipletV2
from qeccm.backends.backend_utils import plot_gate_map, plot_circuit_layout
from experiments.exp_utils.circuit_generator import QECMemory, GenericCircuit
from qiskit.transpiler.passes.layout.enlarge_with_ancilla import EnlargeWithAncilla
from qeccm.src.mar import PartitionedMapRoutePlugin

import qiskit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.layout import Layout
from qiskit.circuit import Qubit, QuantumRegister
from qiskit.transpiler import CouplingMap
from qiskit.transpiler import PassManager, StagedPassManager
from qiskit.transpiler.passes import Unroll3qOrMore, ApplyLayout, TrivialLayout
from qiskit.transpiler.passes.layout.full_ancilla_allocation import FullAncillaAllocation
from qiskit.transpiler.passes.layout.enlarge_with_ancilla import EnlargeWithAncilla
from qiskit.transpiler.passes.layout.set_layout import SetLayout

from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit
from glue.qiskit_qec.stim_tools import get_stim_circuits_with_detectors

def compare_manual_mapping():
    # Generate a chiplet backend, but we only care about one chiplet
    small_backend = BackendChipletV2((1, 2, 10, 10), 5)
    plot_gate_map(backend = small_backend, filename = "experiments/data/mapping_routing/backend.png")

    # Generate surface code d=3
    circuit_generator = QECMemory(None)
    circuit = circuit_generator.generate_code_memory('surface', 1, 1)
    stim_code_circuit = StimCodeCircuit(stim_circuit = circuit)

    #print(stim_code_circuit.qc)


    # TODO: Compare manual mapping with mapping from SABRE

    # Manual mapping
    # Depending on the backend this mapping needs to be aware of the distance of the code. If the patch directly maps
    # to the chiplet, then this is the same as the generate_trivial_layout from qiskit
    circuit_dag = circuit_to_dag(stim_code_circuit.qc)
    layout = Layout()
    regs = circuit_dag.qubits + list(circuit_dag.qregs.values())

    for reg in regs:
        if isinstance(reg, QuantumRegister):
            layout.add_register(reg)
        else:
            layout.add(reg, reg._index)

    # TODO: basic swap
    routing_op = qiskit.transpiler.passes.BasicSwap(coupling_map=CouplingMap(small_backend.coupling_map))
    routing_pm = PassManager([SetLayout(layout),
                              FullAncillaAllocation(small_backend.coupling_map),
                              EnlargeWithAncilla(),
                              ApplyLayout(),
                              routing_op])
    circ_routed = routing_pm.run(stim_code_circuit.qc)


    # TODO: sabre
    routing_op = qiskit.transpiler.passes.SabreSwap(
        coupling_map=CouplingMap(small_backend.coupling_map),
        heuristic='decay',
        seed=42
        )
    routing_pm_sabre = PassManager([TrivialLayout(small_backend.coupling_map), SetLayout(layout),
                              FullAncillaAllocation(small_backend.coupling_map),
                              EnlargeWithAncilla(),
                              ApplyLayout(),
                              routing_op])
    circ_routed_sabre = routing_pm_sabre.run(stim_code_circuit.qc)


    # TODO: compare sabre with manual
    print("Before routing")
    default_circuit = stim_code_circuit.qc
    print(default_circuit.depth())
    print(sum(count for gate, count in default_circuit.count_ops().items() if gate in ["cx", "cz", "swap"]))

    print("BasicSwapRouting of qiskit")
    print(circ_routed.depth())
    print(sum(count for gate, count in circ_routed.count_ops().items() if gate in ["cx", "cz", "swap"]))

    print("SABRE")
    print(circ_routed_sabre.depth())
    print(sum(count for gate, count in circ_routed_sabre.count_ops().items() if gate in ["cx", "cz", "swap"]))

    


def experiment_partitioning():
    # Do the same as above, but now with two surface code patches

    num_qubits = 20
    circuit_generator = GenericCircuit(num_qubits)
    # Note: This is not a stim_circuit !
    circuit = circuit_generator.generate_circuit(2)

    # Initialize backend to map to
    chiplet_backend = BackendChipletV2((2, 2, 5, 5), 4)
    plot_gate_map(backend = chiplet_backend, filename = "experiments/data/mapping_routing/backend.png")

    # NOTE: Map with custom implementation
    mar_pmsp = PartitionedMapRoutePlugin()
    # Pass to construct hypergraph from circuit
    init_pm = mar_pmsp._generate_initial_pass()
    # Pass to perform partition and mapping
    partitioning_pm = mar_pmsp._generate_layout_pass(chiplet_backend)
    # Construct pass manager with all passes
    staged_pm = StagedPassManager(stages=["init", "layout", "routing"], init=init_pm, layout=partitioning_pm)
    mapped_circuit = staged_pm.run(circuit)

    plot_circuit_layout(mapped_circuit,
                        chiplet_backend,
                        filename="experiments/data/mapping_routing/circuit_mapped_to_backend_custom.png")


    routing_op = qiskit.transpiler.passes.SabreSwap(
        coupling_map=CouplingMap(chiplet_backend.coupling_map),
        heuristic='decay',
        seed=42
        )
    routing_pm_sabre = PassManager([TrivialLayout(chiplet_backend.coupling_map),
                              FullAncillaAllocation(chiplet_backend.coupling_map),
                              EnlargeWithAncilla(),
                              ApplyLayout(),
                              routing_op])
    circ_routed_sabre = routing_pm_sabre.run(circuit)
    plot_circuit_layout(circ_routed_sabre,
                        chiplet_backend,
                        filename="experiments/data/mapping_routing/circuit_mapped_to_backend_sabre.png")


if __name__ == "__main__":
    # compare_manual_mapping()

    experiment_partitioning()