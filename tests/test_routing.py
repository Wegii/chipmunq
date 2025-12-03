import sys
import os

from qiskit import QuantumCircuit
sys.path.append(os.path.join(os.getcwd(), "."))

# Circuits
from experiments.exp_utils.circuit_generator import QECMemory, GenericCircuit
# Backend
from qeccm.backends.BackendChipletV2 import BackendChipletV2
from qeccm.backends.backend_utils import plot_circuit_layout_utilization, plot_circuit_layout, plot_gate_map
# Qiskit Transpiler
from qiskit.transpiler import StagedPassManager
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.visualization import circuit_drawer
# Transpiler passes
import qiskit
from qiskit.transpiler import PassManager, StagedPassManager, CouplingMap
from qiskit.transpiler.passes import Unroll3qOrMore, ApplyLayout, TrivialLayout
from qiskit.transpiler.passes.layout.full_ancilla_allocation import FullAncillaAllocation
from qiskit.transpiler.passes.layout.enlarge_with_ancilla import EnlargeWithAncilla
# Custom transpiler plugin
from qeccm.src.mar import PartitionedMapRoutePlugin


def test_routing():
    """Test routing step for remote gates.
    
    Tasks:
        - TODO: Are the remote gates from the backend changed? Are are there only SWAP gates around it added
                Probably SWAP gate around it, and then in the translation stage changed
        - TODO: SWAP gates or realization using CNOT?
                Probably also in translation stage
    """

    # Construct simple circuit
    pass


def test_accelerated_routing() -> None:
    """Test custom routing pass written as accelerated implementation in Rust
    """

    # Construct simple backend
    chiplet_backend = BackendChipletV2((2, 2, 2, 2), 2)
    plot_gate_map(backend = chiplet_backend, filename = "tests/figures/basic_backend.png")

    # Construct test circuit
    circ = QuantumCircuit(6)
    # circ.h(0)
    circ.cz(0, 1)
    circ.cz(0, 2)
    circ.cz(0, 3)
    circ.cz(0, 4)
    circ.cz(0, 5)
    

    pre_routing_pm = PassManager([TrivialLayout(chiplet_backend.coupling_map),
                                  FullAncillaAllocation(chiplet_backend.coupling_map),
                                  EnlargeWithAncilla(),
                                  ApplyLayout()])

    circ_mod = pre_routing_pm.run(circ)
    print(circuit_drawer(circ_mod, output="text"))

    # Convert to dag, since transpiler pases receive this format
    circ_dag = circuit_to_dag(circ_mod)
    # Call rust implementation to perform routing using custom implementation
    transpiled_dag = basic_routing(circ_dag, chiplet_backend.target)
    # Draw circuit
    print(circuit_drawer(dag_to_circuit(transpiled_dag), output="text"))

    routing_op = qiskit.transpiler.passes.BasicSwap(coupling_map=CouplingMap(chiplet_backend.coupling_map))
    routing_pm = PassManager([routing_op])
    circ_routed = routing_pm.run(circ_mod)
    print(circuit_drawer(circ_routed, output="text"))


def test_generic_circuit_routing():
    """Test routing of hypergraph"""

    num_qubits = 5*5 - 5
    circuit_generator = GenericCircuit(num_qubits)
    # Note: This is not a stim_circuit !
    circuit = circuit_generator.generate_circuit(2*2)

    # Initialize backend to map to
    chiplet_backend = BackendChipletV2((2, 2, 5, 5), 5)
    
    mar_pmsp = PartitionedMapRoutePlugin()
    # Construct hypergraph from circui[t
    init_pm = mar_pmsp._generate_initial_pass()
    # Perform partition and mapping
    partitioning_pm = mar_pmsp._generate_layout_pass(chiplet_backend)
    # Perform routing
    routing_pm = mar_pmsp._generate_routing_pass(chiplet_backend)

    staged_pm = StagedPassManager(stages=["init", "layout", "routing"], init=init_pm, layout=partitioning_pm,
                                  routing=routing_pm)
    import time
    start_c = time.time()
    routed_circuit = staged_pm.run(circuit)
    end_c = time.time()
    print(end_c - start_c)
    #print(routed_circuit)

    print(circuit.count_ops())
    print(routed_circuit.count_ops())
    print(routed_circuit.depth())

    # 16x16x100
    # Total: 11.467
    # No routing: 5.21

    # 18x18x100
    # Total: 
    # No routing: 

    #plot_circuit_layout(routed_circuit, chiplet_backend, "data/backends/mapping/routed_circuit_on_backend.png")
    #plot_circuit_layout_utilization(routed_circuit, chiplet_backend, "data/backends/mapping/routed_circuit_on_backend_utilization.png")


def test_surface_memory_circuit_to_hypergraph_partitioning_mapping_routing():
    """Test routing of hypergraph"""

    # Minimum number of qubits for distance 3 surface code
    num_qubits = 26 

    # Generate surface code memory circuit
    circuit_generator = QECMemory(num_qubits)
    surface_memory_circuit = (circuit_generator.generate_code_memory('surface')).qc

    # Initialize backend to map to
    chiplet_backend = BackendChipletV2((2, 2, 3, 3), 3)
    
    mar_pmsp = PartitionedMapRoutePlugin()
    # Construct hypergraph from circui[t
    init_pm = mar_pmsp._generate_initial_pass()
    # Perform partition and mapping
    partitioning_pm = mar_pmsp._generate_layout_pass(chiplet_backend)
    # Perform routing
    routing_pm = mar_pmsp._generate_routing_pass(chiplet_backend)

    staged_pm = StagedPassManager(stages=["init", "layout", "routing"], init=init_pm, layout=partitioning_pm,
                                  routing=routing_pm)
    routed_circuit = staged_pm.run(surface_memory_circuit)
    #print(routed_circuit)

    print(surface_memory_circuit.count_ops())
    print(routed_circuit.count_ops())
    plot_circuit_layout(routed_circuit, chiplet_backend, "data/backends/mapping/routed_circuit_on_backend.png")
    #plot_circuit_layout_utilization(routed_circuit, chiplet_backend, "data/backends/mapping/routed_circuit_on_backend_utilization.png")
    

def test_global_routing():
    """Test routing of hypergraph"""

    num_qubits = 5*5 - 5
    circuit_generator = GenericCircuit(num_qubits)
    # Note: This is not a stim_circuit !
    circuit = circuit_generator.generate_circuit(2*2)

    # Initialize backend to map to
    chiplet_backend = BackendChipletV2((2, 2, 15, 8), 8, "nn", "rotated_grid")
    
    mar_pmsp = PartitionedMapRoutePlugin()
    # Construct hypergraph from circuit
    init_pm = mar_pmsp._generate_initial_pass()
    # Perform partition and mapping
    partitioning_pm = mar_pmsp._generate_layout_pass(chiplet_backend)
    # Perform routing
    routing_pm = mar_pmsp._generate_routing_pass(chiplet_backend,
                                                 routing_type="cost")

    staged_pm = StagedPassManager(stages=["init", "layout", "routing"], init=init_pm, layout=partitioning_pm,
                                  routing=routing_pm)
    
    routed_circuit = staged_pm.run(circuit)

    plot_circuit_layout_utilization(routed_circuit,
                                    chiplet_backend,
                                    filename="tests/data/figures/cost_routing_layout_utilization.png")



if __name__ == "__main__":
    #test_surface_memory_circuit_to_hypergraph_partitioning_mapping_routing()
    
    #test_generic_circuit_routing()

    # test_accelerated_routing()

    test_global_routing()