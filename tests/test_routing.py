import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))

# Circuits
from experiments.exp_utils.circuit_generator import QECMemory, GenericCircuit
# Backend
from qeccm.backends.BackendChipletV2 import BackendChipletV2
from qeccm.backends.backend_utils import plot_circuit_layout_utilization, plot_circuit_layout
# Qiskit Transpiler
from qiskit.transpiler import StagedPassManager
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


def test_generic_circuit_routing():
    """Test routing of hypergraph"""

    num_qubits = 10*10 - 10
    circuit_generator = GenericCircuit(num_qubits)
    # Note: This is not a stim_circuit !
    circuit = circuit_generator.generate_circuit(8*8)

    # Initialize backend to map to
    chiplet_backend = BackendChipletV2((10, 10, 10, 10), 5)
    
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
    #plot_circuit_layout(routed_circuit, chiplet_backend, "data/backends/mapping/routed_circuit_on_backend.png")
    #plot_circuit_layout_utilization(routed_circuit, chiplet_backend, "data/backends/mapping/routed_circuit_on_backend_utilization.png")
    

if __name__ == "__main__":
    test_surface_memory_circuit_to_hypergraph_partitioning_mapping_routing()
    
    #test_generic_circuit_routing()