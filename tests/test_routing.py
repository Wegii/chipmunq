import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))

# Circuits
from experiments.circuit_generator import QECMemory
# Backend
from qeccm.backends.BackendChipletV2 import BackendChipletV2
from qiskit.visualization import plot_gate_map
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


def test_surface_memory_circuit_to_hypergraph_partitioning_mapping_routing():
    """Test routing of hypergraph"""

    # Minimum number of qubits for distance 3 surface code
    num_qubits = 26 

    # Generate surface code memory circuit
    circuit_generator = QECMemory(num_qubits)
    surface_memory_circuit = circuit_generator.generate_code_memory('surface')

    # Initialize backend to map to
    chiplet_backend = BackendChipletV2((2, 5, 5), 1)
    
    target = chiplet_backend.target
    coupling_map_backend = target.build_coupling_map()
    #print(coupling_map_backend)

    plot_gate_map(
        chiplet_backend,
        plot_directed=False,
        filename = "data/backends/new_chiplet.png"
    )
    
    mar_pmsp = PartitionedMapRoutePlugin()
    # Construct hypergraph from circui[t
    init_pm = mar_pmsp._generate_initial_pass()
    # Perform partition and mapping
    partitioning_pm = mar_pmsp._generate_layout_pass(chiplet_backend)
    # Perform routing
    routing_pm = mar_pmsp._generate_routing_pass(chiplet_backend)

    staged_pm = StagedPassManager(stages=["init", "layout", "routing"], init=init_pm, layout=partitioning_pm,
                                  routing=routing_pm)
    a = staged_pm.run(surface_memory_circuit)
    print(a)
    

if __name__ == "__main__":
    test_surface_memory_circuit_to_hypergraph_partitioning_mapping_routing()