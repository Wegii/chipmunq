import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))

# Circuits
from experiments.exp_utils.circuit_generator import QECMemory
# Backend
from qeccm.backends.BackendChipletV2 import BackendChipletV2
from qeccm.backends.backend_utils import plot_gate_map, plot_circuit_layout
# Qiskit Transpiler
from qiskit.transpiler import StagedPassManager
# Custom transpiler plugin
from qeccm.src.mar import PartitionedMapRoutePlugin


def test_surface_memory_circuit_to_hypergraph_partitioning_mapping():
    """Test partitioning and mapping of hypergraph"""

    # Minimum number of qubits for distance 3 surface code
    num_qubits = 26 

    # Generate surface code memory circuit
    circuit_generator = QECMemory(num_qubits)
    surface_memory_circuit = (circuit_generator.generate_code_memory('surface')).qc
    print(surface_memory_circuit.count_ops())

    # Initialize backend to map to
    #chiplet_backend = BackendChipletV2((2, 5, 5), 1)
    chiplet_backend = BackendChipletV2((2, 2, 5, 5), 1)
    
    #target = chiplet_backend.target
    #coupling_map_backend = target.build_coupling_map()
    #print(coupling_map_backend)

    plot_gate_map(
        chiplet_backend,
        filename = "data/backends/test_layout_gate_map.png"
    )
    
    mar_pmsp = PartitionedMapRoutePlugin()
    # Construct hypergraph from circuit
    init_pm = mar_pmsp._generate_initial_pass()
    # Perform partition and mapping
    partitioning_pm = mar_pmsp._generate_layout_pass(chiplet_backend)

    staged_pm = StagedPassManager(stages=["init", "layout"], init=init_pm, layout=partitioning_pm)
    mapped_circuit = staged_pm.run(surface_memory_circuit)

    #print(mapped_circuit)

    
    # print mapping
    plot_circuit_layout(mapped_circuit, chiplet_backend, filename="data/backends/mapping/mapped_circuit_on_backend.png")

if __name__ == "__main__":
    test_surface_memory_circuit_to_hypergraph_partitioning_mapping()