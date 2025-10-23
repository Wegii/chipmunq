# Compare the different available routing and mapping methods by qiskit
# - SABRE
# - AIRouting from: https://github.com/Qiskit/qiskit-ibm-transpiler?tab=readme-ov-file
# - Custom implementation

# TODO: Have a look at new transpiler pass in qiskit: LitinskiTransformation
# TODO: Nice figure for showing runtime: https://www.ibm.com/quantum/blog/qiskit-2-2-release-summary

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))
# Typing
from ast import Tuple
from qiskit import QuantumCircuit
from qeccm.backends.BackendChipletV2 import BackendChipletV2
# Circuits
from experiments.utils.circuit_generator import QECMemory, GenericCircuit
#from eccentric_bench.external.qiskit_qec.src.qiskit_qec.circuits.stim_code_circuit import StimCodeCircuit
# Transpiler
from qiskit.transpiler import StagedPassManager
from qiskit import transpile
from qeccm.src.mar import PartitionedMapRoutePlugin
# Visualization
from qeccm.backends.backend_utils import plot_circuit_layout_utilization, plot_circuit_layout


def custom_partitioned_transpilation(circuit: QuantumCircuit, backend: BackendChipletV2):
    mar_pmsp = PartitionedMapRoutePlugin()
    # Pass to construct hypergraph from circuit
    init_pm = mar_pmsp._generate_initial_pass()
    # Pass to perform partition and mapping
    partitioning_pm = mar_pmsp._generate_layout_pass(backend)
    # Pass to perform routing
    routing_pm = mar_pmsp._generate_routing_pass(backend)
    # Construct pass manager with all passes
    staged_pm = StagedPassManager(stages=["init", "layout", "routing"], init=init_pm, layout=partitioning_pm,
                                  routing=routing_pm)
    # Run passes
    routed_circuit = staged_pm.run(circuit)

    return routed_circuit


def sabre_transpilation(circuit: QuantumCircuit, backend: BackendChipletV2):
    return transpile(circuit,
                     #basis_gates=qiskit_stim_gates,
                     basis_gates=['u1', 'u2', 'u3', 'cx', 'swap', 'h'],
                     optimization_level=0,
                     backend=backend,
                     layout_method="sabre",
                     routing_method="sabre"
        )


def _get_backend(type, c1, c2, n, m, n_inter) -> BackendChipletV2:
    # TODO: combine this with backend_generator.py file

    # TODO: Add option to have different types
    print(c1)
    print(c2)
    print(n)
    print(m)
    backend = BackendChipletV2((c1, c2, n, m), n_inter)
    return backend


def _get_circuit(type, num_patches):
    # Get circuit with non-interacting patches

    if type == "surface":
        distance = 9
        # calculate num_qubits for distance
        num_qubits = 100

        circuit_generator = QECMemory(num_qubits)
        # Selects the maximum code distance given the number of qubits
        # Note: This is a stim circuit!
        circuit = circuit_generator.generate_code_memory('surface', num_patches)
    else:
        num_qubits = 10*10 - 10
        circuit_generator = GenericCircuit(num_qubits)
        # Note: This is not a stim_circuit !
        circuit = circuit_generator.generate_circuit(num_patches)

    return circuit


if __name__  == "__main__":
    # Generate backends of different sizes
    n_inter = 1
    c1, c2 = 8, 8#5, 5
    test_backend = _get_backend(None, c1, c2, 10, 10, n_inter)
    small_backend = _get_backend(None, 2, 2, 10, 10, n_inter)
    medium_backend = _get_backend(None, 4, 4, 10, 10, n_inter)
    big_backend = _get_backend(None, 10, 10, 10, 10, n_inter)

    #stim_circuit = _get_circuit("surface", 2*2)
    generic_patch_circuit = _get_circuit("", c1*c2)

    print(generic_patch_circuit.num_qubits)

    import time
    # Transpilation
    start_c = time.time()
    custom_transpiled_circuit = custom_partitioned_transpilation(generic_patch_circuit, test_backend)
    end_c = time.time()


    start = time.time()
    sabre_transpiled_circuit = sabre_transpilation(generic_patch_circuit, test_backend)
    end = time.time()

    print(f"Custom: Elapsed time: {end_c - start_c:.4f} seconds")
    print(f"SABRE: Elapsed time: {end - start:.4f} seconds")
    print(f"Before: {generic_patch_circuit.count_ops()}")
    print(f"Custom: {custom_transpiled_circuit.count_ops()}")
    print(f"SABRE: {sabre_transpiled_circuit.count_ops()}")

    # Visualization
    #plot_circuit_layout(custom_transpiled_circuit, small_backend,
    #                    "experiments/data/figures/transpiled_circuits/custom_partition.png")
    # Transpilation
    #custom_transpiled_circuit = custom_partitioned_transpilation(stim_circuit.qc, small_backend)
    #sabre_transpiled_circuit = sabre_transpilation(stim_circuit.qc, small_backend)
