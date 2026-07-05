import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))

from experiments.exp_utils.circuit_generator import QECCircuit, QECMemory, generate_gross_code
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.utils import *



def test_gross_compilation():

    circuit, partitions = generate_gross_code(num_qubits = 1)


    backend = BackendChipletV2(
        # 1 Chiplet with 288 qubits
        #size = (1, 1, 24, 12),
        size = (1, 1, 12, 24),                     
        n_inter = 1, 
        # Long range connections                               
        connectivity="torus",
        # Grid layout
        topology="grid",
        long_range_offsets=[(1,0), (2,0), (3,0),
                            (0,1), (0,2), (0,3)],
        num_defective_qubits=0
    )

    stim_code_circuit = StimCodeCircuit(stim_circuit=circuit)

    # Custom transpilation
    print("Starting custom transpilation...")
    custom_circuit = custom_partitioned_transpilation(
        stim_code_circuit.qc, backend, pre_defined_partitions=partitions
    )

    # Sabre transpilation
    print("Starting LigthSABRE transpilation...")
    sabre_circuit = sabre_transpilation(stim_code_circuit.qc, backend)

    def num_2q_gates(circuit):
        ops = circuit.count_ops()
        two_qubit_gate_names = ["cx", "cz", "swap"]
        return sum(ops.get(g, 0) for g in two_qubit_gate_names)

    custom_depth = custom_circuit.depth() - (stim_code_circuit.qc).depth()
    custom_overhead= num_2q_gates(custom_circuit) - num_2q_gates(stim_code_circuit.qc)

    sabre_depth = sabre_circuit.depth() - (stim_code_circuit.qc).depth()
    sabre_overhead = num_2q_gates(sabre_circuit) - num_2q_gates(stim_code_circuit.qc)

    depth_overall = (stim_code_circuit.qc).depth()
    gate_overall = num_2q_gates(stim_code_circuit.qc)


    print(f"Optimal has depth {depth_overall} and num_2q_gates{gate_overall}")
    print(f"SABRE has overhead depth {sabre_depth} and num_2q_gates{sabre_overhead}")
    print(f"Custom has overhead depth {custom_depth} and num_2q_gates{custom_overhead}")
    

if __name__ == "__main__":
    test_gross_compilation()
