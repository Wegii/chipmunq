# How does the performance of the proposed implementation scale as circuit complexity increases?

from __future__ import annotations

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))
sys.path.append(os.path.join(os.getcwd(), "glue/eccentric_bench/"))
import time

# Custom utils
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.circuit_generator import QECMemory, QECCircuit, get_tqec_cnot_rotated
from qeccm.backends.backend_utils import plot_circuit_layout, plot_circuit_layout_utilization
from qeccm.src.reference_partitions import memory_d5
from experiments.exp_utils.simulation_utils import *
from qeccm.backends.backend_utils import plot_gate_map

from experiments.exp_utils.circuit_utils import stim_to_qiskit
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit
from glue.qiskit_qec.stim_tools import get_stim_circuits_with_detectors
from stim import Circuit as StimCircuit

from glue.eccentric_bench.noise import get_noise_model


import numpy as np
import matplotlib.pyplot as plt



def plot_combined(custom_time_storage, sabre_time_storage, filename: str = ""):
    # Extract sorted x values
    np_values = sorted(custom_time_storage.keys())

    # Gather all ks values
    ks_values = sorted({ks for d in custom_time_storage.values() for ks in d.keys()})

    plt.figure(figsize=(9, 6))

    for ks in ks_values:
        # Custom
        y_custom = [custom_time_storage[np].get(ks, None) for np in np_values]
        plt.plot(np_values, y_custom, marker='o', linestyle='-', label=f"Custom ks={ks}")

        # SABRE
        y_sabre = [sabre_time_storage[np].get(ks, None) for np in np_values]
        plt.plot(np_values, y_sabre, marker='x', linestyle='--', label=f"SABRE ks={ks}")

    plt.xlabel("Number of Patches")
    plt.ylabel("Transpilation Time")
    plt.title("Custom vs SABRE Transpilation Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

    

def run_transpilation():

    # Backend configuration
    num_inter_chiplet_connections = 8
    ps_inter = 1e-4

    custom_time_storage = {}
    sabre_time_storage = {}

    n_patches = [1, 10, 50]
    for ks in [2]:#[1, 2, 3, 4]
        for np in n_patches:#, 100, 1000, 10000]:

            if np not in custom_time_storage:
                custom_time_storage[np] = {}
                sabre_time_storage[np] = {}

            # TODO: Calculate necessary backend given number of patches

            # Generate circuit
            circuit, partitions = get_tqec_cnot_rotated(distance_scale = ks,
                                                        n1 = np,
                                                        n2 = 0)

            backend = BackendChipletV2(size = (np*2, np*2, 15, 8),
                            n_inter = num_inter_chiplet_connections,
                            connectivity = "nn",
                            topology = "rotated_grid",
                            inter_chiplet_noise = ps_inter,
                            inter_chiplet_amplification = 1,
                            inter_chiplet_noise_type = "constant",
                            num_defective_qubits=0,
                        )

            # Stim to qiskit
            stim_code_circuit = StimCodeCircuit(stim_circuit = circuit)

            # Custom transpilation
            print("Custom")
            start_custom = time.time()
            custom_circuit = custom_cost_transpilation(stim_code_circuit.qc,
                                                       backend,
                                                       pre_defined_partitions=partitions)
            end_custom = time.time()


            # Sabre transpilation
            print("Sabre")
            start_sabre = time.time()
            sabre_circuit = sabre_transpilation(stim_code_circuit.qc, backend)
            end_sabre = time.time()


            t_dur_custom = end_custom - start_custom
            t_dur_sabre = end_sabre - start_sabre

            custom_time_storage[np][ks] = t_dur_custom
            sabre_time_storage[np][ks] = t_dur_sabre

            # TODO: Store 

    print("Custom:")
    print(custom_time_storage)
    print("Sabre")
    print(sabre_time_storage)

    plot_combined(custom_time_storage,
                  sabre_time_storage,
                  "experiments/evaluation/cnot_scaling.png")

            

if __name__ == "__main__":
    run_transpilation()