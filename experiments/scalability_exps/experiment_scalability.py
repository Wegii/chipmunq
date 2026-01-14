# How does the performance of the proposed implementation scale as circuit complexity increases?

from __future__ import annotations

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))
sys.path.append(os.path.join(os.getcwd(), "glue/eccentric_bench/"))
import time

# Custom utils
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.circuit_generator import get_tqec_cnot_rotated
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
from math import *
import numpy as np


def plot_combined(custom_time_storage, sabre_time_storage, filename: str = ""):
    # Extract sorted x values
    np_values = sorted(custom_time_storage.keys())

    # Gather all ks values
    ks_values = sorted({ks for d in custom_time_storage.values() for ks in d.keys()})

    fig, ax = plt.subplots(figsize=(8, 4))

    colors_custom = [ "#8FB7E1", "#5E97CC", "#3B6FA8"]
    colors_sabre = [ "#E38E8A", "#C85E59", "#9F3B36"]
    style = []

    for i, ks in enumerate(ks_values):
        # SABRE
        y_sabre = [sabre_time_storage[np].get(ks, None) for np in np_values]
        plt.plot(np_values, y_sabre, marker='x', linestyle='--', label=f"SABRE d{2*ks+1}", color=colors_sabre[i])

    for i, ks in enumerate(ks_values):
        # Custom
        y_custom = [custom_time_storage[np].get(ks, None) for np in np_values]
        plt.plot(np_values, y_custom, marker='o', linestyle='-', label=f"Chipmunq d{2*ks+1}", color=colors_custom[i])

    #plt.axvline(x=3)

    ax.text(
        0.81, 1.02, "Lower is better ↓",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color="#5c79bd",
    )

    description = ("routing_method = basic")
    ax.text(
        0, 1.02, description,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold"
    )

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tick_params(axis='both', labelsize=14)

    plt.xlabel("Number of Patches", fontsize=16)
    plt.ylabel("Runtime [s]", fontsize=16)
    plt.yscale("log")

    #plt.grid(True)
    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', ncol=2)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)


def calculate_speedup(custom_time_storage, sabre_time_storage, filename: str = ""):
    # Extract sorted x values
    np_values = sorted(custom_time_storage.keys())

    # Gather all ks values
    ks_values = sorted({ks for d in custom_time_storage.values() for ks in d.keys()})

    fig, ax = plt.subplots(figsize=(8, 4))

    colors_custom = [ "#8FB7E1", "#5E97CC", "#3B6FA8"]
    colors_sabre = [ "#E38E8A", "#C85E59", "#9F3B36"]
    style = []

    for i, ks in enumerate(ks_values):
        speedup = [sabre_time_storage[np].get(ks, None) / custom_time_storage[np].get(ks, None) for np in np_values]
        
        print(np.mean(speedup))
        #print(np.std(speedup))



def run_exp_scalability():

    # Backend configuration
    ps_inter = 1e-4

    custom_time_storage = {}
    sabre_time_storage = {}

    n_patches = [1, 2, 4, 6, 8]#range(1, 8, 2)#10, 2) 
    """
    for ks in [1, 2, 3]:#, 2, 3]:
        for num_p in n_patches:

            if num_p not in custom_time_storage:
                custom_time_storage[num_p] = {}
                sabre_time_storage[num_p] = {}

            # TODO: Calculate necessary backend given number of patches

            # Generate circuit
            print("Generating circuit")
            circuit, partitions = get_tqec_cnot_rotated(distance_scale = ks,
                                                        n1 = num_p,
                                                        n2 = 0)
            
            # Calculate required number of chiplets given the number of patches that we want to place
            backend_size_chiplets = 2*(num_p+1)
            if ks == 1:
                chiplet_size = (backend_size_chiplets, backend_size_chiplets, 11, 6)
                nic = 5
            elif ks == 2:
                chiplet_size = (backend_size_chiplets, backend_size_chiplets, 15, 8)
                nic = 7
            elif ks == 3:
                chiplet_size = (backend_size_chiplets, backend_size_chiplets, 19, 10)
                nic = 9
            elif ks == 4:
                chiplet_size = (num_p, num_p, 23, 12)
                nic = 11
            print("Generating backend")
            backend = BackendChipletV2(size = chiplet_size,
                            n_inter = nic,
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
            custom_circuit = custom_partitioned_transpilation(stim_code_circuit.qc,
                                                       backend,
                                                       pre_defined_partitions=partitions)
            end_custom = time.time()
            print("Custom done")

            # Sabre transpilation
            print("Sabre")
            start_sabre = time.time()
            sabre_circuit = sabre_transpilation(stim_code_circuit.qc, backend)
            end_sabre = time.time()
            print("SABRE done")


            t_dur_custom = end_custom - start_custom
            t_dur_sabre = end_sabre - start_sabre

            custom_time_storage[num_p][ks] = t_dur_custom
            sabre_time_storage[num_p][ks] = t_dur_sabre

    # Write results to file
    with open(f"experiments/evaluation/scalability/timing_custom.pkl", "wb") as f:
        pickle.dump(custom_time_storage, f)

    with open(f"experiments/evaluation/scalability/timing_sabre.pkl", "wb") as f:
        pickle.dump(sabre_time_storage, f)
    
    """
    # Load pre-computed results
    with open(f"experiments/evaluation/scalability/timing_custom.pkl", "rb") as f:
        custom_time_storage = pickle.load(f)
    with open(f"experiments/evaluation/scalability/timing_sabre.pkl", "rb") as f:
        sabre_time_storage = pickle.load(f)

    print("Custom:")
    print(custom_time_storage)
    print("Sabre")
    print(sabre_time_storage)

    #plot_combined(custom_time_storage,
    #              sabre_time_storage,
    #              "experiments/evaluation/scalability/cnot_scaling.png")

    calculate_speedup(custom_time_storage,
                      sabre_time_storage,
                      "experiments/evaluation/scalability/cnot_scaling_speedup.png")

if __name__ == "__main__":
    run_exp_scalability()