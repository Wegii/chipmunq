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
from experiments.utils import *


def plot_combined(custom_time_storage, sabre_time_storage, filename: str = ""):
    # Extract sorted x values
    np_values = sorted(custom_time_storage.keys())
    x_val = [3*x for x in np_values]

    # Gather all ks values
    ks_values = sorted({ks for d in custom_time_storage.values() for ks in d.keys()})

    
    tex_fonts = {
        # Use LaTeX to write all text
        # "text.usetex": True,
        "font.family": "serif",
        # Font sizes
        "axes.labelsize": FONTSIZE*1.5,
        "font.size": FONTSIZE*1.2,
        "legend.fontsize": (FONTSIZE - 2)*1.5,
        "xtick.labelsize": (FONTSIZE - 1)*1.5,
        "ytick.labelsize": (FONTSIZE - 1)*1.5,
        "axes.titlesize": 10,
        # Line and marker styles
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "lines.markeredgewidth": 1.5,
        "lines.markeredgecolor": "black",
        # Error bar cap size
        "errorbar.capsize": 3,
    }

    plt.rcParams.update(tex_fonts)

    #fig, ax = plt.subplots(figsize=(WIDTH_FIGSIZE*1.2, HEIGHT_FIGSIZE*1.7))
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.5, WIDTH_FIGSIZE*0.5))
    #fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.6, WIDTH_FIGSIZE))

    colors_custom = [ "#8FB7E1", "#5E97CC", "#3B6FA8"]
    colors_sabre = [ "#E38E8A", "#C85E59", "#9F3B36"]

    inter_markers = ['x', 'o', 's']

    handles = []
    for i, ks in enumerate(ks_values):
        # SABRE
        y_sabre = [sabre_time_storage[np].get(ks, None) for np in np_values]
        h = plt.plot(x_val, y_sabre, marker=inter_markers[i], linestyle='--', label=f"LightSABRE, d={2*ks+1}", color=colors_sabre[i])
        # TODO: Calculate correct variance
       #y_std = np.array([2, 2, 2, 2, 2])
       # plt.fill_between(x_val, 
       #          y_sabre - y_std, 
       #          y_sabre + y_std, 
       #          color="#E38E8A", 
       #          alpha=0.3, 
       #          edgecolor="none")
        handles.extend(h)

    #for i, ks in enumerate(ks_values):
        # Custom
        y_custom = [custom_time_storage[np].get(ks, None) for np in np_values]
        h = plt.plot(x_val, y_custom, marker=inter_markers[i], linestyle='-', label=f"Chipmunq, d={2*ks+1}", color=colors_custom[i])
        handles.extend(h)

    #plt.axvline(x=3)


    ax.text(
        -0.01, 1.025, "a) Number of patches affecting runtime",
        transform=ax.transAxes,
        fontweight="bold"
    )

    ax.text(
        0.3, 1.15, "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    #description = ("routing_method = basic")
    #ax.text(
    #    0, 1.02, description,
    #    transform=ax.transAxes,
    #    fontweight="bold"
    #)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tick_params(axis='both', labelsize=14)

    plt.xlabel("Number of patches", fontsize=FONTSIZE*1.5)
    plt.ylabel("Runtime [s]", fontsize=FONTSIZE*1.5)
    plt.yscale("log")

    #plt.grid(True)
    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    #ax.legend(loc='lower right', ncol=2)
    #plt.tight_layout()
    fig.subplots_adjust(left=0.15, right=0.95, top=0.83, bottom=0.21)
    plt.savefig(filename,
                format="pdf")
    plt.close(fig)

    legend_fig = plt.figure(figsize=(3, 2))
    legend = legend_fig.legend(handles = handles,
                               loc = 'center',
                               frameon = False,
                               ncols = 3,
                               columnspacing=1.5)
    legend_fig.savefig(filename + 'legend.pdf', bbox_inches='tight', format="pdf")
    plt.close(legend_fig)



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

            # Generate circuit
            print("Generating circuit")
            circuit, partitions = get_tqec_cnot_rotated(distance_scale = ks,
                                                        n1 = num_p,
                                                        n2 = 0)
            
            # Calculate required number of chiplets given the number of patches that we want to place
            bx = num_p
            by = num_p
            #if num_p <= 4: bx, by = 4, 4
            #if num_p == 6: bx, by = 6, 4
            #if num_p == 8: bx, by = 8, 4
            
            bx, by = 2*(num_p+1), 2*(num_p+1)

            if ks == 1:
                chiplet_size = (bx, by, 11, 6)
                nic = 5
            elif ks == 2:
                chiplet_size = (bx, by, 15, 8)
                nic = 7
            elif ks == 3:
                chiplet_size = (bx, by, 19, 10)
                nic = 9
                
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

    plot_combined(custom_time_storage,
                  sabre_time_storage,
                  "experiments/evaluation/scalability/cnot_scaling.pdf")

    calculate_speedup(custom_time_storage,
                      sabre_time_storage,
                      "experiments/evaluation/scalability/cnot_scaling_speedup.pdf")


# Backend configuration
def run_single_run():
    ps_inter = 1e-4

    num_p = 8 #[1, 2, 4, 6, 8]#range(1, 8, 2)#10, 2) 
    ks = 3

    # ks=2, num_p=8,
    # Original: 17.47
    # No routing: 

    print("Generating circuit")
    circuit, partitions = get_tqec_cnot_rotated(distance_scale = ks,
                                                n1 = num_p,
                                                n2 = 0)
    
    # Calculate required number of chiplets given the number of patches that we want to place
    bx = num_p
    by = num_p
    if num_p <= 4: bx, by = 4, 4
    if num_p == 6: bx, by = 6, 4
    if num_p == 8: bx, by = 8, 4

    bx, by = 2*(num_p+1), 2*(num_p+1)

    if ks == 1:
        chiplet_size = (bx, by, 11, 6)
        nic = 5
    elif ks == 2:
        chiplet_size = (bx, by, 15, 8)
        nic = 7
    elif ks == 3:
        chiplet_size = (bx, by, 19, 10)
        nic = 9
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
    #sabre_circuit = sabre_transpilation(stim_code_circuit.qc, backend)
    end_sabre = time.time()
    print("SABRE done")


    t_dur_custom = end_custom - start_custom
    t_dur_sabre = end_sabre - start_sabre


    print("Custom:")
    print(t_dur_custom)
    print("Sabre")
    print(t_dur_sabre)


if __name__ == "__main__":
    run_exp_scalability()

    # run_single_run()