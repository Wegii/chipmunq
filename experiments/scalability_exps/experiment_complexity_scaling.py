from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.getcwd(), "."))

from experiments.exp_utils.circuit_generator import get_tqec_cnot_rotated
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.utils import *
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit

# Plotting
import pickle
import time
from math import *
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import FixedLocator
from pathlib import Path


def runtime_scaling(reproduce: bool = False):
    # Backend configuration
    ps_inter = 1e-4

    custom_time_storage = {}

    # Number of logical CNOTs constructed using lattice surgery
    n_patches = [1, 2, 4]

    # Code distance of surface code
    code_distance = 3

    n_chiplets = [4, 6, 8, 10, 12]#list(range(4, 11))

    output_dir = Path("experiments/evaluation/scalability")
    output_dir.mkdir(parents=True, exist_ok=True)

    if reproduce:
        for nc in n_chiplets:
            if code_distance == 1:
                chiplet_size = (nc, nc, 11, 6)
                nic = 5
            elif code_distance == 2:
                chiplet_size = (nc, nc, 15, 8)
                nic = 7
            elif code_distance == 3:
                chiplet_size = (nc, nc, 19, 10)
                nic = 9
            elif code_distance == 4:
                chiplet_size = (nc, nc, 23, 12)
                nic = 11
            elif code_distance == 5:
                chiplet_size = (nc, nc, 27, 14)
                nic = 13
            elif code_distance == 6:
                chiplet_size = (nc, nc, 31, 16)
                nic = 15
            elif code_distance == 7:
                chiplet_size = (nc, nc, 35, 18)
                nic = 17

            print("Generating backend")
            backend = BackendChipletV2(
                size=chiplet_size,
                n_inter=nic,
                connectivity="nn",
                topology="rotated_grid",
                inter_chiplet_noise=ps_inter,
                inter_chiplet_amplification=1,
                inter_chiplet_noise_type="constant",
                num_defective_qubits=0,
            )
            
            for num_p in n_patches:
                if nc not in custom_time_storage:
                    custom_time_storage[nc] = {}
                    
                # Generate circuit
                print("Generating circuit")
                circuit, partitions = get_tqec_cnot_rotated(distance_scale=code_distance, n1=num_p, n2=0)

                # Stim to qiskit
                stim_code_circuit = StimCodeCircuit(stim_circuit=circuit)

                # Custom transpilation
                print("Custom")
                start_custom = time.time()
                _ = custom_partitioned_transpilation(stim_code_circuit.qc, backend, pre_defined_partitions=partitions)
                end_custom = time.time()
                print("Custom done")

                t_dur_custom = end_custom - start_custom
                
                custom_time_storage[nc][num_p] = t_dur_custom

        with open(output_dir / f"timing_custom_d{code_distance}.pkl", "wb") as f:
                pickle.dump(custom_time_storage, f)

    # Load pre-computed results
    with open(output_dir / f"timing_custom_d{code_distance}.pkl", "rb") as f:
        custom_time_storage = pickle.load(f)

    plot_runtime_scaling(custom_time_storage, filename=f"experiments/evaluation/scalability/compilation_scalability_d{code_distance}.pdf")


def runtime_scaling_logical_qubits(reproduce: bool = False):
    # Backend configuration
    ps_inter = 1e-4

    custom_time_storage = {}

    # Number of logical CNOTs constructed using lattice surgery
    n_patches = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Code distance of surface code
    code_distances = [1, 2, 3]#[1, 2, 3, 4]

    # Number of chiplets
    nc = 12

    output_dir = Path("experiments/evaluation/scalability")
    output_dir.mkdir(parents=True, exist_ok=True)

    if reproduce:
        chiplet_size = (nc, nc, 23, 12)
        nic = 11
        
        print("Generating backend")
        backend = BackendChipletV2(
            size=chiplet_size,
            n_inter=nic,
            connectivity="nn",
            topology="rotated_grid",
            inter_chiplet_noise=ps_inter,
            inter_chiplet_amplification=1,
            inter_chiplet_noise_type="constant",
            num_defective_qubits=0,
        )

        for code_distance in code_distances:
            for num_p in n_patches:
                
                if code_distance not in custom_time_storage:
                    custom_time_storage[code_distance] = {}
                    
                # Generate circuit
                print("Generating circuit")
                circuit, partitions = get_tqec_cnot_rotated(distance_scale=code_distance, n1=num_p, n2=0)

                # Stim to qiskit
                stim_code_circuit = StimCodeCircuit(stim_circuit=circuit)

                # Custom transpilation
                print("Custom")
                start_custom = time.time()
                _ = custom_partitioned_transpilation(stim_code_circuit.qc, backend, pre_defined_partitions=partitions)
                end_custom = time.time()
                print("Custom done")

                t_dur_custom = end_custom - start_custom
                
                custom_time_storage[code_distance][num_p] = t_dur_custom

        with open(output_dir / f"timing_custom_lq_nc{nc}.pkl", "wb") as f:
                pickle.dump(custom_time_storage, f)

    # Load pre-computed results
    with open(output_dir / f"timing_custom_lq_nc{nc}.pkl", "rb") as f:
        custom_time_storage = pickle.load(f)
    print(custom_time_storage)

    # Flip the keys around
    custom_time_storage = {
        num_p: {cd: custom_time_storage[cd][num_p] for cd in custom_time_storage}
        for num_p in n_patches
    }
        
    plot_runtime_scaling(custom_time_storage, lq=True, filename=f"experiments/evaluation/scalability/compilation_scalability_nc{nc}.pdf")


def plot_runtime_scaling(custom_time_storage, lq: bool = False, filename: str = ""):

    np_values = sorted(custom_time_storage.keys())
    x_val = [x*x for x in np_values]

    # Gather all ks values
    ks_values = sorted({ks for d in custom_time_storage.values() for ks in d.keys()})

    tex_fonts = {
        # Use LaTeX to write all text
        # "text.usetex": True,
        "font.family": "serif",
        # Font sizes
        "axes.labelsize": FONTSIZE * 1.5,
        "font.size": FONTSIZE * 1.2,
        "legend.fontsize": (FONTSIZE - 2) * 1.5,
        "xtick.labelsize": (FONTSIZE - 1) * 1.5,
        "ytick.labelsize": (FONTSIZE - 1) * 1.5,
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
    #fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE * 2.5, WIDTH_FIGSIZE * 0.5))
    fig, ax = plt.subplots(figsize=((HEIGHT_FIGSIZE * 2.5)*0.6, (HEIGHT_FIGSIZE * 2.5)*0.6))

    colors_custom = []
    colors_custom = ["#AEC6CF", "#77DD77", "#FFB347",]
    inter_markers = ["o", "s", "^"]
    handles = []
    for i, ks in enumerate(ks_values):
        y_custom = [custom_time_storage[np].get(ks, None) for np in np_values]
        h = plt.plot(
            x_val,
            y_custom,
            marker=inter_markers[i],
            linestyle="-",
            label=f"lq = {ks}" if not lq else f"d = {2 * ks + 1}",
            color=colors_custom[i],
        )
        handles.extend(h)


    if not lq:
        ax.text(0.025, 1.025, "a) Hardware scaling", transform=ax.transAxes, fontweight="bold")
        plt.xlabel("#Chiplets", fontsize=FONTSIZE * 1.5)
        ax.xaxis.set_major_locator(FixedLocator(x_val[::2]))
        plt.ylim(0, 10)

    else:
        ax.text(0.0, 1.025, "b) Circuit complexity", transform=ax.transAxes, fontweight="bold")
        plt.xlabel("#Logical qubits", fontsize=FONTSIZE * 1.5)
        plt.ylim(0, 30)

    ax.text(
        0.1,
        1.1,
        "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    legend = plt.legend(handles=handles, loc="upper left", frameon=True, ncols=1, columnspacing=1.5)

    plt.tick_params(axis="both", labelsize=14)
    plt.ylabel("Runtime [s]", fontsize=FONTSIZE * 1.5)

    plt.grid(axis="y", which="major", linestyle="--", alpha=0.5)
    fig.subplots_adjust(left=0.24, right=0.95, top=0.88, bottom=0.19)
    plt.savefig(filename, format="pdf")
    plt.close(fig)



if __name__ == "__main__":
    # Scalability with respect to hardware size
    runtime_scaling(reproduce=False)

    # Scalability with respect to number of logical qubits
    runtime_scaling_logical_qubits(reproduce=False)