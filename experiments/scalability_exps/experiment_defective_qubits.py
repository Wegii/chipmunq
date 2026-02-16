# 3. How do defective qubits affect the resulting circuit?


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
import pickle
from experiments.utils import *
from collections import defaultdict


def plot_combined(custom_depth, custom_overhead, custom_utilization, title_left: str, filename: str = ""):

    # placement modes (outer keys)
    placement_modes = list(custom_depth.keys())  # ["default", "size_aware"]

    # defective qubit counts (inner keys)
    df_values = sorted(custom_depth[placement_modes[0]].keys())  # [1,2,3]

    ks = list(custom_depth[placement_modes[0]][df_values[0]].keys())[0]

    # X-axis (one position per defective-qubit count)
    x = np.arange(len(df_values))  # [0,1,2]

    # Two bars per group
    width = 0.35

    # Colors
    pastel_blue = "#A7D9ED" #'#A7D9ED'
    pastel_orange = "lightcoral" # '#6476AD' # '#F7C6A2'
    if title_left == "Single patch configuration":
        tl_label = ["a) ", "b) ", "c) "]
    else:
        tl_label = ["d) ", "e) ", "f) "]

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

    colors = [pastel_blue, pastel_orange]
    hatches = ['//', 'o']  # one hatch per placement mode

    labels = placement_modes  # ["default", "size_aware"]

    # ----------- Depth Overhead -------------
    #fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.6, WIDTH_FIGSIZE))
    #fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.5, WIDTH_FIGSIZE*0.92))
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.5, WIDTH_FIGSIZE*0.5))

    for i, mode in enumerate(placement_modes):
        vals = [custom_depth[mode][df][ks] for df in df_values]
        ax.bar(
            x + i * width - width/2,
            vals,
            width,
            label=labels[i],
            color=colors[i],
            hatch=hatches[i],
            edgecolor='black'
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(df) for df in df_values])
    ax.set_xlabel("#Defective qubits")
    ax.set_ylabel("Depth overhead")
    #ax.legend(loc='upper left')
    ax.set_ylim(0, 1250)

    ax.text(
        .1, 1.02, tl_label[0] + title_left,
        transform=ax.transAxes,
        fontweight="bold"
    )

    ax.text(
        0.27, 1.15, "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    fig.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.2)
    #fig.subplots_adjust(left=0.16, right=0.97, top=0.89, bottom=0.13)
    fig.savefig(f"{filename}_depth.pdf", format="pdf")
    plt.close(fig)

    # ----------- 2Q Gate Overhead -------------
    #fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.6, WIDTH_FIGSIZE))
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.5, WIDTH_FIGSIZE*0.5))

    for i, mode in enumerate(placement_modes):
        vals = [custom_overhead[mode][df][ks] for df in df_values]
        ax.bar(
            x + i * width - width/2,
            vals,
            width,
            label=labels[i],
            color=colors[i],
            hatch=hatches[i],
            edgecolor='black'
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(df) for df in df_values])
    ax.set_xlabel("#Defective qubits")
    ax.set_ylabel("#2q gate overhead")
    #ax.legend(loc='upper left')
    ax.set_ylim(0, 5500)

    ax.text(
        .1, 1.02, tl_label[1] + title_left,
        transform=ax.transAxes,
        fontweight="bold"
    )

    ax.text(
        0.27, 1.15, "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    fig.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.2)
    fig.savefig(f"{filename}_overhead.pdf", format="pdf")
    plt.close(fig)


    # ----------- Backend Utilization -------------
    #fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.6, WIDTH_FIGSIZE*0.7))
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.5, WIDTH_FIGSIZE*0.5))

    handles = []
    for i, mode in enumerate(placement_modes):
        vals = [custom_utilization[mode][df][ks] for df in df_values]
        handle = ax.bar(
            x + i * width - width/2,
            vals,
            width,
            yerr=0.1,          
            capsize=4,      
            error_kw={'elinewidth': 2, 'ecolor': 'black'},
            label=labels[i],
            color=colors[i],
            hatch=hatches[i],
            edgecolor='black'
        )
        handles.append(handle)

    ax.set_xticks(x)
    ax.set_xticklabels([str(df) for df in df_values])
    ax.set_xlabel("#Defective qubits")
    ax.set_ylabel("Utilization")
    #ax.legend(loc='upper left')
    ax.set_ylim(0, 1)

    ax.text(
        .1, 1.02, tl_label[2] + title_left,
        transform=ax.transAxes,
        fontweight="bold"
    )

    ax.text(
        0.27, 1.15, "Higher is better ↑",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    fig.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.2)
    fig.savefig(f"{filename}_utilization.pdf", format="pdf")
    plt.close(fig)

    legend_fig = plt.figure(figsize=(3, 2))
    legend = legend_fig.legend(handles = [handles[0], handles[1]],
                               loc = 'center',
                               frameon = False,
                               ncols = 3)
    legend_fig.savefig(filename + 'legend.pdf', bbox_inches='tight', format="pdf")
    plt.close(legend_fig)


def ci95_bootstrap(values, df_values, mode, ks):
    means = []
    err_low = []
    err_high = []
    for df in df_values:
        val = list(values[mode][df][ks].values())
        mean = np.mean(val)
        # Create fake replications by sampling own data with replacement
        boot_means = [np.mean(np.random.choice(val, size=len(val), replace=True)) 
                    for _ in range(5000)]
        # Find the bounds where 95% of those means fall
        low_perc = np.percentile(boot_means, 2.5)
        high_perc = np.percentile(boot_means, 97.5)
        
        means.append(mean)
        err_low.append(mean - low_perc)
        err_high.append(high_perc - mean)
    
    return means, [err_low, err_high]


def plot_combined_backends(custom_depth,
                           custom_overhead,
                           custom_utilization,
                           sabre_depth,
                           sabre_overhead,
                           sabre_utilization,
                           filename: str = ""):
    
    # placement modes (outer keys)
    placement_modes = list(custom_depth[0].keys())  # ["default", "size_aware"]

    # defective qubit counts (inner keys)
    df_values = sorted(custom_depth[0][placement_modes[0]].keys())  # [1,2,3]

    ks = list(custom_depth[0][placement_modes[0]][df_values[0]].keys())[0]

    # X-axis (one position per defective-qubit count)
    x = np.arange(len(df_values))  # [0,1,2]

    # Two bars per group
    width = 0.4/4#0.35/4#0.35/2

    title_left = ""

    # Colors
    colors = ["#4682B4", "#AEC6CF", "#F08080", "#F7C6A2"]
    hatches = ['//', 'o']  # one hatch per placement mode
   
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


    labels = ["center", "size-aware"]
    
    # ----------- Depth Overhead -------------
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.5, WIDTH_FIGSIZE*0.5))

    for i, mode in enumerate(placement_modes):
        # Single patch
        values_mean_custom, values_err_custom = ci95_bootstrap(custom_depth[0], df_values, mode, ks)
        values_mean_sabre, values_err_sabre = ci95_bootstrap(sabre_depth[0], df_values, mode, ks)

        # Custom
        ax.bar(
            x + i * 2*width - 3.5*width,
            values_mean_custom,
            width,
            yerr=values_err_custom,          
            capsize=2,      
            error_kw={'elinewidth': 1.5, 'ecolor': 'black'},
            label=labels[i],
            color=colors[i],
            hatch=hatches[0],
            edgecolor='black'
        )
        # SABRE
        ax.bar(
            x + i * 2*width - 2.5*width,
            values_mean_sabre,
            width,
            yerr=values_err_sabre,          
            capsize=2,      
            error_kw={'elinewidth': 1.5, 'ecolor': 'black'},
            label=labels[i],
            color=colors[i],
            hatch=hatches[1],
            edgecolor='black'
        )
        

        # Multi patch
        values_mean_custom, values_err_custom = ci95_bootstrap(custom_depth[1], df_values, mode, ks)
        values_mean_sabre, values_err_sabre = ci95_bootstrap(sabre_depth[1], df_values, mode, ks)
        # Custom
        ax.bar(
            x + (2+i) * width*2 - 3.5*width,#1.5*width,
            values_mean_custom,
            width,
            yerr=values_err_custom,          
            capsize=2,      
            error_kw={'elinewidth': 1.5, 'ecolor': 'black'},
            label=labels[i] + "multi",
            color=colors[2+i],
            hatch=hatches[0],
            edgecolor='black'
        )
        # SABRE
        ax.bar(
            x + (2+i) * width*2 - 2.5*width,
            values_mean_sabre,
            width,
            yerr=values_err_sabre,          
            capsize=2,      
            error_kw={'elinewidth': 1.5, 'ecolor': 'black'},
            label=labels[i] + "multi",
            color=colors[2+i],
            hatch=hatches[1],
            edgecolor='black'
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(df) for df in df_values])
    ax.set_xlabel("#Defective qubits")
    ax.set_ylabel("Depth overhead")
    #ax.legend(loc='upper left')
    #ax.set_ylim(0, 1250)

    ax.text(
        -0.1, 1.02, "a) Defective qubits affecting circuit depth",
        transform=ax.transAxes,
        fontweight="bold"
    )

    ax.text(
        0.27, 1.15, "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    fig.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.21)
    fig.savefig(f"{filename}_depth.pdf", format="pdf")
    plt.close(fig)



    
    # ----------- 2Q Gate Overhead -------------
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.5, WIDTH_FIGSIZE*0.5))

    for i, mode in enumerate(placement_modes):
        # Single patch
        values_mean_custom, values_err_custom = ci95_bootstrap(custom_overhead[0], df_values, mode, ks)
        values_mean_sabre, values_err_sabre = ci95_bootstrap(sabre_overhead[0], df_values, mode, ks)

        # Custom
        ax.bar(
            x + i * 2*width - 3.5*width,
            values_mean_custom,
            width,
            yerr=values_err_custom,          
            capsize=2,      
            error_kw={'elinewidth': 1.5, 'ecolor': 'black'},
            label=labels[i],
            color=colors[i],
            hatch=hatches[0],
            edgecolor='black'
        )
        # SABRE
        ax.bar(
            x + i * 2*width - 2.5*width,
            values_mean_sabre,
            width,
            yerr=values_err_sabre,          
            capsize=2,      
            error_kw={'elinewidth': 1.5, 'ecolor': 'black'},
            label=labels[i],
            color=colors[i],
            hatch=hatches[1],
            edgecolor='black'
        )
        

        # Multi patch
        values_mean_custom, values_err_custom = ci95_bootstrap(custom_overhead[1], df_values, mode, ks)
        values_mean_sabre, values_err_sabre = ci95_bootstrap(sabre_overhead[1], df_values, mode, ks)
        # Custom
        ax.bar(
            x + (2+i) * width*2 - 3.5*width,#1.5*width,
            values_mean_custom,
            width,
            yerr=values_err_custom,          
            capsize=2,      
            error_kw={'elinewidth': 1.5, 'ecolor': 'black'},
            label=labels[i] + "multi",
            color=colors[2+i],
            hatch=hatches[0],
            edgecolor='black'
        )
        # SABRE
        ax.bar(
            x + (2+i) * width*2 - 2.5*width,
            values_mean_sabre,
            width,
            yerr=values_err_sabre,          
            capsize=2,      
            error_kw={'elinewidth': 1.5, 'ecolor': 'black'},
            label=labels[i] + "multi",
            color=colors[2+i],
            hatch=hatches[1],
            edgecolor='black'
        )


    ax.set_xticks(x)
    ax.set_xticklabels([str(df) for df in df_values])
    ax.set_xlabel("#Defective qubits")
    ax.set_ylabel("#2q gate overhead")
    #ax.legend(loc='upper left')
    #ax.set_ylim(0, 5500)

    ax.text(
        -0.05, 1.02, "b) Defective qubits affecting #2q gates",
        transform=ax.transAxes,
        fontweight="bold"
    )

    ax.text(
        0.27, 1.15, "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    fig.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.21)
    fig.savefig(f"{filename}_overhead.pdf", format="pdf")
    plt.close(fig)



    
    # ----------- Backend Utilization -------------
    #fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.6, WIDTH_FIGSIZE*0.7))
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.5, WIDTH_FIGSIZE*0.5))

    handles = []
    for i, mode in enumerate(placement_modes):
        # Single patch
        values_mean_custom, values_err_custom = ci95_bootstrap(custom_utilization[0], df_values, mode, ks)
        values_mean_sabre, values_err_sabre = ci95_bootstrap(sabre_utilization[0], df_values, mode, ks)

        # Custom
        h = ax.bar(
            x + i * 2*width - 3.5*width,
            values_mean_custom,
            width,
            yerr=values_err_custom,          
            capsize=2,      
            error_kw={'elinewidth': 1.5, 'ecolor': 'black'},
            label="Chipmunq Single patch: " + labels[i],
            color=colors[i],
            hatch=hatches[0],
            edgecolor='black'
        )
        handles.append(h)
        # SABRE
        h = ax.bar(
            x + i * 2*width - 2.5*width,
            values_mean_sabre,
            width,
            yerr=values_err_sabre,          
            capsize=2,      
            error_kw={'elinewidth': 1.5, 'ecolor': 'black'},
            label="LightSABRE Single patch: " + labels[i],
            color=colors[i],
            hatch=hatches[1],
            edgecolor='black'
        )
        handles.append(h)
        

        # Multi patch
        values_mean_custom, values_err_custom = ci95_bootstrap(custom_utilization[1], df_values, mode, ks)
        values_mean_sabre, values_err_sabre = ci95_bootstrap(sabre_utilization[1], df_values, mode, ks)
        # Custom
        h = ax.bar(
            x + (2+i) * width*2 - 3.5*width,#1.5*width,
            values_mean_custom,
            width,
            yerr=values_err_custom,          
            capsize=2,      
            error_kw={'elinewidth': 1.5, 'ecolor': 'black'},
            label="Chipmunq Multi patch: " + labels[i],
            color=colors[2+i],
            hatch=hatches[0],
            edgecolor='black'
        )
        handles.append(h)


        # SABRE
        h = ax.bar(
            x + (2+i) * width*2 - 2.5*width,
            values_mean_sabre,
            width,
            yerr=values_err_sabre,          
            capsize=2,      
            error_kw={'elinewidth': 1.5, 'ecolor': 'black'},
            label="LightSABRE Multi patch: " + labels[i],
            color=colors[2+i],
            hatch=hatches[1],
            edgecolor='black'
        )
        handles.append(h)


    ax.set_xticks(x)
    ax.set_xticklabels([str(df) for df in df_values])
    ax.set_xlabel("#Defective qubits")
    ax.set_ylabel("Utilization")
    #ax.legend(loc='upper left')
    ax.set_ylim(0, 1.2)

    ax.text(
        -0.22, 1.02, "c) Defective qubits affecting chiplet utilization",
        transform=ax.transAxes,
        fontweight="bold"
    )

    ax.text(
        0.27, 1.15, "Higher is better ↑",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    fig.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.21)
    fig.savefig(f"{filename}_utilization.pdf", format="pdf")
    plt.close(fig)

    legend_fig = plt.figure(figsize=(3, 2))
    legend = legend_fig.legend(handles = handles,
                               loc = 'center',
                               frameon = False,
                               ncols = 4)
    legend_fig.savefig(filename + 'legend.pdf', bbox_inches='tight', format="pdf")
    plt.close(legend_fig)
    

def calculate_qpu_utilization(circuit, backend):

    # Iterate over circuit
    num_qubits_per_chiplet = backend.n * backend.m
    utilized_chiplets = set()

    num_qubits = backend.num_qubits
    cmap = backend.coupling_map

    qubits = []
    qubit_labels = [""] * num_qubits

    bit_locations = {
        bit: {"register": register, "index": index}
        for register in circuit._layout.initial_layout.get_registers()
        for index, bit in enumerate(register)
    }
    for index, qubit in enumerate(circuit._layout.initial_layout.get_virtual_bits()):
        if qubit not in bit_locations:
            bit_locations[qubit] = {"register": None, "index": index}

    for key, val in circuit._layout.initial_layout.get_virtual_bits().items():
        bit_register = bit_locations[key]["register"]
        if bit_register is None or bit_register.name != "ancilla":
            qubits.append(val)
            qubit_labels[val] = str(bit_locations[key]["index"])

    utilized_qubits = 0
    for qubit in qubits:
        if qubit != '':
            utilized_chiplets.add(backend.node_to_chiplet[int(qubit)])
            utilized_qubits += 1

    print(utilized_chiplets)
    print(f"Calculated utilization of {utilized_qubits/(len(utilized_chiplets)*num_qubits_per_chiplet)    }")
    return utilized_qubits/(len(utilized_chiplets)*num_qubits_per_chiplet)  


def recursive_dict():
    return defaultdict(recursive_dict)


def run_exp_defective():

    # Backend configuration
    num_inter_chiplet_connections = 8
    ps_inter = 1e-4

    # Compilation configuration
    compilation = ["sabre", "custom"]

    # Placement location of patches on a 
    patch_placement = ["center", "size_aware"]

    # Run for two backend configuration:
    # - Backend fits single patch 
    # - Backend fits multiple patches
    backend_config = ["single_patch", "multi_patch"]
    
    """
    for comp in compilation:
        for bc in backend_config:

            custom_depth = {}
            custom_overhead = {}

            custom_depth = recursive_dict()
            custom_overhead = recursive_dict()
            custom_utilization = recursive_dict()

            defective_qubits = [0, 1, 2, 3]
            num_iterations = 10
            num_dupl = 1

            if bc == "single_patch":
                nx, nm = 15, 8
            elif bc == "multi_patch":
                nx, nm = 23, 14

            # Compile a circuit to the defect free backend during the first iteration
            defect_free_compilation = True
            
            # Iterate over placement methods
            for pp in patch_placement:
                # Iterate over code size
                for ks in [2]:#[1, 2, 3, 4]
                    # Generate circuit
                    circuit, partitions = get_tqec_cnot_rotated(distance_scale = ks,
                                                                n1 = 1,
                                                                n2 = 0)
                    # Stim to qiskit
                    stim_code_circuit = StimCodeCircuit(stim_circuit = circuit)
                    
                    # Iterate over number of defective qubits
                    for df in defective_qubits:
                        # Perform multiple iterations, since defective qubits are selected randomly
                        for run in range(0, num_iterations):
                            ic = 0
                            while True:
                                try:
                                    backend = BackendChipletV2(size = (num_dupl*6, num_dupl*6, nx, nm),
                                                        n_inter = num_inter_chiplet_connections,
                                                        connectivity = "nn",
                                                        topology = "rotated_grid",
                                                        inter_chiplet_noise = ps_inter,
                                                        inter_chiplet_amplification = 1,
                                                        inter_chiplet_noise_type = "constant",
                                                        num_defective_qubits=df,
                                                        chiplet_seed = run + 42 + ic,
                                                        sabre_defective = comp == "sabre"
                                                    )

                                    # Custom transpilation
                                    if comp == "sabre":
                                        # In order to tackle defects into account, it is necessary to 
                                        defective_circuit = sabre_transpilation(stim_code_circuit.qc, backend)
                                    else:
                                        defective_circuit = custom_cost_transpilation(stim_code_circuit.qc,
                                                                                      backend,
                                                                                      pre_defined_partitions = partitions,
                                                                                      patch_initialization = pp)
                                    break                                
                                except Exception as e:
                                    ic += 1
                                    print("Unable to place patches given location of defective qubits!")
                                    print("Retrying...")
                            

                            def num_2q_gates(circuit):
                                ops = circuit.count_ops()
                                two_qubit_gate_names = ["cx", "cz", "swap"]
                                return sum(ops.get(g, 0) for g in two_qubit_gate_names)

                            # Calculate qpu utilization
                            custom_utilization[pp][df][ks][run] = calculate_qpu_utilization(defective_circuit, backend)

                            custom_depth[pp][df][ks][run] = defective_circuit.depth() - (stim_code_circuit.qc).depth()
                            custom_overhead[pp][df][ks][run] = num_2q_gates(defective_circuit) - num_2q_gates(stim_code_circuit.qc)

        
            with open(f"experiments/evaluation/defective_qubits/{comp}_depth_{bc}.pkl", "wb") as f:
                pickle.dump(custom_depth, f)
            with open(f"experiments/evaluation/defective_qubits/{comp}_overhead_{bc}.pkl", "wb") as f:
                pickle.dump(custom_overhead, f)
            with open(f"experiments/evaluation/defective_qubits/{comp}_utilization_{bc}.pkl", "wb") as f:
                pickle.dump(custom_utilization, f)
    """

    """
        with open(f"experiments/evaluation/defective_qubits/custom_depth_{bc}.pkl", "rb") as f:
            custom_depth = pickle.load(f)
        with open(f"experiments/evaluation/defective_qubits/custom_overhead_{bc}.pkl", "rb") as f:
            custom_overhead = pickle.load(f)
        with open(f"experiments/evaluation/defective_qubits/custom_utilization_{bc}.pkl", "rb") as f:
            custom_utilization = pickle.load(f)

        plot_combined(custom_depth,
                      custom_overhead,
                      custom_utilization,
                      title_left = ("Multi patch configuration" if bc == "multi_patch"
                                    else "Single patch configuration"),
                      filename = f"experiments/evaluation/defective_qubits/{bc}_overhead")
    """        

    backend_config = ["single_patch", "multi_patch"]
    custom_depth_combined = []
    custom_overhead_combined = []
    custom_utilization_combined = []
    sabre_depth_combined = []
    sabre_overhead_combined = []
    sabre_utilization_combined = []
    for bc in backend_config:       
        # SABRE
        with open(f"experiments/evaluation/defective_qubits/sabre_depth_{bc}.pkl", "rb") as f:
            sabre_depth = pickle.load(f)
        with open(f"experiments/evaluation/defective_qubits/sabre_overhead_{bc}.pkl", "rb") as f:
            sabre_overhead = pickle.load(f)
        with open(f"experiments/evaluation/defective_qubits/sabre_utilization_{bc}.pkl", "rb") as f:
            sabre_utilization = pickle.load(f)

        sabre_depth_combined.append(sabre_depth)
        sabre_overhead_combined.append(sabre_overhead)
        sabre_utilization_combined.append(sabre_utilization)

        # Custom
        with open(f"experiments/evaluation/defective_qubits/custom_depth_{bc}.pkl", "rb") as f:
            custom_depth = pickle.load(f)
        with open(f"experiments/evaluation/defective_qubits/custom_overhead_{bc}.pkl", "rb") as f:
            custom_overhead = pickle.load(f)
        with open(f"experiments/evaluation/defective_qubits/custom_utilization_{bc}.pkl", "rb") as f:
            custom_utilization = pickle.load(f)

        custom_depth_combined.append(custom_depth)
        custom_overhead_combined.append(custom_overhead)
        custom_utilization_combined.append(custom_utilization)

    plot_combined_backends(custom_depth_combined,
                            custom_overhead_combined,
                            custom_utilization_combined,
                            sabre_depth_combined,
                            sabre_overhead_combined,
                            sabre_utilization_combined,
                            filename = f"experiments/evaluation/defective_qubits/combined_overhead")

            
if __name__ == "__main__":
    run_exp_defective()