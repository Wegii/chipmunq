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
    ax.set_xlabel("#Defective Qubits")
    ax.set_ylabel("Depth Overhead")
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
    ax.set_xlabel("#Defective Qubits")
    ax.set_ylabel("2q Gate Overhead")
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
    ax.set_xlabel("#Defective Qubits")
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


def plot_combined_backends(custom_depth, custom_overhead, custom_utilization, filename: str = ""):

    # placement modes (outer keys)
    placement_modes = list(custom_depth[0].keys())  # ["default", "size_aware"]

    # defective qubit counts (inner keys)
    df_values = sorted(custom_depth[0][placement_modes[0]].keys())  # [1,2,3]

    ks = list(custom_depth[0][placement_modes[0]][df_values[0]].keys())[0]

    # X-axis (one position per defective-qubit count)
    x = np.arange(len(df_values))  # [0,1,2]

    # Two bars per group
    width = 0.35/2

    title_left = ""

    # Colors
    colors = ["#4682B4", "#AEC6CF", "#F7C6A2", "#F08080"]
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



    labels = ["center", "size_aware"]  # ["default", "size_aware"]

    # ----------- Depth Overhead -------------
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.5, WIDTH_FIGSIZE*0.5))

    for i, mode in enumerate(placement_modes):
        vals = [custom_depth[0][mode][df][ks] for df in df_values]
        ax.bar(
            x + i * width - 1.5*width,
            vals,
            width,
            label=labels[i],
            color=colors[i],
            hatch=hatches[i],
            edgecolor='black'
        )

        vals = [custom_depth[1][mode][df][ks] for df in df_values]
        ax.bar(
            x + (2+i) * width - 1.5*width,
            vals,
            width,
            label=labels[i] + "multi",
            color=colors[2+i],
            hatch=hatches[i],
            edgecolor='black'
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(df) for df in df_values])
    ax.set_xlabel("#Defective Qubits")
    ax.set_ylabel("Depth Overhead")
    #ax.legend(loc='upper left')
    ax.set_ylim(0, 1250)

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

    fig.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.2)
    fig.savefig(f"{filename}_depth.pdf", format="pdf")
    plt.close(fig)



    
    # ----------- 2Q Gate Overhead -------------
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.5, WIDTH_FIGSIZE*0.5))

    for i, mode in enumerate(placement_modes):
        vals = [custom_overhead[0][mode][df][ks] for df in df_values]
        ax.bar(
            x + i * width - 1.5*width,
            vals,
            width,
            label=labels[i],
            color=colors[i],
            hatch=hatches[i],
            edgecolor='black'
        )

        vals = [custom_overhead[1][mode][df][ks] for df in df_values]
        ax.bar(
            x + (2+i) * width - 1.5*width,
            vals,
            width,
            label=labels[i] + "multi",
            color=colors[2+i],
            hatch=hatches[i],
            edgecolor='black'
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(df) for df in df_values])
    ax.set_xlabel("#Defective Qubits")
    ax.set_ylabel("2q Gate Overhead")
    #ax.legend(loc='upper left')
    ax.set_ylim(0, 5500)

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

    fig.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.2)
    fig.savefig(f"{filename}_overhead.pdf", format="pdf")
    plt.close(fig)



    
    # ----------- Backend Utilization -------------
    #fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.6, WIDTH_FIGSIZE*0.7))
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.5, WIDTH_FIGSIZE*0.5))

    handles = []
    for i, mode in enumerate(placement_modes):
        vals = [custom_utilization[0][mode][df][ks] for df in df_values]
        h = ax.bar(
            x + i * width - 1.5*width,
            vals,
            width,
            yerr=0.1,          
            capsize=4,      
            error_kw={'elinewidth': 2, 'ecolor': 'black'},
            label="single patch: " + labels[i],
            color=colors[i],
            hatch=hatches[i],
            edgecolor='black'
        )
        handles.append(h)

        vals = [custom_utilization[1][mode][df][ks] for df in df_values]
        h = ax.bar(
            x + (2+i) * width - 1.5*width,
            vals,
            width,
            yerr=0.1,          
            capsize=4,      
            error_kw={'elinewidth': 2, 'ecolor': 'black'},
            label="multi patch: " + labels[i],
            color=colors[2+i],
            hatch=hatches[i],
            edgecolor='black'
        )
        handles.append(h)

    ax.set_xticks(x)
    ax.set_xticklabels([str(df) for df in df_values])
    ax.set_xlabel("#Defective Qubits")
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

    fig.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.2)
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


def run_exp_defective():

    # Backend configuration
    num_inter_chiplet_connections = 8
    ps_inter = 1e-4

    # Placement location of patches on a 
    patch_placement = ["center", "size_aware"]

    # Run for two backend configuration:
    # - Backend fits single patch 
    # - Backend fits multiple patches
    backend_config = ["single_patch", "multi_patch"]

    for bc in backend_config:

        custom_depth = {}
        custom_overhead = {}

        custom_depth = {pp: {} for pp in patch_placement}
        custom_overhead = {pp: {} for pp in patch_placement}
        custom_utilization = {pp: {} for pp in patch_placement}

        np = 1
        defective_qubits = [0, 1, 2, 3]

        # Compile a circuit to the defect free backend during the first iteration
        defect_free_compilation = True
        """
        for pp in patch_placement:
            for ks in [2]:#[1, 2, 3, 4]
                for df in defective_qubits:

                    if df not in custom_depth[pp]:
                        custom_depth[pp][df] = {}
                        custom_overhead[pp][df] = {}
                        custom_utilization[pp][df] = {}
                        

                    # Generate circuit
                    circuit, partitions = get_tqec_cnot_rotated(distance_scale = ks,
                                                                n1 = np,
                                                                n2 = 0)

                    if bc == "single_patch":
                        nx, nm = 15, 8
                    elif bc == "multi_patch":
                        nx, nm = 23, 14

                    backend = BackendChipletV2(size = (np*6, np*6, nx, nm),
                                                n_inter = num_inter_chiplet_connections,
                                                connectivity = "nn",
                                                topology = "rotated_grid",
                                                inter_chiplet_noise = ps_inter,
                                                inter_chiplet_amplification = 1,
                                                inter_chiplet_noise_type = "constant",
                                                num_defective_qubits=df,
                                            )
                    
                    backend_non_defective = BackendChipletV2(size = (np*6, np*6, nx, nm),
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
                    print("Df")
                    
                    defective_circuit = custom_cost_transpilation(stim_code_circuit.qc,
                                                            backend,
                                                            pre_defined_partitions = partitions,
                                                            patch_initialization = pp)

                    if defect_free_compilation:
                        if bc == "multi_patch":
                            defect_free_circuit = custom_cost_transpilation(stim_code_circuit.qc,
                                                                        backend_non_defective,
                                                                        pre_defined_partitions=partitions,
                                                                        patch_initialization = "size_aware")
                        else:
                            defect_free_circuit = custom_cost_transpilation(stim_code_circuit.qc,
                                                                        backend_non_defective,
                                                                        pre_defined_partitions=partitions,
                                                                        patch_initialization = "center")

                        #plot_circuit_layout(defect_free_circuit,
                        #        backend_non_defective,
                        #        filename=f"experiments/evaluation/defective_qubits/backend/{bc}_{pp}_layout_{df}_defect_free.png")
                        #defect_free_compilation = False
                    
                    
                    #plot_circuit_layout(defective_circuit,
                    #                    backend,
                    #                    filename=f"experiments/evaluation/defective_qubits/backend/{bc}_{pp}_layout_{df}.png")
                    

                    def num_2q_gates(circuit):
                        ops = circuit.count_ops()
                        two_qubit_gate_names = ["cx", "cz", "swap"]
                        return sum(ops.get(g, 0) for g in two_qubit_gate_names)

                    
                    # Calculate qpu utilization
                    custom_utilization[pp][df][ks] = calculate_qpu_utilization(defective_circuit, backend)

                    custom_depth[pp][df][ks] = defective_circuit.depth() - (stim_code_circuit.qc).depth()
                    custom_overhead[pp][df][ks] = num_2q_gates(defective_circuit) - num_2q_gates(stim_code_circuit.qc)

    
        with open(f"experiments/evaluation/defective_qubits/custom_depth_{bc}.pkl", "wb") as f:
            pickle.dump(custom_depth, f)
        with open(f"experiments/evaluation/defective_qubits/custom_overhead_{bc}.pkl", "wb") as f:
            pickle.dump(custom_overhead, f)
        with open(f"experiments/evaluation/defective_qubits/custom_utilization_{bc}.pkl", "wb") as f:
            pickle.dump(custom_utilization, f)
        

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
    for bc in backend_config:            
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
                            filename = f"experiments/evaluation/defective_qubits/combined_overhead")

            
if __name__ == "__main__":
    run_exp_defective()