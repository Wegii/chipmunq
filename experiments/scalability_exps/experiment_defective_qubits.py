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
    pastel_blue   = "#5c79bd"
    pastel_orange = "#ef8d38"
    colors = [pastel_blue, pastel_orange]
    hatches = ['/', '\\']  # one hatch per placement mode

    labels = placement_modes  # ["default", "size_aware"]

    # ----------- Depth Overhead -------------
    fig, ax = plt.subplots(figsize=(5, 5))

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
    ax.set_ylabel("Circuit Depth Overhead")
    ax.legend()

    ax.text(
        0, 1.02, title_left,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold"
    )

    ax.text(
        0.71, 1.02, "Lower is better ↓",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color=pastel_blue,
    )

    fig.tight_layout()
    fig.savefig(f"{filename}_depth.png", dpi=300)
    plt.show()

    # ----------- 2Q Gate Overhead -------------
    fig, ax = plt.subplots(figsize=(5, 5))

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
    ax.legend()

    ax.text(
        0, 1.02, title_left,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold"
    )

    ax.text(
        0.71, 1.02, "Lower is better ↓",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color=pastel_blue,
    )

    fig.tight_layout()
    fig.savefig(f"{filename}_overhead.png", dpi=300)
    plt.show()


    # ----------- Backend Utilization -------------
    fig, ax = plt.subplots(figsize=(5, 5))

    for i, mode in enumerate(placement_modes):
        vals = [custom_utilization[mode][df][ks] for df in df_values]
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
    ax.set_ylabel("Utilization")
    ax.legend()
    ax.set_ylim(0, 1)

    ax.text(
        0, 1.02, title_left,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold"
    )

    ax.text(
        0.71, 1.02, "Higher is better ↑",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color=pastel_blue,
    )

    fig.tight_layout()
    fig.savefig(f"{filename}_utilization.png", dpi=300)
    plt.show()


def plot_utilization():
    pass


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
        defective_qubits = [1, 2, 3]

        # Compile a circuit to the defect free backend during the first iteration
        defect_free_compilation = True

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
                        #        filename=f"experiments/evaluation/defective_qubits/{bc}_{pp}_layout_{df}_defect_free.png")
                        #defect_free_compilation = False
                    
                    
                    #plot_circuit_layout(defective_circuit,
                    #                    backend,
                    #                    filename=f"experiments/evaluation/defective_qubits/{bc}_{pp}_layout_{df}.png")
                    

                    def num_2q_gates(circuit):
                        ops = circuit.count_ops()
                        two_qubit_gate_names = ["cx", "cz", "swap"]
                        return sum(ops.get(g, 0) for g in two_qubit_gate_names)

                    
                    # Calculate qpu utilization
                    custom_utilization[pp][df][ks] = calculate_qpu_utilization(defective_circuit, backend)

                    custom_depth[pp][df][ks] = defective_circuit.depth() - (defect_free_circuit).depth()
                    custom_overhead[pp][df][ks] = num_2q_gates(defective_circuit) - num_2q_gates(defect_free_circuit)

        print(custom_utilization)
        plot_combined(custom_depth,
                      custom_overhead,
                      custom_utilization,
                      title_left = "b) Multi patch configuration" if bc == "multi_patch" else "a) Single patch configuration",
                      filename = f"experiments/evaluation/defective_qubits/{bc}_overhead")

            
if __name__ == "__main__":
    run_exp_defective()