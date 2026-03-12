from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.getcwd(), "."))

from experiments.exp_utils.circuit_generator import get_tqec_cnot_rotated
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit
from experiments.exp_utils.simulation_utils import *
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.utils import *

# Plotting
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def ci95_bootstrap(values, df_values, mode, ks):
    means = []
    err_low = []
    err_high = []
    for df in df_values:
        val = list(values[mode][df][ks].values())
        mean = np.mean(val)
        # Create fake replications by sampling own data with replacement
        boot_means = [np.mean(np.random.choice(val, size=len(val), replace=True)) for _ in range(5000)]
        # Find the bounds where 95% of those means fall
        low_perc = np.percentile(boot_means, 2.5)
        high_perc = np.percentile(boot_means, 97.5)

        means.append(mean)
        err_low.append(mean - low_perc)
        err_high.append(high_perc - mean)

    return means, [err_low, err_high]


def plot_combined_backends(
    custom_depth,
    custom_overhead,
    custom_utilization,
    sabre_depth,
    sabre_overhead,
    sabre_utilization,
    filename: str = "",
):

    # placement modes (outer keys)
    placement_modes = list(custom_depth[0].keys())  # ["default", "size_aware"]

    # defective qubit counts (inner keys)
    df_values = sorted(custom_depth[0][placement_modes[0]].keys())

    ks = list(custom_depth[0][placement_modes[0]][df_values[0]].keys())[0]

    # X-axis (one position per defective-qubit count)
    x = np.arange(len(df_values))  # [0,1,2]

    # Two bars per group
    width = 0.5 / 4  # 0.35/4#0.35/2

    title_left = ""

    # Colors
    # colors = ["#4682B4", "#AEC6CF", "#F08080", "#F7C6A2"]
    colors_custom = ["#A7D9ED", "#5B9BD5", "#D9D9D9", "#7F7F7F"]
    colors_sabre = ["#F7C6A2", "#E68A5C", "#F7A2A2", "#D65C5C"]

    hatches = ["...", "//", "xxx", "ooo"]  # one hatch per placement mode

    tex_fonts = {
        # Use LaTeX to write all text
        # "text.usetex": True,
        "font.family": "serif",
        # Font sizes
        "axes.labelsize": FONTSIZE * 1.5,
        "font.size": FONTSIZE * 1.2,
        "legend.fontsize": (FONTSIZE - 2) * 1.5,
        "xtick.labelsize": (FONTSIZE - 1) * 1.3,
        "ytick.labelsize": (FONTSIZE - 1) * 1.3,
        "axes.titlesize": 10,
        # Hatches
        "hatch.linewidth": 0.5,
        # Line and marker styles
        "lines.linewidth": 2,
        "lines.markersize": 3,
        "lines.markeredgewidth": 1.5,
        "lines.markeredgecolor": "black",
        # Error bar cap size
        "errorbar.capsize": 3,
    }

    plt.rcParams.update(tex_fonts)

    labels = ["center", "size-aware"]

    # Depth Overhead
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE * 2.5, WIDTH_FIGSIZE * 0.5))

    for i, mode in enumerate(placement_modes):
        # Single patch
        values_mean_custom, values_err_custom = ci95_bootstrap(custom_depth[0], df_values, mode, ks)
        values_mean_sabre, values_err_sabre = ci95_bootstrap(sabre_depth[0], df_values, mode, ks)

        # Custom
        ax.bar(
            x + i * 1 * width - 2.5 * width,
            values_mean_custom,
            width,
            yerr=values_err_custom,
            capsize=2,
            error_kw={"elinewidth": 1, "ecolor": "black"},
            label=labels[i],
            color=colors_custom[i],
            hatch=hatches[i],
            edgecolor="black",
        )
        if i < 1:
            # SABRE
            ax.bar(
                x + i * 2 * width - 0.5 * width,
                values_mean_sabre,
                width,
                yerr=values_err_sabre,
                capsize=2,
                error_kw={"elinewidth": 1, "ecolor": "black"},
                label=labels[i],
                color=colors_sabre[i],
                # hatch=hatches[i],
                edgecolor="black",
            )

        # Multi patch
        values_mean_custom, values_err_custom = ci95_bootstrap(custom_depth[1], df_values, mode, ks)
        values_mean_sabre, values_err_sabre = ci95_bootstrap(sabre_depth[1], df_values, mode, ks)
        # Custom
        ax.bar(
            x + (2 + i) * width * 2 - 3.5 * width - i * width,
            values_mean_custom,
            width,
            yerr=values_err_custom,
            capsize=2,
            error_kw={"elinewidth": 1, "ecolor": "black"},
            label=labels[i] + "multi",
            color=colors_custom[i + 2],
            hatch=hatches[i],
            edgecolor="black",
        )
        if i < 1:
            # SABRE
            ax.bar(
                x + (2 + i) * width * 2 - 1.5 * width,
                values_mean_sabre,
                width,
                yerr=values_err_sabre,
                capsize=2,
                error_kw={"elinewidth": 1, "ecolor": "black"},
                label=labels[i] + "multi",
                color=colors_sabre[i + 2],
                # hatch=hatches[i + 2],
                edgecolor="black",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(df) for df in df_values])
    ax.set_xlabel("#Defective qubits")
    ax.set_ylabel("Depth overhead")
    # ax.legend(loc='upper left')
    # ax.set_ylim(0, 1250)

    ax.text(-0.1, 1.02, "a) Defective qubits affecting circuit depth", transform=ax.transAxes, fontweight="bold")

    ax.text(
        0.27,
        1.15,
        "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    fig.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.21)
    fig.savefig(f"{filename}_depth.pdf", format="pdf")
    plt.close(fig)

    # 2Q Gate Overhead
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE * 2.5, WIDTH_FIGSIZE * 0.5))

    for i, mode in enumerate(placement_modes):
        # Single patch
        values_mean_custom, values_err_custom = ci95_bootstrap(custom_overhead[0], df_values, mode, ks)
        values_mean_sabre, values_err_sabre = ci95_bootstrap(sabre_overhead[0], df_values, mode, ks)

        # Custom
        ax.bar(
            x + i * 1 * width - 2.5 * width,
            values_mean_custom,
            width,
            yerr=values_err_custom,
            capsize=2,
            error_kw={"elinewidth": 1, "ecolor": "black"},
            label=labels[i],
            color=colors_custom[i],
            hatch=hatches[i],
            edgecolor="black",
        )
        if i < 1:
            # SABRE
            ax.bar(
                x + i * 2 * width - 0.5 * width,
                values_mean_sabre,
                width,
                yerr=values_err_sabre,
                capsize=2,
                error_kw={"elinewidth": 1, "ecolor": "black"},
                label=labels[i],
                color=colors_sabre[i],
                # hatch=hatches[i],
                edgecolor="black",
            )

        # Multi patch
        values_mean_custom, values_err_custom = ci95_bootstrap(custom_overhead[1], df_values, mode, ks)
        values_mean_sabre, values_err_sabre = ci95_bootstrap(sabre_overhead[1], df_values, mode, ks)
        # Custom
        ax.bar(
            x + (2 + i) * width * 2 - 3.5 * width - i * width,
            values_mean_custom,
            width,
            yerr=values_err_custom,
            capsize=2,
            error_kw={"elinewidth": 1, "ecolor": "black"},
            label=labels[i] + "multi",
            color=colors_custom[i + 2],
            hatch=hatches[i],
            edgecolor="black",
        )
        if i < 1:
            # SABRE
            ax.bar(
                x + (2 + i) * width * 2 - 1.5 * width,
                values_mean_sabre,
                width,
                yerr=values_err_sabre,
                capsize=2,
                error_kw={"elinewidth": 1, "ecolor": "black"},
                label=labels[i] + "multi",
                color=colors_sabre[i + 2],
                # hatch=hatches[i + 2],
                edgecolor="black",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(df) for df in df_values])
    ax.set_xlabel("#Defective qubits")
    ax.set_ylabel("#2q gate overhead")
    # ax.legend(loc='upper left')
    # ax.set_ylim(0, 5500)

    ax.text(-0.05, 1.02, "b) Defective qubits affecting #2q gates", transform=ax.transAxes, fontweight="bold")

    ax.text(
        0.27,
        1.15,
        "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    fig.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.21)
    fig.savefig(f"{filename}_overhead.pdf", format="pdf")
    plt.close(fig)

    # Backend Utilization
    # fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.6, WIDTH_FIGSIZE*0.7))
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE * 2.5, WIDTH_FIGSIZE * 0.5))

    handles = []
    for i, mode in enumerate(placement_modes):
        # Single patch
        values_mean_custom, values_err_custom = ci95_bootstrap(custom_utilization[0], df_values, mode, ks)
        values_mean_sabre, values_err_sabre = ci95_bootstrap(sabre_utilization[0], df_values, mode, ks)

        # Custom
        h = ax.bar(
            x + i * 1 * width - 2.5 * width,
            values_mean_custom,
            width,
            yerr=values_err_custom,
            capsize=2,
            error_kw={"elinewidth": 1, "ecolor": "black"},
            label="Chipmunq, single patch, " + labels[i],
            color=colors_custom[i],
            hatch=hatches[i],
            edgecolor="black",
        )
        handles.append(h)
        if i < 1:
            # SABRE
            h = ax.bar(
                x + i * 2 * width - 0.5 * width,
                values_mean_sabre,
                width,
                yerr=values_err_sabre,
                capsize=2,
                error_kw={"elinewidth": 1, "ecolor": "black"},
                label="LightSABRE, single patch",
                color=colors_sabre[i],
                # hatch=hatches[i],
                edgecolor="black",
            )
            handles.append(h)

        # Multi patch
        values_mean_custom, values_err_custom = ci95_bootstrap(custom_utilization[1], df_values, mode, ks)
        values_mean_sabre, values_err_sabre = ci95_bootstrap(sabre_utilization[1], df_values, mode, ks)
        # Custom
        h = ax.bar(
            x + (2 + i) * width * 2 - 3.5 * width - i * width,
            values_mean_custom,
            width,
            yerr=values_err_custom,
            capsize=2,
            error_kw={"elinewidth": 1, "ecolor": "black"},
            label="Chipmunq, multi patch, " + labels[i],
            color=colors_custom[i + 2],
            hatch=hatches[i],
            edgecolor="black",
        )
        handles.append(h)

        if i < 1:
            # SABRE
            h = ax.bar(
                x + (2 + i) * width * 2 - 1.5 * width,
                values_mean_sabre,
                width,
                yerr=values_err_sabre,
                capsize=2,
                error_kw={"elinewidth": 1, "ecolor": "black"},
                label="LightSABRE, multi patch",
                color=colors_sabre[i + 2],
                # hatch=hatches[i + 2],
                edgecolor="black",
            )
            handles.append(h)

    ax.set_xticks(x)
    ax.set_xticklabels([str(df) for df in df_values])
    ax.set_xlabel("#Defective qubits")
    ax.set_ylabel("Utilization")
    # ax.legend(loc='upper left')
    ax.set_ylim(0, 1.2)

    ax.text(-0.22, 1.02, "c) Defective qubits affecting chiplet utilization", transform=ax.transAxes, fontweight="bold")

    ax.text(
        0.27,
        1.15,
        "Higher is better ↑",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    fig.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.21)
    fig.savefig(f"{filename}_utilization.pdf", format="pdf")
    plt.close(fig)

    legend_fig = plt.figure(figsize=(3, 2))
    legend = legend_fig.legend(
        handles=[handles[0], handles[4], handles[2], handles[5]],
        loc="center",
        frameon=False,
        ncols=4,
        columnspacing=1.5,
    )
    legend_fig.savefig(filename + "legend_custom.pdf", bbox_inches="tight", format="pdf")
    plt.close(legend_fig)

    legend_fig = plt.figure(figsize=(3, 2))
    legend = legend_fig.legend(
        handles=[handles[1], handles[3]], loc="center", frameon=False, ncols=4, columnspacing=1.5
    )
    legend_fig.savefig(filename + "legend_sabre.pdf", bbox_inches="tight", format="pdf")
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
        if qubit != "":
            utilized_chiplets.add(backend.node_to_chiplet[int(qubit)])
            utilized_qubits += 1

    print(utilized_chiplets)
    print(f"Calculated utilization of {utilized_qubits / (len(utilized_chiplets) * num_qubits_per_chiplet)}")
    return utilized_qubits / (len(utilized_chiplets) * num_qubits_per_chiplet)


def recursive_dict() -> defaultdict:
    return defaultdict(recursive_dict)


def run_exp_defective(reproduce: bool = False) -> None:

    # Backend configuration
    num_inter_chiplet_connections = 8
    ps_inter = 1e-4

    # Compilation configuration
    compilation = ["sabre", "custom"]

    # Placement location of patches on a
    patch_placement = ["center", "size_aware"]

    # Run for two backend configuration:
    # - Backend fits a single patch
    # - Backend fits multiple patches
    backend_config = ["single_patch", "multi_patch"]

    # Number of defective qubits
    defective_qubits = [0, 1, 2, 3]

    # Number of iterations per configuration
    num_iterations = 10

    # Backend size
    num_dupl = 1

    # Size of surface code
    code_size = [2]

    if reproduce:
        for comp in compilation:
            for bc in backend_config:
                custom_depth = {}
                custom_overhead = {}

                custom_depth = recursive_dict()
                custom_overhead = recursive_dict()
                custom_utilization = recursive_dict()

                if bc == "single_patch":
                    nx, nm = 15, 8
                elif bc == "multi_patch":
                    nx, nm = 23, 14

                # Iterate over placement methods
                for pp in patch_placement:
                    # Iterate over code size
                    for ks in code_size:
                        # Generate circuit
                        circuit, partitions = get_tqec_cnot_rotated(distance_scale=ks, n1=1, n2=0)
                        # Stim to qiskit
                        stim_code_circuit = StimCodeCircuit(stim_circuit=circuit)

                        # Iterate over number of defective qubits
                        for df in defective_qubits:
                            # Perform multiple iterations, since defective qubits are selected randomly
                            for run in range(0, num_iterations):
                                ic = 0
                                while True:
                                    try:
                                        backend = BackendChipletV2(
                                            size=(num_dupl * 6, num_dupl * 6, nx, nm),
                                            n_inter=num_inter_chiplet_connections,
                                            connectivity="nn",
                                            topology="rotated_grid",
                                            inter_chiplet_noise=ps_inter,
                                            inter_chiplet_amplification=1,
                                            inter_chiplet_noise_type="constant",
                                            num_defective_qubits=df,
                                            rng_seed=run + 42 + ic,
                                            sabre_defective=comp == "sabre",
                                        )

                                        # Custom transpilation
                                        if comp == "sabre":
                                            # In order to tackle defects into account, it is necessary to
                                            defective_circuit = sabre_transpilation(stim_code_circuit.qc, backend)
                                        else:
                                            defective_circuit = custom_cost_transpilation(
                                                stim_code_circuit.qc,
                                                backend,
                                                pre_defined_partitions=partitions,
                                                patch_initialization=pp,
                                            )
                                        # Compilation succeeded
                                        break
                                    except Exception as e:
                                        # Compilation failed resulting in a retry using a new seed for the RNG
                                        ic += 1
                                        print("Unable to place patches given location of defective qubits!")
                                        print("Retrying...")

                                def num_2q_gates(circuit):
                                    ops = circuit.count_ops()
                                    two_qubit_gate_names = ["cx", "cz", "swap"]
                                    return sum(ops.get(g, 0) for g in two_qubit_gate_names)

                                # Calculate qpu utilization
                                custom_utilization[pp][df][ks][run] = calculate_qpu_utilization(
                                    defective_circuit, backend
                                )

                                custom_depth[pp][df][ks][run] = (
                                    defective_circuit.depth() - (stim_code_circuit.qc).depth()
                                )
                                custom_overhead[pp][df][ks][run] = num_2q_gates(defective_circuit) - num_2q_gates(
                                    stim_code_circuit.qc
                                )

                output_dir = Path("experiments/evaluation/defective_qubits")
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_dir / f"{comp}_depth_{bc}.pkl", "wb") as f:
                    pickle.dump(custom_depth, f)
                with open(output_dir / f"{comp}_overhead_{bc}.pkl", "wb") as f:
                    pickle.dump(custom_overhead, f)
                with open(output_dir / f"{comp}_utilization_{bc}.pkl", "wb") as f:
                    pickle.dump(custom_utilization, f)

    # Generate plots
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

    plot_combined_backends(
        custom_depth_combined,
        custom_overhead_combined,
        custom_utilization_combined,
        sabre_depth_combined,
        sabre_overhead_combined,
        sabre_utilization_combined,
        filename="experiments/evaluation/defective_qubits/combined_overhead",
    )


if __name__ == "__main__":
    run_exp_defective(reproduce=True)
