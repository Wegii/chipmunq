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
from matplotlib.ticker import MaxNLocator
from pathlib import Path


# Panel geometry, identical to experiment_statistics (panels b - d): four panels across the full text width
TEXT_WIDTH_IN = 7.0
PANEL_W = TEXT_WIDTH_IN / 4           # 1.75 in
PANEL_H = 1.6
FONT_PT = 7
PANEL_MARGINS = dict(left=0.27, right=0.97, top=0.80, bottom=0.24)

def plot_combined(custom_time_storage, sabre_time_storage, filename: str = ""):
    # Extract sorted x values
    np_values = sorted(custom_time_storage.keys())
    x_val = [x for x in np_values]

    # Gather all ks values
    ks_values = sorted({ks for d in custom_time_storage.values() for ks in d.keys()})

    # Panel a) of the statistics row: drawn at its printed size like panels b) - d) of experiment_statistics
    # (1.75 x 1.6 in, 7 pt, same margins), so the four panels line up; include at natural width, no scaling.
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": FONT_PT,
        "axes.labelsize": FONT_PT,
        "axes.titlesize": FONT_PT,
        "legend.fontsize": FONT_PT,
        "xtick.labelsize": FONT_PT - 1,
        "ytick.labelsize": FONT_PT - 1,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "ytick.minor.size": 1.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.labelpad": 2,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "lines.markeredgewidth": 0.6,
        "lines.markeredgecolor": "black",
        "errorbar.capsize": 1.5,
    })

    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))

    colors_custom = ["#8FB7E1", "#5E97CC", "#3B6FA8", "#2D5682"]
    colors_sabre = ["#E38E8A", "#C85E59", "#9F3B36", "#6D2926"]

    inter_markers = ["x", "o", "s", "^"]

    handles = []
    for i, ks in enumerate(ks_values):
        # SABRE
        y_sabre = [sabre_time_storage[np].get(ks, None) for np in np_values]
        # Values above 1e3 are removed due to timeout
        y_sabre = [y for y in y_sabre if y < 1e3]
        h = plt.plot(
            x_val[: len(y_sabre)],
            y_sabre,
            marker=inter_markers[i],
            linestyle="--",
            label=f"LightSABRE, d={2 * ks + 1}",
            color=colors_sabre[i],
        )
        handles.extend(h)

        # for i, ks in enumerate(ks_values):
        # Custom
        y_custom = [custom_time_storage[np].get(ks, None) for np in np_values]
        h = plt.plot(
            x_val,
            y_custom,
            marker=inter_markers[i],
            linestyle="-",
            label=f"Chipmunq, d={2 * ks + 1}",
            color=colors_custom[i],
        )
        handles.extend(h)

    # plt.axvline(x=3)

    # Title and "better" hint centred on the plot rectangle, as in panels b) - d)
    ax.text(0.5, 1.03, "a) Runtime scaling", transform=ax.transAxes, fontweight="bold", ha="center", va="bottom")
    ax.text(0.5, 1.16, "Lower is better ↓", transform=ax.transAxes, fontweight="bold", color=plot_lib_color,
            ha="center", va="bottom")

    # description = ("routing_method = basic")
    # ax.text(
    #    0, 1.02, description,
    #    transform=ax.transAxes,
    #    fontweight="bold"
    # )

    plt.ylim(0.05, 1e4)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Number of CNOTs")
    plt.ylabel("Runtime [s]")
    plt.yscale("log")

    # plt.grid(True)
    plt.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
    # ax.legend(loc='lower right', ncol=2)
    # plt.tight_layout()
    fig.subplots_adjust(**PANEL_MARGINS)
    plt.savefig(filename, format="pdf")
    plt.close(fig)

    legend_fig = plt.figure(figsize=(TEXT_WIDTH_IN, 0.4))
    legend = legend_fig.legend(handles=handles, loc="center", frameon=False, ncols=4, columnspacing=1.0,
                               handlelength=1.8, handletextpad=0.4)
    legend_fig.savefig(filename + "legend.pdf", bbox_inches="tight", format="pdf")
    plt.close(legend_fig)


def calculate_speedup(custom_time_storage, sabre_time_storage):
    # Extract sorted x values
    np_values = sorted(custom_time_storage.keys())

    # Gather all ks values
    ks_values = sorted({ks for d in custom_time_storage.values() for ks in d.keys()})

    all_speedup = []

    for i, ks in enumerate(ks_values):
        sabre_values = [sabre_time_storage[np].get(ks, None) for np in np_values]
        custom_values = [custom_time_storage[np].get(ks, None) for np in np_values]

        # Only account runtime values below the runtime threshold of 1e3
        sabre_clean = np.array([y for y in sabre_values if y < 1e3])
        custom_reduced = np.array(custom_values[0 : len(sabre_clean)])

        speedup = sabre_clean / custom_reduced
        all_speedup.extend(speedup)

        print(f"d={2 * ks + 1}: {np.mean(speedup)}")
        # print(np.std(speedup))

    print(f"All speedups: {all_speedup}")
    print(f"Average speedup: {np.mean(np.array(all_speedup))}")


def run_exp_scalability(reproduce: bool = False) -> None:

    # Backend configuration
    ps_inter = 1e-4

    custom_time_storage = {}
    sabre_time_storage = {}

    # Number of logical CNOTs constructed using lattice surgery
    n_patches = [1, 2, 4, 6, 8, 16]

    # Code distance of surface code
    code_distance = [1, 2, 3, 7]

    if reproduce:
        for ks in code_distance:
            for num_p in n_patches:
                if num_p not in custom_time_storage:
                    custom_time_storage[num_p] = {}
                    sabre_time_storage[num_p] = {}

                # Generate circuit
                print("Generating circuit")
                circuit, partitions = get_tqec_cnot_rotated(distance_scale=ks, n1=num_p, n2=0)

                # Calculate required number of chiplets given the number of patches that we want to place
                bx = num_p
                by = num_p

                bx, by = 2 * (num_p + 1), 2 * (num_p + 1)

                if ks == 1:
                    chiplet_size = (bx, by, 11, 6)
                    nic = 5
                elif ks == 2:
                    chiplet_size = (bx, by, 15, 8)
                    nic = 7
                elif ks == 3:
                    chiplet_size = (bx, by, 19, 10)
                    nic = 9
                elif ks == 4:
                    chiplet_size = (bx, by, 23, 12)
                    nic = 11
                elif ks == 5:
                    chiplet_size = (bx, by, 27, 14)
                    nic = 13
                elif ks == 6:
                    chiplet_size = (bx, by, 31, 16)
                    nic = 15
                elif ks == 7:
                    chiplet_size = (bx, by, 35, 18)
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

                # Stim to qiskit
                stim_code_circuit = StimCodeCircuit(stim_circuit=circuit)

                # Custom transpilation
                print("Custom")
                start_custom = time.time()
                _ = custom_partitioned_transpilation(stim_code_circuit.qc, backend, pre_defined_partitions=partitions)
                end_custom = time.time()
                print("Custom done")

                # Sabre transpilation
                print("Sabre")
                # Only run SABRE for certain configurations, as it otherwise times-out
                sabre_run = ks <= 3 or (ks > 3 and num_p < 3)
                start_sabre = time.time()
                if sabre_run:
                    _ = sabre_transpilation(stim_code_circuit.qc, backend)
                end_sabre = time.time()
                print("SABRE done")

                t_dur_custom = end_custom - start_custom
                if not sabre_run:
                    t_dur_sabre = 1e4
                else:
                    t_dur_sabre = end_sabre - start_sabre

                custom_time_storage[num_p][ks] = t_dur_custom
                sabre_time_storage[num_p][ks] = t_dur_sabre

        # Write results to file
        output_dir = Path("experiments/evaluation/scalability")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "timing_custom.pkl", "wb") as f:
            pickle.dump(custom_time_storage, f)

        with open(output_dir / "timing_sabre.pkl", "wb") as f:
            pickle.dump(sabre_time_storage, f)

    # Load pre-computed results
    with open("experiments/evaluation/scalability/timing_custom.pkl", "rb") as f:
        custom_time_storage = pickle.load(f)
    with open("experiments/evaluation/scalability/timing_sabre.pkl", "rb") as f:
        sabre_time_storage = pickle.load(f)

    print("Custom:")
    print(custom_time_storage)
    print("Sabre")
    print(sabre_time_storage)

    plot_combined(custom_time_storage, sabre_time_storage, "experiments/evaluation/scalability/cnot_scaling.pdf")

    calculate_speedup(custom_time_storage, sabre_time_storage)


def run_single_run():
    ps_inter = 1e-4

    num_p = 8  # [1, 2, 4, 6, 8]#range(1, 8, 2)#10, 2)
    ks = 3

    # ks=2, num_p=8,
    # Original: 17.47
    # No routing:

    print("Generating circuit")
    circuit, partitions = get_tqec_cnot_rotated(distance_scale=ks, n1=num_p, n2=0)

    # Calculate required number of chiplets given the number of patches that we want to place
    bx = num_p
    by = num_p
    if num_p <= 4:
        bx, by = 4, 4
    if num_p == 6:
        bx, by = 6, 4
    if num_p == 8:
        bx, by = 8, 4

    bx, by = 2 * (num_p + 1), 2 * (num_p + 1)

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

    # Stim to qiskit
    stim_code_circuit = StimCodeCircuit(stim_circuit=circuit)

    # Custom transpilation
    print("Custom")
    start_custom = time.time()
    custom_circuit = custom_partitioned_transpilation(stim_code_circuit.qc, backend, pre_defined_partitions=partitions)
    end_custom = time.time()
    print("Custom done")

    # Sabre transpilation
    print("Sabre")
    start_sabre = time.time()
    # sabre_circuit = sabre_transpilation(stim_code_circuit.qc, backend)
    end_sabre = time.time()
    print("SABRE done")

    t_dur_custom = end_custom - start_custom
    t_dur_sabre = end_sabre - start_sabre

    print("Custom:")
    print(t_dur_custom)
    print("Sabre")
    print(t_dur_sabre)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Panel a): Chipmunq vs. LightSABRE runtime for #CNOTs and distances")
    ap.add_argument("--plot-only", action="store_true", help="only replot the saved timings (no compilation)")
    a = ap.parse_args()
    run_exp_scalability(reproduce=not a.plot_only)

    # run_single_run()