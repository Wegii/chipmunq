from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.getcwd(), "."))

# MECH
sys.path.append(os.path.join(os.getcwd(), "./external/baseline/MECH"))
from external.baseline.MECH.Circuit import *
from external.baseline.MECH.Chiplet import *
from external.baseline.MECH.HighwayOccupancy import *
from external.baseline.MECH.Router import *
from external.baseline.MECH.MECHBenchmarks import *
from external.baseline.MECH.transpile_mech import transpile_circuit_MECH
import networkx as nx
from networkx.classes import Graph

# QECC-Synth
sys.path.append(os.path.join(os.getcwd(), "./external/baseline/QECC_Synth/SurfStitch/MyCode/src"))
from external.baseline.QECC_Synth.SurfStitch.MyCode.src.transpile_qeccsynth import transpile_circuit_QECCSynth

# SABRE
sys.path.append(os.path.join(os.getcwd(), "./external/baseline/SABRE"))
from external.baseline.SABRE.transpile_sabre import transpile_circuit_SABRE

# Plotting
from matplotlib.ticker import MaxNLocator
from experiments.related_work_exps.utils import *
from experiments.exp_utils.utils import *
import pickle
from pathlib import Path


def plot_gate_overhead(mech_overhead, qeccsynth_overhead, qiskit_overhead, type: str, filename: str = "") -> None:

    if type == "gate_overhead":
        mech_2q_overhead = [mech_overhead[d]["2q_gates_overhead"] for d in sorted(mech_overhead.keys())]
        qeccsynth_2q_overhead = [
            1 + qeccsynth_overhead[d]["2q_gates_overhead"] for d in sorted(qeccsynth_overhead.keys())
        ]
        qiskit_2q_overhead = [1 + qiskit_overhead[d]["2q_gates_overhead"] for d in sorted(qiskit_overhead.keys())]
    elif type == "inter_chiplet":
        mech_2q_overhead = [mech_overhead[d]["cross-chip"] for d in sorted(mech_overhead.keys())]
        qeccsynth_2q_overhead = [1 + qeccsynth_overhead[d]["cross-chip"] for d in sorted(qeccsynth_overhead.keys())]
        qiskit_2q_overhead = [qiskit_overhead[d]["cross-chip"] for d in sorted(qiskit_overhead.keys())]

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
    # fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.5, WIDTH_FIGSIZE*0.92))
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE * 2.5, WIDTH_FIGSIZE * 0.5))

    distances = sorted(qeccsynth_overhead.keys())
    x_val = [2 * x + 1 for x in distances]

    section_titles = x_val

    x = np.arange(len(section_titles))
    width = 0.25

    qiskit_bar = ax.bar(
        x - width, qiskit_2q_overhead, width, label="LightSABRE", color="lightcoral", hatch="o", edgecolor="black"
    )

    mech_bar = ax.bar(x, mech_2q_overhead, width, label="MECH", color="#A7D9ED", hatch="//", edgecolor="black")

    qecc_synth_bar = ax.bar(
        x + width, qeccsynth_2q_overhead, width, label="QECC-Synth", color="#B2D8B2", hatch="/", edgecolor="black"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(section_titles)

    # Add annotation
    if type == "gate_overhead":
        title = "b) Effect of distance on #2q gate overhead"
        shift = -0.09
    else:
        title = "c) Effect of distance on #inter-chiplet gates"
        shift = -0.12
    ax.text(shift, 1.02, title, transform=ax.transAxes, fontweight="bold")

    ax.text(
        0.3,
        1.15,
        "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    plt.tick_params(axis="both", labelsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xlabel("Surface code distance", fontsize=16)
    if type == "gate_overhead":
        description = "#2q gate overhead"
    else:
        description = "#inter-chiplet gates"
    plt.ylabel(description, fontsize=16)
    plt.yscale("log")

    plt.grid(True, which="major", linestyle="--", alpha=0.5)

    # ax.legend(loc='upper left')

    fig.subplots_adjust(left=0.175, right=0.95, top=0.83, bottom=0.18)

    plt.savefig(filename + ".pdf", format="pdf")
    plt.close(fig)

    # Create custom figure for legend
    legend_fig = plt.figure(figsize=(3, 2))
    legend = legend_fig.legend(handles=[qiskit_bar, mech_bar, qecc_synth_bar], loc="center", frameon=False, ncols=3)
    legend_fig.savefig(filename + "_legend.pdf", bbox_inches="tight", format="pdf")
    plt.close(legend_fig)


def run_statistics(reproduce: bool = False) -> None:
    # Code distance of surface code
    code_distances = [2, 3, 4, 5]

    if reproduce:
        # Dictionary for storing circuit statistics
        qeccsynth_overhead_storage = {}
        mech_overhead_storage = {}
        qiskit_overhead_storage = {}

        for d in code_distances:
            cycles = d
            code = get_surface_code_stim(d, cycles)

            # TODO: calculate chiplet size based on distance
            n = m = int(d * 1.5)

            monolithic_backend, qubit_num, data_qubit_num = generate_simple_backend(n, m)
            architecture = generate_qecc_synth_backend_from_mech(monolithic_backend)
            cm = generate_qiskit_backend_from_mech(monolithic_backend)

            # Print backend to file
            # display_simple_backend(monolithic_backend, f"experiments/evaluation/related_work/backends/monolithic_{n}_{m}.png")

            # MECH
            circuit_mech = transpile_circuit_MECH(code.qc, monolithic_backend)
            result_mech = calc_circuit_mech_stats(circuit_mech, code.qc)

            # QECCsynth
            circuit_qeccsynth, result = transpile_circuit_QECCSynth(d, architecture, f"square_{n}_{m}_{m}")
            result_qeccsynth = calc_circuit_qiskit_stats(circuit_qeccsynth, monolithic_backend, result)

            # Qiskit
            circuit_qiskit = transpile_circuit_SABRE(circuit=code.qc, coupling_map=cm)
            result_qiskit = calc_circuit_qiskit_stats(circuit_qiskit, monolithic_backend, initial_circuit=code.qc)

            qeccsynth_overhead_storage[d] = result_qeccsynth
            mech_overhead_storage[d] = result_mech
            qiskit_overhead_storage[d] = result_qiskit

        # Write results to file
        output_dir = Path("experiments/evaluation/related_work")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "overhead_mech.pkl", "wb") as f:
            pickle.dump(mech_overhead_storage, f)
        with open(output_dir / "overhead_qeccsynth.pkl", "wb") as f:
            pickle.dump(qeccsynth_overhead_storage, f)
        with open(output_dir / "overhead_sabre.pkl", "wb") as f:
            pickle.dump(qiskit_overhead_storage, f)

    # Load pre-computed results
    with open("experiments/evaluation/related_work/overhead_mech.pkl", "rb") as f:
        mech_overhead_storage = pickle.load(f)
    with open("experiments/evaluation/related_work/overhead_qeccsynth.pkl", "rb") as f:
        qeccsynth_overhead_storage = pickle.load(f)
    with open("experiments/evaluation/related_work/overhead_sabre.pkl", "rb") as f:
        qiskit_overhead_storage = pickle.load(f)

    # Overhead of gates
    plot_gate_overhead(
        mech_overhead_storage,
        qeccsynth_overhead_storage,
        qiskit_overhead_storage,
        "gate_overhead",
        "experiments/evaluation/related_work/memory_overhead",
    )

    # Utilization of inter-chiplet links
    plot_gate_overhead(
        mech_overhead_storage,
        qeccsynth_overhead_storage,
        qiskit_overhead_storage,
        "inter_chiplet",
        "experiments/evaluation/related_work/memory_inter_chiplet",
    )


if __name__ == "__main__":
    run_statistics(reproduce=True)
