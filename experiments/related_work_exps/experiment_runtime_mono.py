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
import time
from pathlib import Path


def plot_runtime(mech_overhead, qeccsynth_overhead, qiskit_overhead, filename: str = "") -> None:
    # Code distance of surface code
    distances = [2, 3, 4, 5, 6, 7]

    # Timeout-value for qecc-synth

    mech_2q_overhead = [mech_overhead[("mono", d)] for d in distances]
    qeccsynth_2q_overhead = [qeccsynth_overhead[("mono", d)] for d in distances]
    qiskit_2q_overhead = [qiskit_overhead[("mono", d)] for d in distances]

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
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE * 2.5, WIDTH_FIGSIZE * 0.5))

    x_val = [2 * x + 1 for x in distances]

    section_titles = x_val

    x = np.arange(len(section_titles))
    width = 0.25

    ax.bar(x - width, qiskit_2q_overhead, width, label="LightSABRE", color="lightcoral", hatch="o", edgecolor="black")

    ax.bar(x, mech_2q_overhead, width, label="MECH", color="#A7D9ED", hatch="//", edgecolor="black")

    bars_qeccsynth = ax.bar(
        x + width, qeccsynth_2q_overhead, width, label="QECC-Synth", color="#B2D8B2", hatch="/", edgecolor="black"
    )

    # Adjust styling of last bar for qecc-synth to show that it timed out
    tb = [bars_qeccsynth[-1], bars_qeccsynth[-2]]
    for timeout_bar in tb:
        timeout_bar.set_facecolor("white")
        # timeout_bar.set_edgecolor('#B2D8B2')
        # timeout_bar.set_facecolor('white')
        timeout_bar.set_edgecolor("red")
        timeout_bar.set_linestyle("--")
        timeout_bar.set_hatch("xxx")
        timeout_bar.set_linewidth(2)
        timeout_bar.set_path_effects([])
        timeout_bar.set_edgecolor("red")
        timeout_bar.set_edgecolor("#B2D8B2")

    ax.text(x[-1] + width, 1200, "T/O", ha="center", va="bottom", color="red", fontweight="bold", fontsize=12)
    ax.text(x[-2] + width, 1200, "T/O", ha="center", va="bottom", color="red", fontweight="bold", fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(section_titles)

    # Add annotation
    title = "a) Effect of distance on compilation time"
    ax.text(-0.06, 1.02, title, transform=ax.transAxes, fontweight="bold")

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

    plt.xlabel("Surface code distance", fontsize=(FONTSIZE - 1) * 1.5)

    description = "Runtime [s]"
    plt.ylabel(
        description,
    )
    plt.yscale("log")
    ax.set_ylim(0, 5e3)

    plt.grid(True, which="major", linestyle="--", alpha=0.5)
    # ax.legend(loc='upper left')

    fig.subplots_adjust(left=0.175, right=0.95, top=0.83, bottom=0.2)

    plt.savefig(filename, format="pdf")
    plt.close(fig)


def run_runtime_scaling(reproduce: bool = False) -> None:
    # Code distance of surface code
    code_distances = [2, 3, 4, 5, 6, 7]
    # Configuration of chiplet backend. 'mono' creates monolythic chip
    backend = ["mono"]

    if reproduce:
        qeccsynth_time_storage = {}
        mech_time_storage = {}
        sabre_time_storage = {}

        for b in backend:
            for d in code_distances:
                cycles = d
                code = get_surface_code_stim(d, cycles)

                # TODO: calculate chiplet size based on distance
                n = m = int(d * 1.5)
                if b == "chiplet":
                    n_icc = 1
                else:
                    n_icc = None

                monolithic_backend, _, _ = generate_simple_backend(n, m, n_icc)
                architecture = generate_qecc_synth_backend_from_mech(monolithic_backend)
                cm = generate_qiskit_backend_from_mech(monolithic_backend)

                # Print backend to file
                display_simple_backend(
                    monolithic_backend, f"experiments/evaluation/related_work/backends/{b}_{n}_{m}_{n_icc}.png"
                )

                # MECH
                start_mech = time.time()
                _ = transpile_circuit_MECH(code.qc, monolithic_backend)
                end_mech = time.time()

                # QECCsynth
                if d <= 5:
                    start_qeccsynth = time.time()
                    _ = transpile_circuit_QECCSynth(d, architecture, f"square_{n}_{m}_{m}")
                    end_qeccsynth = time.time()

                # Qiskit
                start_sabre = time.time()
                _ = transpile_circuit_SABRE(circuit=code.qc, coupling_map=cm)
                end_sabre = time.time()

                if d <= 5:
                    qeccsynth_time_storage[(b, d)] = end_qeccsynth - start_qeccsynth
                else:
                    qeccsynth_time_storage[(b, d)] = 1e3
                mech_time_storage[(b, d)] = end_mech - start_mech
                sabre_time_storage[(b, d)] = end_sabre - start_sabre

        # Save results to file
        output_dir = Path("experiments/evaluation/related_work")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "timing_mech_mono.pkl", "wb") as f:
            pickle.dump(mech_time_storage, f)
        with open(output_dir / "timing_qeccsynth_mono.pkl", "wb") as f:
            pickle.dump(qeccsynth_time_storage, f)
        with open(output_dir / "timing_sabre_mono.pkl", "wb") as f:
            pickle.dump(sabre_time_storage, f)

    # Load pre-computed results
    with open("experiments/evaluation/related_work/timing_mech_mono.pkl", "rb") as f:
        mech_time_storage = pickle.load(f)
    with open("experiments/evaluation/related_work/timing_qeccsynth_mono.pkl", "rb") as f:
        qeccsynth_time_storage = pickle.load(f)
    with open("experiments/evaluation/related_work/timing_sabre_mono.pkl", "rb") as f:
        sabre_time_storage = pickle.load(f)

    # Plot runtime scaling for various code distances
    plot_runtime(
        mech_time_storage,
        qeccsynth_time_storage,
        sabre_time_storage,
        "experiments/evaluation/related_work/memory_scaling_mono.pdf",
    )


if __name__ == "__main__":
    run_runtime_scaling(reproduce=True)
