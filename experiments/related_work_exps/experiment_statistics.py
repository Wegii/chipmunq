from __future__ import annotations

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))
sys.path.append(os.path.join(os.getcwd(), "glue/eccentric_bench/"))

import time
import pickle

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

# Plotting
from experiments.utils import *
from experiments.related_work_exps.utils import *
from matplotlib.ticker import MaxNLocator


def plot_combined(mech_overhead, qeccsynth_overhead, filename: str = ""):

    # Calculate number of two-qubit gates
    mech_2q_overhead = [mech_overhead[d]["on-chip"] + mech_overhead[d]["cross-chip"] for d in sorted(mech_overhead.keys())]
    qeccsynth_2q_overhead = [qeccsynth_overhead[d]["on-chip"] + qeccsynth_overhead[d]["cross-chip"] for d in sorted(qeccsynth_overhead.keys())]

    fig, ax = plt.subplots(figsize=(WIDTH_FIGSIZE*1.2, HEIGHT_FIGSIZE*1.7))

    distances = sorted(qeccsynth_overhead.keys())
    x_val = [2*x+1 for x in distances]

    #y_mech = [mech_2q_overhead[di] for di in distances]
    plt.plot(x_val, mech_2q_overhead, marker='x', linestyle='-', label=f"MECH", color = "#3B6FA8")

    #y_qeccsynth = [qeccsynth_2q_overhead[di] for di in distances]
    plt.plot(x_val, qeccsynth_2q_overhead, marker='o', linestyle='--', label=f"qecc_synth", color = "#9F3B36")

    ax.text(
        0.73, 1.02, "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    plt.tick_params(axis='both', labelsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xlabel("Distance", fontsize=16)
    plt.ylabel("#2q gates", fontsize=16)
    plt.yscale("log")

    #plt.grid(True)
    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right')
    fig.subplots_adjust(left=0.15, right=0.95, top=0.93, bottom=0.15)
    plt.savefig(filename,
                format="pdf")
    plt.close(fig)



def run_statistics():
    code_distances = [2, 3, 4, 5] # range(2, 5)
    qeccsynth_overhead_storage = {}
    mech_overhead_storage = {}

    for d in code_distances:
        cycles = d
        code = get_surface_code_stim(d, cycles)

        # TODO: calculate chiplet size based on distance
        n = m = int(d*1.5)

        monolithic_backend, qubit_num, data_qubit_num = generate_simple_backend(n, m)
        architecture = generate_qecc_synth_backend_from_mech(monolithic_backend)

        # Print backend to file
        #display_simple_backend(monolithic_backend, f"experiments/evaluation/related_work/backends/monolithic_{n}_{m}.png")


        # MECH
        circuit_mech = transpile_circuit_MECH(code.qc, monolithic_backend)
        result_mech = calc_circuit_mech_stats(circuit_mech)

        # QECCsynth
        circuit_qeccsynth = transpile_circuit_QECCSynth(d, architecture, f'square_{n}_{m}_{m}')
        result_qeccsynth = calc_circuit_qiskit_stats(circuit_qeccsynth, monolithic_backend)

        qeccsynth_overhead_storage[d] = result_qeccsynth
        mech_overhead_storage[d] = result_mech

    # Write results to file
    with open(f"experiments/evaluation/related_work/overhead_mech.pkl", "wb") as f:
        pickle.dump(mech_overhead_storage, f)

    with open(f"experiments/evaluation/related_work/overhead_qeccsynth.pkl", "wb") as f:
        pickle.dump(qeccsynth_overhead_storage, f)
    
    
    
if __name__ == "__main__":
    # run_statistics()


    # Load pre-computed results
    with open(f"experiments/evaluation/related_work/overhead_mech.pkl", "rb") as f:
        mech_overhead_storage = pickle.load(f)
    with open(f"experiments/evaluation/related_work/overhead_qeccsynth.pkl", "rb") as f:
        qeccsynth_overhead_storage = pickle.load(f)

    plot_combined(mech_overhead_storage,
                  qeccsynth_overhead_storage,
                  "experiments/evaluation/related_work/memory_overhead.pdf")
    