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


def plot_combined(mech_runtime, qeccsynth_runtime, filename: str = ""):
    colors_qeccsynth = [ "#8FB7E1", "#5E97CC", "#3B6FA8"]
    colors_mech = [ "#E38E8A", "#C85E59", "#9F3B36"]

    fig, ax = plt.subplots(figsize=(WIDTH_FIGSIZE*1.2, HEIGHT_FIGSIZE*1.7))


    distances = sorted(mech_runtime.keys())
    x_val = [2*x+1 for x in distances]

    y_mech = [mech_runtime[di] for di in distances]
    plt.plot(x_val, y_mech, marker='x', linestyle='-', label=f"MECH", color = "#3B6FA8")

    y_qeccsynth = [qeccsynth_runtime[di] for di in distances]
    plt.plot(x_val, y_qeccsynth, marker='o', linestyle='--', label=f"qecc_synth", color = "#9F3B36")

    ax.text(
        0.73, 1.02, "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    plt.tick_params(axis='both', labelsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xlabel("Distance", fontsize=16)
    plt.ylabel("Runtime [s]", fontsize=16)
    plt.yscale("log")

    #plt.grid(True)
    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right')
    fig.subplots_adjust(left=0.15, right=0.95, top=0.93, bottom=0.15)
    plt.savefig(filename,
                format="pdf")
    plt.close(fig)



def run_runtime_scaling():
    code_distances = [2, 3, 4, 5] # range(2, 5)
    qeccsynth_time_storage = {}
    mech_time_storage = {}

    for d in code_distances:
        cycles = d
        code = get_surface_code_stim(d, cycles)

        # TODO: calculate chiplet size based on distance
        n = m = int(d*1.5)

        monolithic_backend, qubit_num, data_qubit_num = generate_simple_backend(n, m)
        architecture = generate_qecc_synth_backend_from_mech(monolithic_backend)

        # Print backend to file
        display_simple_backend(monolithic_backend, f"experiments/evaluation/related_work/backends/monolithic_{n}_{m}.png")
            
        # MECH
        start_mech = time.time()
        _ = transpile_circuit_MECH(code.qc, monolithic_backend)
        end_mech = time.time()

        # QECCsynth
        start_qeccsynth = time.time()
        _ = transpile_circuit_QECCSynth(d, architecture, f'square_{n}_{m}_{m}')
        end_qeccsynth = time.time()


        qeccsynth_time_storage[d] = end_qeccsynth - start_qeccsynth
        mech_time_storage[d] = end_mech - start_mech

    # Write results to file
    with open(f"experiments/evaluation/related_work/timing_mech.pkl", "wb") as f:
        pickle.dump(mech_time_storage, f)

    with open(f"experiments/evaluation/related_work/timing_qeccsynth.pkl", "wb") as f:
        pickle.dump(qeccsynth_time_storage, f)
    
    
    
if __name__ == "__main__":
    # run_runtime_scaling()


    # Load pre-computed results
    with open(f"experiments/evaluation/related_work/timing_mech.pkl", "rb") as f:
        mech_time_storage = pickle.load(f)
    with open(f"experiments/evaluation/related_work/timing_qeccsynth.pkl", "rb") as f:
        qeccsynth_time_storage = pickle.load(f)

    plot_combined(mech_time_storage,
                  qeccsynth_time_storage,
                  "experiments/evaluation/related_work/memory_scaling.pdf")
    