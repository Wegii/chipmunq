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

# SABRE
sys.path.append(os.path.join(os.getcwd(), "./external/baseline/SABRE"))
from external.baseline.SABRE.transpile_sabre import transpile_circuit_SABRE

# Plotting
from experiments.utils import *
from experiments.related_work_exps.utils import *
from matplotlib.ticker import MaxNLocator


def plot_runtime(mech_overhead, qeccsynth_overhead, qiskit_overhead, filename: str = ""):

    distances = [2, 5]

    # Timeout-value for qecc-synth
    qeccsynth_overhead[("chiplet", 11)] = 1e4
    
    mech_2q_overhead = ([mech_overhead[("mono", d)] for d in distances] + 
                        [mech_overhead[("chiplet", d)] for d in distances])
    qeccsynth_2q_overhead = ([qeccsynth_overhead[("mono", d)]for d in distances] + 
                             [qeccsynth_overhead[("chiplet", d)]for d in distances])
    qiskit_2q_overhead = ([qiskit_overhead[("mono", d)] for d in distances] + 
                          [qiskit_overhead[("chiplet", d)] for d in distances])

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
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.5, WIDTH_FIGSIZE*0.5))

    x_val = [2*x+1 for x in distances] + [2*x+1 for x in distances]

    section_titles = x_val 

    x = np.arange(len(section_titles))
    width = 0.25

    ax.bar(x-width, qiskit_2q_overhead, width,
           label="LightSABRE", color="lightcoral",
           hatch='o', edgecolor='black')

    ax.bar(x, mech_2q_overhead, width,
           label="MECH", color="#A7D9ED",
           hatch='//', edgecolor='black')
    
    bars_qeccsynth = ax.bar(x+width, qeccsynth_2q_overhead, width,
                            label="QECC-Synth", color="#B2D8B2",
                            hatch='/', edgecolor='black')
    
    # Adjust styling of last bar for qecc-synth to show that it timed out
    timeout_bar = bars_qeccsynth[-1] 
    timeout_bar.set_facecolor('white')      
    timeout_bar.set_edgecolor('#B2D8B2')        
    timeout_bar.set_linestyle('--')        
    timeout_bar.set_hatch('xxx')           
    timeout_bar.set_linewidth(2)
    ax.text(x[-1] + width, 1200, "T/O", ha='center', va='bottom', 
        color='black', fontweight='bold', fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(section_titles)

    # Add annotation

    title = "a) Effect of distance on compilation time"
    ax.text(
        -0.06, 1.02, title,
        transform=ax.transAxes,
        fontweight="bold"
    )

    ax.text(
        0.3, 1.1, "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    plt.tick_params(axis='both', labelsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.text(.5, -0.21, "Monolithic", 
        transform=ax.get_xaxis_transform(),
        ha='center',
        fontsize = (FONTSIZE - 1)*1.5)
    
    ax.text(2.5, -0.21, "Chiplet", 
        transform=ax.get_xaxis_transform(),
        ha='center',
        fontsize = (FONTSIZE - 1)*1.5)

    ax.text(1.5, -0.29, "Surface code distance", 
            transform=ax.get_xaxis_transform(),
            ha='center',
            fontsize = (FONTSIZE - 1)*1.5)
    #plt.xlabel("Surface Code Distance", fontsize=16)

    description = "Runtime [s]"
    plt.ylabel(description,)
    plt.yscale("log")
    ax.set_ylim(0, 5e3)

    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    # ax.legend(loc='upper left')

    fig.subplots_adjust(left=0.175, right=0.95, top=0.87, bottom=0.2)
    plt.savefig(filename,
                format="pdf")
    plt.close(fig)



def run_runtime_scaling():
    code_distances = [2, 5] # [2, 3, 4, 5]
    backend = ["mono", "chiplet"]
    
    qeccsynth_time_storage = {}
    mech_time_storage = {}
    sabre_time_storage = {}

    for b in backend:
        for d in code_distances:
            cycles = d
            code = get_surface_code_stim(d, cycles)

            # TODO: calculate chiplet size based on distance
            n = m = int(d*1.5)
            if b == "chiplet":
                n_icc = 1
            else:
                n_icc = None

            monolithic_backend, _, _ = generate_simple_backend(n, m, n_icc)
            architecture = generate_qecc_synth_backend_from_mech(monolithic_backend)
            cm = generate_qiskit_backend_from_mech(monolithic_backend)

            # Print backend to file
            display_simple_backend(monolithic_backend, f"experiments/evaluation/related_work/backends/{b}_{n}_{m}_{n_icc}.png")
                
            # MECH
            start_mech = time.time()
            _ = transpile_circuit_MECH(code.qc, monolithic_backend)
            end_mech = time.time()

            # QECCsynth
            if not(b == "chiplet" and d == 5):
                start_qeccsynth = time.time()
                _ = transpile_circuit_QECCSynth(d, architecture, f'square_{n}_{m}_{m}')
                end_qeccsynth = time.time()

            # Qiskit
            start_sabre = time.time()
            _ = transpile_circuit_SABRE(circuit = code.qc, coupling_map = cm)
            end_sabre = time.time()

            if not(b == "chiplet" and d == 5):
                qeccsynth_time_storage[(b, d)] = end_qeccsynth - start_qeccsynth
            else:
                qeccsynth_time_storage[(b, d)] = 1e3
            mech_time_storage[(b, d)] = end_mech - start_mech
            sabre_time_storage[(b, d)] = end_sabre - start_sabre

    # Write results to file
    with open(f"experiments/evaluation/related_work/timing_mech.pkl", "wb") as f:
        pickle.dump(mech_time_storage, f)

    with open(f"experiments/evaluation/related_work/timing_qeccsynth.pkl", "wb") as f:
        pickle.dump(qeccsynth_time_storage, f)

    with open(f"experiments/evaluation/related_work/timing_sabre.pkl", "wb") as f:
        pickle.dump(sabre_time_storage, f)
    
    
    
if __name__ == "__main__":
    #run_runtime_scaling()

    
    # Load pre-computed results
    with open(f"experiments/evaluation/related_work/timing_mech.pkl", "rb") as f:
        mech_time_storage = pickle.load(f)
    with open(f"experiments/evaluation/related_work/timing_qeccsynth.pkl", "rb") as f:
        qeccsynth_time_storage = pickle.load(f)
    with open(f"experiments/evaluation/related_work/timing_sabre.pkl", "rb") as f:
        sabre_time_storage = pickle.load(f)

    plot_runtime(mech_time_storage,
                  qeccsynth_time_storage,
                  sabre_time_storage,
                  "experiments/evaluation/related_work/memory_scaling.pdf")
    
    