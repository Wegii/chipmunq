from __future__ import annotations

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))
sys.path.append(os.path.join(os.getcwd(), "glue/eccentric_bench/"))

# Custom utils
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.circuit_generator import get_tqec_cnot_rotated
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import pickle

def plot_combined(custom_depth,
                  custom_overhead,
                  sabre_depth,
                  sabre_overhead,
                  depth_overall,
                  gate_overall,
                  filename: str = ""):
    ks = list(next(iter(custom_depth.values())).keys())[0]
    np_values = sorted(custom_depth.keys())

    section_titles = ["Small", "Medium", "Big"]

    # Extract values
    custom_depth_vals = [depth_overall[np][ks]+custom_depth[np][ks] for np in np_values]
    sabre_depth_vals = [depth_overall[np][ks]+sabre_depth[np][ks] for np in np_values]
    general_depth_vals = [depth_overall[np][ks] for np in np_values]

    custom_over_vals = [gate_overall[np][ks]+custom_overhead[np][ks] for np in np_values]
    sabre_over_vals = [gate_overall[np][ks]+sabre_overhead[np][ks] for np in np_values]
    general_over_vals = [gate_overall[np][ks] for np in np_values]

    x = np.arange(len(np_values))
    width = 0.25#0.35

    # Pastel colors
    #pastel_blue = "#5c79bd"
    #pastel_orange = "#ef8d38"
    pastel_blue = '#A7D9ED'
    pastel_orange = '#F7C6A2'

    # Create depth statistics
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.bar(x-width, general_depth_vals, width,
           label="Default", color="gray",
           hatch='//', edgecolor='black')
    
    ax.bar(x, custom_depth_vals, width,
           label="Custom", color=pastel_blue,
           hatch='/', edgecolor='black')

    ax.bar(x+width, sabre_depth_vals, width,
           label="SABRE", color=pastel_orange,
           hatch='o', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(section_titles)
    ax.set_xlabel("Circuit size")
    ax.set_ylabel("Circuit Depth Overhead")
    ax.legend()

    # Add annotation
    ax.text(-0.025, 1.05, 'Lower is better ↓',
            transform=ax.transAxes,
            fontsize=10,
            fontweight='bold',
            va='top',
            ha='left')

    fig.tight_layout()
    fig.savefig(f"{filename}_depth.png", dpi=300)
    plt.show()





    # Create 2q gate overhead
    fig, ax = plt.subplots(figsize=(5, 5))
    print(general_depth_vals)

    ax.bar(x - width, general_over_vals, width,
           label="Default", color="gray",
           hatch='/', edgecolor='black')
    
    ax.bar(x, custom_over_vals, width,
           label="Custom", color=pastel_blue,
           hatch='/', edgecolor='black')

    ax.bar(x + width, sabre_over_vals, width,
           label="SABRE", color=pastel_orange,
           hatch='o', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(section_titles)
    ax.set_ylabel("2q Gate Overhead ")
    ax.legend()

    # Add annotation
    ax.text(-0.025, 1.05, 'Lower is better ↓',
            transform=ax.transAxes,
            fontsize=10,
            fontweight='bold',
            va='top',
            ha='left')

    fig.tight_layout()
    fig.savefig(f"{filename}_overhead.png", dpi=300)
    plt.show()



def plot_combined_split(custom_depth,
                  custom_overhead,
                  sabre_depth,
                  sabre_overhead,
                  depth_overall,
                  gate_overall,
                  filename: str = ""):
    
    pastel_blue = '#A7D9ED'
    pastel_orange = '#F7C6A2'



    ks = list(next(iter(custom_depth.values())).keys())[0]
    np_values = sorted(custom_depth.keys())

    section_titles = ["Small", "Medium", "Big"]

    # Extract values
    custom_depth_vals = [depth_overall[np][ks]+custom_depth[np][ks] for np in np_values]
    sabre_depth_vals = [depth_overall[np][ks]+sabre_depth[np][ks] for np in np_values]
    general_depth_vals = [depth_overall[np][ks] for np in np_values]

    custom_over_vals = [gate_overall[np][ks]+custom_overhead[np][ks] for np in np_values]
    sabre_over_vals = [gate_overall[np][ks]+sabre_overhead[np][ks] for np in np_values]
    general_over_vals = [gate_overall[np][ks] for np in np_values]

    x = np.arange(len(np_values))
    width = 0.25#0.35


    # Create depth statistics
    fig, ax = plt.subplots(figsize=(6, 8))
    
    ax.bar(x-width, general_depth_vals, width,
           label="Default", color="lightcoral",
           hatch='//', edgecolor='black')
    
    ax.bar(x, custom_depth_vals, width,
           label="Chipmunq (Ours)", color=pastel_blue,
           hatch='/', edgecolor='black')

    ax.bar(x+width, sabre_depth_vals, width,
           label="(Light-)SABRE", color=pastel_orange,
           hatch='o', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(section_titles)
    ax.set_xlabel("Circuit size", fontsize=12)
    ax.set_ylabel("Circuit depth", fontsize=12)
    ax.legend()

    # Add annotation
    ax.text(0.7, 1.03, 'Lower is better ↓',
            transform=ax.transAxes,
            fontsize=10,
            fontweight='bold',
            color="#5c79bd",
            va='top',
            ha='left')
    


    #fig.tight_layout()
    fig.savefig(f"{filename}_depth.png", dpi=300)
    plt.show()



    # Calculate the maximum value across all bars for the upper y-limit
    max_overall_val = max(max(general_over_vals), max(custom_over_vals), max(sabre_over_vals))

    # Add a small padding to the upper y-limit for better visualization
    upper_ylim = max_overall_val*1.001

    fig = plt.figure(figsize=(6, 8)) # Adjust figure size as needed
    gs = gridspec.GridSpec(2, 1, height_ratios=[5, 20], hspace=0.1) # Swapped height_ratios

    # Top subplot (for values above the break, e.g., 60,000 to max)
    ax_top = plt.subplot(gs[0]) # Now gs[0] for the top part
    ax_top.bar(x - width, general_over_vals, width,
            label="Default", color="lightcoral",
            hatch='//', edgecolor='black')
    ax_top.bar(x, custom_over_vals, width,
            label="Chipmunq (Ours)", color=pastel_blue,
            hatch='/', edgecolor='black')
    ax_top.bar(x + width, sabre_over_vals, width,
            label="(Light-)SABRE", color=pastel_orange,
            hatch='o', edgecolor='black')


    # Limit for upper
    ax_top.set_ylim(65600, upper_ylim)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([])
    ax_top.tick_params(axis='y', length=5)

    # Bottom subplot (for values below the break, e.g., 0 to 40,000)
    ax_bottom = plt.subplot(gs[1]) # Now gs[1] for the bottom part
    ax_bottom.bar(x - width, general_over_vals, width,
                label="Default", color="lightcoral",
                hatch='/', edgecolor='black')
    ax_bottom.bar(x, custom_over_vals, width,
                label="Chipmunq (Ours)", color=pastel_blue,
                hatch='/', edgecolor='black')
    ax_bottom.bar(x + width, sabre_over_vals, width,
                label="(Light-)SABRE", color=pastel_orange,
                hatch='o', edgecolor='black')

    # Limit for bottom
    ax_bottom.set_ylim(0, 40000)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(section_titles)
    ax_bottom.tick_params(axis='y', length=5) 

    ax_bottom.spines['top'].set_visible(False)
    ax_top.spines['bottom'].set_visible(False)

    ax_top.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_bottom.tick_params(axis='x', which='both', top=False)

    # Size of horizontal lines of the split
    d = 0.015

    kwargs = dict(transform=ax_bottom.transAxes, color='k', clip_on=False, linewidth=1.5)
    # Bottom-left split
    ax_bottom.plot((-d, +d), (1 , 1), **kwargs)      
    # Bottom-right split  
    ax_bottom.plot((1 - d, 1 + d), (1, 1), **kwargs) 

    kwargs.update(transform=ax_top.transAxes)
    # Top-left split
    ax_top.plot((-d, +d), (0, 0), **kwargs)
    # Top-right split
    ax_top.plot((1 - d, 1 + d), (0, 0), **kwargs)


    fig.text(0.0, 0.5, "#2q Gates", va='center', rotation='vertical', fontsize=12)
    fig.text(0.45, 0.05, "Circuit Size", va='center', rotation='horizontal', fontsize=12)
    ax_top.text(0.7, 1.2, 'Lower is better ↓',
                transform=ax_top.transAxes,
                fontsize=10,
                fontweight='bold',
                color="#5c79bd",
                va='top',
                ha='left')

    ax_top.legend(loc='upper left')

    #fig.tight_layout()
    fig.savefig(f"{filename}_overhead.png", dpi=300)
    plt.show()
    

def run_exp_statistics():

    # Backend configuration
    num_inter_chiplet_connections = 8
    ps_inter = 1e-4

    custom_depth = {}
    custom_overhead = {}
    sabre_depth = {}
    sabre_overhead = {}
    depth_overall = {}
    gate_overall = {}
    
    n_patches = [1, 3, 6]
    """
    for ks in [2]:#[1, 2, 3, 4]
        for np in n_patches:

            if np not in custom_depth:
                custom_depth[np] = {}
                custom_overhead[np] = {}
                sabre_depth[np] = {}
                sabre_overhead[np] = {}

                depth_overall[np] = {}
                gate_overall[np] = {}
                
            # TODO: Calculate necessary backend given number of patches

            # Generate circuit
            circuit, partitions = get_tqec_cnot_rotated(distance_scale = ks,
                                                        n1 = np,
                                                        n2 = 0)

            backend = BackendChipletV2(size = (np*2, np*2, 15, 8),
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
            print("Custom")
            custom_circuit = custom_partitioned_transpilation(stim_code_circuit.qc,
                                                       backend,
                                                       pre_defined_partitions=partitions)

            # Sabre transpilation
            print("Sabre")
            sabre_circuit = sabre_transpilation(stim_code_circuit.qc, backend)

            def num_2q_gates(circuit):
                ops = circuit.count_ops()
                two_qubit_gate_names = ["cx", "cz", "swap"]
                return sum(ops.get(g, 0) for g in two_qubit_gate_names)

            custom_depth[np][ks] = custom_circuit.depth() - (stim_code_circuit.qc).depth()
            custom_overhead[np][ks] = num_2q_gates(custom_circuit) - num_2q_gates(stim_code_circuit.qc)

            sabre_depth[np][ks] = sabre_circuit.depth() - (stim_code_circuit.qc).depth()
            sabre_overhead[np][ks] = num_2q_gates(sabre_circuit) - num_2q_gates(stim_code_circuit.qc)

            depth_overall[np][ks] = (stim_code_circuit.qc).depth()
            gate_overall[np][ks] = num_2q_gates(stim_code_circuit.qc)
            

    # Save data
    with open(f"experiments/evaluation/scalability/custom_depth.pkl", "wb") as f:
        pickle.dump(custom_depth, f)
    with open(f"experiments/evaluation/scalability/custom_overhead.pkl", "wb") as f:
        pickle.dump(custom_overhead, f)
    with open(f"experiments/evaluation/scalability/sabre_depth.pkl", "wb") as f:
        pickle.dump(sabre_depth, f)
    with open(f"experiments/evaluation/scalability/sabre_overhead.pkl", "wb") as f:
        pickle.dump(sabre_overhead, f)
    with open(f"experiments/evaluation/scalability/depth_overall.pkl", "wb") as f:
        pickle.dump(depth_overall, f)
    with open(f"experiments/evaluation/scalability/gate_overall.pkl", "wb") as f:
        pickle.dump(gate_overall, f)
    """

    # Load files
    with open(f"experiments/evaluation/scalability/custom_depth.pkl", "rb") as f:
        custom_depth = pickle.load(f)
    with open(f"experiments/evaluation/scalability/custom_overhead.pkl", "rb") as f:
        custom_overhead = pickle.load(f)
    with open(f"experiments/evaluation/scalability/sabre_depth.pkl", "rb") as f:
        sabre_depth = pickle.load(f)
    with open(f"experiments/evaluation/scalability/sabre_overhead.pkl", "rb") as f:
        sabre_overhead = pickle.load(f)
    with open(f"experiments/evaluation/scalability/depth_overall.pkl", "rb") as f:
        depth_overall = pickle.load(f)
    with open(f"experiments/evaluation/scalability/gate_overall.pkl", "rb") as f:
        gate_overall = pickle.load(f)



    plot_combined(custom_depth,
                  custom_overhead,
                  sabre_depth,
                  sabre_overhead,
                  depth_overall,
                  gate_overall,
                  "experiments/evaluation/scalability/cnot_scaling_overhead")
    
    
    plot_combined_split(custom_depth,
                  custom_overhead,
                  sabre_depth,
                  sabre_overhead,
                  depth_overall,
                  gate_overall,
                  "experiments/evaluation/scalability/cnot_scaling_overhead_split")

            
if __name__ == "__main__":
    run_exp_statistics()