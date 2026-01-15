# How is the logical error rate influenced by a limited number of inter-chiplet connections?

from __future__ import annotations

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))
sys.path.append(os.path.join(os.getcwd(), "glue/eccentric_bench/"))

# Custom utils
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.circuit_generator import QECMemory, QECCircuit, get_tqec_cnot_rotated
from qeccm.backends.backend_utils import plot_circuit_layout, plot_circuit_layout_utilization
from qeccm.src.reference_partitions import memory_d5
from experiments.exp_utils.simulation_utils import *

from experiments.exp_utils.circuit_utils import stim_to_qiskit
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit
from glue.qiskit_qec.stim_tools import get_stim_circuits_with_detectors
from stim import Circuit as StimCircuit

from glue.eccentric_bench.noise import get_noise_model

from tqec.utils.noise_model import NoiseModel
from tqec.computation.block_graph import BlockGraph
from tqec.utils.enums import Basis
from tqec.utils.position import Position3D
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle


def plot_evaluation(stats, filename, inter_chiplet_noise, num_inter):

    error_rates = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    physical_error_rates = set()
    d_values = set()
    for s in stats:
        ler = s.errors / (s.shots - s.discards)
        p = s.json_metadata['p']
        t = str(s.json_metadata['run_name'])
        d = str(s.json_metadata['d'])
        
        error_rates[t][d][p].append(ler)
        physical_error_rates.add(p)
        d_values.add(d)

    physical_error_rates = sorted(list(physical_error_rates))

    fig, ax = plt.subplots(figsize=(6, 5))
    # Plot identity (x = y)
    plt.plot(physical_error_rates, physical_error_rates, linestyle="--", linewidth=1.5, color="#000000B3", label=f'x=y')

    colors_5 = (["#B7D1EC", "#8FB7E1", "#5E97CC", "#3B6FA8", "#2A5687"])
    colors_7 = (["#F0B3B0", "#E38E8A", "#C85E59", "#9F3B36", "#7F2E2A"])

    inter_markers = ['', 'x', 'o', 's','^']
    color_list = [colors_5, colors_7]

    for i, d in enumerate(d_values):
        errors = defaultdict(dict)
        for ni, n in enumerate(num_inter[::-1]):
            for p in physical_error_rates:
                errors[p] = error_rates[str(n)][d][p][0]

            ys_custom = [errors[p] for p in physical_error_rates]
            plt.plot(physical_error_rates,
                     ys_custom,
                     linewidth = 1,
                     marker = inter_markers[ni],
                     markersize = 3,
                     markerfacecolor="none",
                     linestyle= "solid" if d == "5" else "solid",
                     color = color_list[i][ni],
                     label = f'({d}, {n})')
            

    if inter_chiplet_noise == 0.0001:
        ps_inter_text = r"$1e^{-4}$"
    elif inter_chiplet_noise == 0.001:
        ps_inter_text = r"$1e^{-3}$"
    elif inter_chiplet_noise == 0.01:
        ps_inter_text = r"$1e^{-2}$"
    description = (r"$p_{inter}$ = " + f"{ps_inter_text}")

    ax.text(
        0, 1.02, description,
        transform=ax.transAxes,
        fontsize=9,
        #fontweight="bold"
    )

    ax.text(
        0.75, 1.02, "Lower is better ↓",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color="#5c79bd",
    )
    
    #plt.ylim(-0.01, 0.9)
    plt.ylim(5e-8, 1e0)
    plt.xscale('log')
    plt.yscale('log')


    plt.xlabel("Physical error rate")
    plt.ylabel("Logical error rate")
    plt.legend(loc="lower right", ncol=2)
    #plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


def plot_difference(stats, filename, inter_chiplet_noise):
    # TODO: Plot difference

    # TODO: Get run with run_name = 8

    # Difference between maximum inter chiplet connections and limited ones

    # Calculate error rate and group
    error_rates = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    physical_error_rates = set()
    d_values = set()
    for s in stats:
        ler = s.errors / (s.shots - s.discards)
        p = s.json_metadata['p']
        t = str(s.json_metadata['run_name'])
        d = str(s.json_metadata['d'])
        
        error_rates[t][d][p].append(ler)
        physical_error_rates.add(p)
        d_values.add(d)

    physical_error_rates = sorted(list(physical_error_rates)) #sorted(physical_error_rates)
    #print(physical_error_rates)

    colors_5 = (["#B7D1EC", "#8FB7E1", "#5E97CC", "#3B6FA8", "#2A5687"])
    colors_7 = (["#F0B3B0", "#E38E8A", "#C85E59", "#9F3B36", "#7F2E2A"])

    diff_1 = defaultdict(dict)
    diff_2 = defaultdict(dict)
    diff_4 = defaultdict(dict)
    diff_6 = defaultdict(dict)

    for d in d_values:
        for p in physical_error_rates:
            diff_1[d][p] = (error_rates['1'][d][p][0]/error_rates['8'][d][p][0])#(error_rates['1'][d][p][0] / error_rates['8'][d][p][0])
            diff_2[d][p] = (error_rates['2'][d][p][0] / error_rates['8'][d][p][0])
            diff_4[d][p] = (error_rates['4'][d][p][0] / error_rates['8'][d][p][0])
            diff_6[d][p] = (error_rates['6'][d][p][0] / error_rates['8'][d][p][0])

    fig, ax = plt.subplots(figsize=(8, 3)) 

    ys_custom = [diff_1['5'][p] for p in physical_error_rates]
    plt.plot(physical_error_rates, ys_custom, marker='x', markerfacecolor="none", linewidth=1.5, color=colors_5[0], label=f'(5, 1)')

    ys_custom = [diff_2['5'][p] for p in physical_error_rates]
    plt.plot(physical_error_rates, ys_custom, marker='o', markerfacecolor="none", linewidth=1.5, color=colors_5[1], label=f'(5, 2)')

    ys_custom = [diff_4['5'][p] for p in physical_error_rates]
    plt.plot(physical_error_rates, ys_custom, marker='s', markerfacecolor="none", linewidth=1.5, color=colors_5[2], label=f'(5, 4)')

    ys_custom = [diff_6['5'][p] for p in physical_error_rates]
    plt.plot(physical_error_rates, ys_custom, marker='^', markerfacecolor="none", linewidth=1.5, color=colors_5[3], label=f'(5, 6)')


    ys_custom = [diff_1['7'][p] for p in physical_error_rates]
    plt.plot(physical_error_rates, ys_custom, marker='x', markerfacecolor="none", linewidth=1.5, color=colors_7[0], label=f'(7, 1)')

    ys_custom = [diff_2['7'][p] for p in physical_error_rates]
    plt.plot(physical_error_rates, ys_custom, marker='o', markerfacecolor="none", linewidth=1.5, color=colors_7[1], label=f'(7, 2)')

    ys_custom = [diff_4['7'][p] for p in physical_error_rates]
    plt.plot(physical_error_rates, ys_custom, marker='s', markerfacecolor="none", linewidth=1.5, color=colors_7[2], label=f'(7, 4)')

    ys_custom = [diff_6['7'][p] for p in physical_error_rates]
    plt.plot(physical_error_rates, ys_custom, marker='^', markerfacecolor="none", linewidth=1.5, color=colors_7[3], label=f'(7, 6)')


    if inter_chiplet_noise == 0.0001:
        ps_inter_text = r"$1e^{-4}$"
    elif inter_chiplet_noise == 0.001:
        ps_inter_text = r"$1e^{-3}$"
    elif inter_chiplet_noise == 0.01:
        ps_inter_text = r"$1e^{-2}$"
    description = (r"$p_{inter}$ = " + f"{ps_inter_text}")

    ax.text(
        0, 1.02, description,
        transform=ax.transAxes,
        fontsize=9,
        #fontweight="bold"
    )

    ax.text(
        0.81, 1.02, "Lower is better ↓",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color="#5c79bd",
    )
    
    plt.ylim(0.9, 170)
    #plt.xlim(1e-4, 1e-1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Physical Error Rate")
    #plt.ylabel(r"Δ($LER_{Reduced} - LER_{Full}$)")
    plt.ylabel(r"$LER_{Reduced} / LER_{Full}$")
    plt.legend(loc="lower right")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()



def run_exp_distributed_inter_chiplet() -> None:

    # Number of inter_chiplet_connections
    num_inter_chiplet_connections = [8, 6, 4, 2, 1]
    
    # Noise level
    ps = list(np.logspace(-4, -1, 10)) #list(np.logspace(-4, -1, 10))

    # Inter-chiplet noise level
    inter_chiplet_noise = [1e-4, 1e-3, 1e-2]
    
    for ps_inter in inter_chiplet_noise:
        
        # Transpilation
        ts = [str(i) for i in num_inter_chiplet_connections]
        ks = [2, 3]
        routing_types = ["default"] #["cost", "default"]

        transpiled_circuits = {}
        """
        def get_circuit(n_icc, p_icc, amp_icc, t: str , routing_type: str, k: int = 1) -> StimCircuit:
            if (n_icc, p_icc, amp_icc, routing_type, k) in transpiled_circuits:
                # Circuit does not need to be transpiled again
                print("Found")
                return transpiled_circuits[(n_icc, p_icc, amp_icc, routing_type, k)]
            else:
                circuit, partitions = get_tqec_cnot_rotated(distance_scale = k,
                                        n1 = 1,
                                        n2 = 0)
                
                backend = get_backend(n_icc = n_icc,
                                        p_icc = p_icc,
                                        amp_icc = amp_icc,
                                        k = k)

                # Transpile circuit to backend
                _, custom_circuit, _, _ = transpile_stim_circuit(circuit,
                                                                backend,
                                                                pre_defined_partitions = partitions,
                                                                routing_type = routing_type,#"cost",
                                                                routing_alpha = 1.0,
                                                                routing_beta = 1.0)
                # Convert circuit to stim
                custom_circuit_stim = get_stim_circuits_with_detectors(custom_circuit)[0][0]
                # Add circuit to dictionary, in order to not transpile this circuit configuration again
                transpiled_circuits[(n_icc, p_icc, amp_icc, routing_type, k)] = custom_circuit_stim

                plot_circuit_layout(custom_circuit,
                                    backend,
                                    filename=f"experiments/evaluation/inter_chiplet/backend_mapping/layout_{n_icc}_{k}.png")
                
                plot_circuit_layout_utilization(custom_circuit,
                                                backend,
                                                filename=f"experiments/evaluation/inter_chiplet/backend_mapping/mapping_{n_icc}_{k}.png")


                return custom_circuit_stim

        def get_backend(n_icc: int, p_icc: float, amp_icc: float, t: str = "", k: int = 1) -> BackendChipletV2:
            if t == "default":
                return None
            else:
                if k == 1:
                    chiplet_size = (6, 6, 11, 6)
                elif k == 2:
                    chiplet_size = (6, 6, 15, 8)
                elif k == 3:
                    chiplet_size = (6, 6, 19, 10)
            
                return BackendChipletV2(size = chiplet_size,#(2, 2, 15, 8),
                                        n_inter = n_icc,
                                        connectivity = "nn",
                                        topology = "rotated_grid",
                                        inter_chiplet_noise = p_icc,
                                        inter_chiplet_amplification = amp_icc,
                                        inter_chiplet_noise_type = "constant"
                                        )

        def _get_sinter_task():
            # Construct sinter task for multiple code distances and noise levels
            yield from (
                sinter.Task(
                    circuit=circuit,
                    json_metadata={"d": 2 * k + 1, "r": 2 * k + 1, "p": p, "run_name": t, "p_inter": p_icc},
                )
                for circuit, k, p, t, rt, p_icc in (
                    (get_noise_model("modsi1000",
                                    None,
                                    p,
                                    None, 
                                    remote = (None if t == "default" else
                                            get_backend(int(t), ps_inter, 1, t, k).inter_chiplet_connections)
                                    ).noisy_circuit(
                                        get_circuit(-1 if t == "default" else int(t), ps_inter, 1, t, routing_type=rt, k = k)
                                        ), k, p, t, rt, ps_inter)
                    
                    for t in ts
                    for rt in routing_types
                    for k in ks
                    for p in ps
                )
            )

        stats = run_sinter_simulation(_get_sinter_task, ks, ps)

        with open(f"experiments/evaluation/inter_chiplet/inter_chiplet_{ps_inter}_sweep.pkl", "wb") as f:
            pickle.dump(stats, f)
        
        """
        with open(f"experiments/evaluation/inter_chiplet/inter_chiplet_{ps_inter}_sweep.pkl", "rb") as f:
            stats = pickle.load(f)

        plot_evaluation(stats,
                        filename = f"experiments/evaluation/inter_chiplet/inter_chiplet_{ps_inter}.png",
                        inter_chiplet_noise = ps_inter,
                        num_inter = num_inter_chiplet_connections)
        
        plot_difference(stats,
                        filename = f"experiments/evaluation/inter_chiplet/inter_chiplet_{ps_inter}_difference.png",
                        inter_chiplet_noise = ps_inter)
        
        
        
    #plot_interconnect_sweep(stats,
    #                        filename = f"experiments/evaluation/chiplet_evaluation/single_cnot_rotated_inter_chiplet_sweep.png")


if __name__ == "__main__":
    run_exp_distributed_inter_chiplet()