from __future__ import annotations

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))
sys.path.append(os.path.join(os.getcwd(), "glue/eccentric_bench/"))

# Custom utils
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.circuit_generator import get_tqec_cnot_rotated
from experiments.exp_utils.simulation_utils import *
from glue.qiskit_qec.stim_tools import get_stim_circuits_with_detectors
from stim import Circuit as StimCircuit
from glue.eccentric_bench.noise import get_noise_model
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
from experiments.utils import *


def plot_evaluation(stat, filename, inter_chiplet_noise):
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

    d_values = sorted(d_values)
    physical_error_rates = sorted(list(physical_error_rates))

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

    #fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.6, WIDTH_FIGSIZE))
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.5, WIDTH_FIGSIZE*0.5))
    handles = []
    # Plot identity (x = y)
    h = plt.plot(physical_error_rates, physical_error_rates, linestyle="--", linewidth=1.5, color="#000000B3", label=f'x=y')
    handles.extend(h)

    colors_transpiled = ([ "#5E97CC", "#3B6FA8", "#2A5687"])
    colors_default = ([ "#C85E59", "#9F3B36", "#7F2E2A"])
    color_list = [colors_default, colors_transpiled]

    inter_markers = ['x', 'o', 's']
    #plot_label = ["Basic, Low Variance", "Basic, High Variance", "Cost, Low Variance", "Cost, High Variance", "Tradeoff, High Variance", "Tradeoff, High Variance"]
    #for ti, t in enumerate(["basic10", "basic100", "cost_inter10", "cost_inter100", "cost_tradeoff10", "cost_tradeoff100"]):
    plot_label = ["Basic, Low Variance",  "Cost, Low Variance",  "Tradeoff, Low Variance", "Basic, High Variance", "Cost, High Variance", "Tradeoff, High Variance"]
    for ti, t in enumerate(["basic10", "cost_inter10", "cost_tradeoff10", "basic100", "cost_inter100", "cost_tradeoff100"]):
        for i, d in enumerate(d_values):
            errors = defaultdict(dict)

            for p in physical_error_rates:
                errors[p] = error_rates[t][d][p][0]
                
            ys_custom = [errors[p] for p in physical_error_rates]

            line_color = ("#2A5687" if (t in ["basic10", "cost_inter10", "cost_tradeoff10"])
                          else "#7F2E2A")
            line_style = ("-" if (t in ["basic10", "basic100"])
                          else "--")
            if t in ["cost_inter10", "cost_inter100"]:
                marker = "x"
            elif t in ["cost_tradeoff10", "cost_tradeoff100"]:
                marker = "o"
            else:
                marker = ""
            h = plt.plot(physical_error_rates,
                        ys_custom,
                        #linewidth = 1.5,
                        marker = marker,
                        #markersize = 4,
                        #markerfacecolor="none",
                        linestyle= line_style,
                        color = line_color,
                        label = plot_label[ti])
            handles.extend(h)
            
    if inter_chiplet_noise == 0.0001:
        ps_inter_text = r"$1e^{-4}$"
    elif inter_chiplet_noise == 0.001:
        ps_inter_text = r"$1e^{-3}$"
    elif inter_chiplet_noise == 0.01:
        ps_inter_text = r"$1e^{-2}$"
    description = (r"$p_{inter}$ = " + f"{ps_inter_text}, d = 5")

    ax.text(
        .05, 1.02, "a) Effect of cost_routing on LER",
        transform=ax.transAxes,
        fontweight="bold"
    )

    ax.text(
        0.3, 1.13, "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )
    
    #plt.ylim(-0.01, 0.9)
    plt.ylim(1e-6, 1e0)
    plt.xlim(1e-4, 1e-2)
    plt.xscale('log')
    plt.yscale('log')


    plt.xlabel("Physical error rate")
    plt.ylabel("Logical error rate")
    #plt.legend(loc="lower right", ncol=1)
    #plt.grid(True, which='both', linestyle='--', alpha=0.5)
    #fig.subplots_adjust(left=0.16, right=0.97, top=0.89, bottom=0.13)
    fig.subplots_adjust(left=0.22, right=0.95, top=0.85, bottom=0.21)
    plt.savefig(filename, format="pdf")
    plt.close(fig)

    legend_fig = plt.figure(figsize=(3, 2))
    legend = legend_fig.legend(handles = handles,
                               loc = 'center',
                               frameon = False,
                               ncols = 3)
    legend_fig.savefig(filename + 'legend.pdf', bbox_inches='tight', format="pdf")
    plt.close(legend_fig)


def plot_error_improvement(stats, filename, inter_chiplet_noise, alpha, beta):
    # Difference between basic routing and cost routing (two options there)

    # Calculate error rate and group
    error_rates = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    physical_error_rates = set()
    d_values = set()
    for s in stats:
        ler = s.errors / (s.shots - s.discards)
        p = s.json_metadata['p']
        t = s.json_metadata['run_name']
        d = s.json_metadata['d']

        error_rates[t][d][p].append(ler)
        physical_error_rates.add(p)
        d_values.add(d)

    physical_error_rates = sorted(physical_error_rates)

    diff_low_cost = defaultdict(dict)
    diff_high_cost = defaultdict(dict)
    diff_low_cost_tradeoff = defaultdict(dict)
    diff_high_cost_tradeoff = defaultdict(dict)

    for d in d_values:
        for p in physical_error_rates:
            diff_low_cost[d][p] = error_rates['cost_inter10'][d][p][0] - error_rates['basic10'][d][p][0]
            diff_low_cost_tradeoff[d][p] = error_rates['cost_tradeoff10'][d][p][0]  - error_rates['basic10'][d][p][0]

            diff_high_cost[d][p] = error_rates['cost_inter100'][d][p][0] - error_rates['basic100'][d][p][0]
            diff_high_cost_tradeoff[d][p] = error_rates['cost_tradeoff100'][d][p][0] - error_rates['basic100'][d][p][0]


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

    #fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.6*1.5, WIDTH_FIGSIZE/1.5))
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.5, WIDTH_FIGSIZE*0.5))

    ax.axhline(0.0, color='black', linestyle='--', linewidth=2, alpha=0.3)

    ps_rates = sorted(diff_low_cost[5].keys())
    ys_custom = [diff_low_cost[5][p] for p in physical_error_rates]
    plt.plot(ps_rates, ys_custom, marker='x', color='#2A5687', linestyle='--', label=f'Cost, Low Variance')

    ps_rates = sorted(diff_low_cost_tradeoff[5].keys())
    ys_custom = [diff_low_cost_tradeoff[5][p] for p in physical_error_rates]
    plt.plot(ps_rates, ys_custom, marker='o', color='#2A5687', linestyle='--', label=f'Tradeoff, Low Variance')

    ps_rates = sorted(diff_high_cost[5].keys())
    ys_custom = [diff_high_cost[5][p] for p in physical_error_rates]
    plt.plot(ps_rates, ys_custom, marker='x', color="#7F2E2A", linestyle='--', label=f'Cost, High Variance')

    ps_rates = sorted(diff_high_cost_tradeoff[5].keys())
    ys_custom = [diff_high_cost_tradeoff[5][p] for p in physical_error_rates]
    plt.plot(ps_rates, ys_custom, marker='o', color='#7F2E2A', linestyle='--', label=f'Tradeoff, High Variance')

    description = (r"$p_{inter}$ = " +
                   f"{inter_chiplet_noise}, " +
                   r"$\alpha_{cost} = $" + str(3*alpha) +
                   r", $\alpha_{tradeoff} = $" + str(alpha) +
                   r", $\beta = $" + str(beta)
                   )

    #ax.text(
    #    0, 1.02, description,
    #    transform=ax.transAxes,
    #    #fontweight="bold"
    #)

    ax.text(
        -.1, 1.02, "b) Relative effect of cost_routing on LER",
        transform=ax.transAxes,
        fontweight="bold"
    )

    ax.text(
        0.3, 1.13, "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )
        
    
    plt.ylim(-10e-1, 0.5)
    plt.xlim(1e-4, 1e-2)
    plt.xscale('log')
    #plt.yscale('log')
    plt.yscale('symlog', linthresh=1e-3)  # linear within ±0.001

    plt.xlabel("Physical Error Rate")
    #plt.ylabel(r"Δ($LER_{Routing Method} - LER_{Basic}$)")
    plt.ylabel(r"Δ$LER_{Routing}$")
    #plt.legend(loc="lower right", ncol=2)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    #fig.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.2)
    fig.subplots_adjust(left=0.22, right=0.95, top=0.85, bottom=0.21)
    
    plt.savefig(filename, format="pdf")
    plt.close(fig)


def perform_noise_aware_routing():
    # Noise strength:
    #   - Low noise: 1e-4
    #   - Medium noise: 1e-3
    # Setup:
    #   - Distributed CNOT
    #   - Distance: 5
    #   - Full connectivity

    # Minimize depth
    # TODO: alpha = beta = 0

    # Cost routing completely focusing on taking the best inter-chiplet connection
    # TODO: alpha should be quite high

    # Tradeoff between inter-chiplet noise and depth
    # TODO: figure out what is a good value for alpha and beta

    routing_alpha = [0.5, 1, 2, 3]

    # Noise level
    ps = list(np.logspace(-4, -1, 20))

    # Inter-chiplet noise level
    inter_chiplet_noise = [1e-4, 1e-3, 1e-2]

    # Distance 5
    k = 2

    # Routing types
    rts = ["basic", "cost_inter", "cost_tradeoff"]
    
    # Inter chiplet noise variance
    #   - Low variance: [1, 10]*inter_connect_noise
    #   - High variance: [1, 100]*inter_connect_noise
    ic_noise_model = [10, 100]#[10, 100]#[5, 10]#[5, 10]

    circuit, partitions = get_tqec_cnot_rotated(distance_scale = k,
                                        n1 = 1,
                                        n2 = 0)

    for ra in routing_alpha:
        for ps_inter in inter_chiplet_noise:

            transpiled_circuits = {}
            
            def get_circuit(routing_type: str, inter_noise_factor: int, distance_scale: int) -> StimCircuit:

                if (routing_type, inter_noise_factor) in transpiled_circuits:
                    # Circuit does not need to be transpiled again
                    print("Found")
                    return transpiled_circuits[(routing_type, inter_noise_factor)]
                else:
                    n_icc, backend = get_backend(inter_noise_factor = inter_noise_factor,
                                            d = distance_scale)
                    
                    # Transpile circuit to backend
                    if routing_type == "cost_inter":
                        routing_type_u = "cost"
                        routing_alpha = 3*ra*ps_inter #3*ps_inter
                        routing_beta = 1
                    elif routing_type == "cost_tradeoff":
                        routing_type_u = "cost"
                        routing_alpha = ra*ps_inter #1*ps_inter
                        routing_beta = 1
                    else:
                        routing_type_u = "cost"
                        routing_alpha = 0
                        routing_beta = 0

                    _, custom_circuit, _, _ = transpile_stim_circuit(circuit,
                                                                    backend,
                                                                    pre_defined_partitions = partitions,
                                                                    routing_type = routing_type_u,
                                                                    routing_alpha = routing_alpha,
                                                                    routing_beta = routing_beta)
                    # Convert circuit to stim
                    custom_circuit_stim = get_stim_circuits_with_detectors(custom_circuit)[0][0]
                    # Add circuit to dictionary, in order to not transpile this circuit configuration again
                    transpiled_circuits[(routing_type, inter_noise_factor)] = custom_circuit_stim
                    #transpiled_circuits[(routing_type)] = custom_circuit_stim

                    return custom_circuit_stim

            def get_backend(inter_noise_factor: int, d: int) -> BackendChipletV2:
                
                # Depending on the distance, each chiplet needs to be scaled
                if d == 1:
                    chiplet_size = (2, 2, 11, 6)
                    nic = 5
                elif d == 2:
                    chiplet_size = (2, 2, 15, 8)
                    nic = 7
                elif d == 3:
                    chiplet_size = (2, 2, 19, 10)
                    nic = 9
                elif d == 4:
                    chiplet_size = (2, 2, 23, 12)
                    nic = 11

                backend = BackendChipletV2(size = chiplet_size,
                                            n_inter = nic,
                                            connectivity = "nn",
                                            topology = "rotated_grid",
                                            inter_chiplet_noise = ps_inter,
                                            inter_chiplet_amplification = 1,
                                            inter_chiplet_rfactor = inter_noise_factor,
                                            inter_chiplet_noise_type = "random",
                                            num_defective_qubits = 0,
                                            )

                return nic, backend

            def _get_sinter_task():
                # Construct sinter task for multiple code distances and noise levels
                yield from (
                    sinter.Task(
                        circuit=circuit,
                        # TODO: the naming is incorrect
                        json_metadata={"d": 2 * k + 1, "p": p, "run_name": rt + str(icnm),},
                    )
                    for circuit, k, p, rt, icnm in (
                        (get_noise_model("modsi1000",
                                        None,
                                        p,
                                        None, 
                                        remote = (get_backend(inter_noise_factor = icnm,
                                                            d = k)[1].inter_chiplet_connections)
                                        ).noisy_circuit(
                                            get_circuit(routing_type = rt,
                                                        inter_noise_factor = icnm,
                                                        distance_scale = k)
                                            ), k, p, rt, icnm)

                        for p in ps
                        for icnm in ic_noise_model
                        for rt in rts
                    )
                )
            
            stats = run_sinter_simulation(_get_sinter_task, [k], ps)

            with open(f"experiments/evaluation/qec_routing/routing_{ra}_{ps_inter}_sweep.pkl", "wb") as f:
                pickle.dump(stats, f)
        

if __name__ == "__main__":
    # perform_noise_aware_routing()

    for ra in [0.5, 1, 2, 3]:
        for ps in [0.0001, 0.001, 0.01]:

            if ps == 0.0001:
                ps_inter_text = r"$1e^{-4}$"
            elif ps == 0.001:
                ps_inter_text = r"$1e^{-3}$"
            elif ps == 0.01:
                ps_inter_text = r"$1e^{-2}$"

            with open(f"experiments/evaluation/qec_routing/routing_{ra}_{ps}_sweep.pkl", "rb") as f:
                stats = pickle.load(f)

            plot_evaluation(stats, f"experiments/evaluation/qec_routing/routing_{ra}_{ps}.pdf", ps)

            plot_error_improvement(stats,
                                   f"experiments/evaluation/qec_routing/routing_difference_{ra}_{ps}.pdf",
                                   ps_inter_text,
                                   ra,
                                   1)
