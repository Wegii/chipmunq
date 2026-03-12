from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.getcwd(), "."))

from stim import Circuit as StimCircuit
from experiments.exp_utils.circuit_generator import get_tqec_cnot_rotated
from experiments.exp_utils.simulation_utils import *
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.utils import *
from qeccm.backends.backend_utils import plot_circuit_layout_utilization, plot_circuit_layout
from experiments.exp_utils.circuit_noise import get_noise_model

# Plotting
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_evaluation(stats, filename, inter_chiplet_noise, num_inter):

    error_rates = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    physical_error_rates = set()
    d_values = set()
    for s in stats:
        ler = s.errors / (s.shots - s.discards)
        p = s.json_metadata["p"]
        t = str(s.json_metadata["run_name"])
        d = str(s.json_metadata["d"])

        error_rates[t][d][p].append(ler)
        physical_error_rates.add(p)
        d_values.add(d)

    physical_error_rates = sorted(list(physical_error_rates))

    tex_fonts = {
        # Use LaTeX to write all text
        # "text.usetex": True,
        "font.family": "serif",
        # Font sizes
        "axes.labelsize": FONTSIZE * 1.5,
        "font.size": FONTSIZE * 1.2,
        "legend.fontsize": (FONTSIZE - 2) * 1.3,
        "xtick.labelsize": (FONTSIZE - 1) * 1.3,
        "ytick.labelsize": (FONTSIZE - 1) * 1.3,
        "axes.titlesize": 10,
        # Line and marker styles
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "lines.markeredgewidth": 1.5,
        "lines.markeredgecolor": "black",
        # Error bar cap size
        "errorbar.capsize": 3,
    }

    plt.rcParams.update(tex_fonts)

    # fig, ax = plt.subplots(figsize=(6, 5))
    # fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.6, WIDTH_FIGSIZE))
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE * 2.5, WIDTH_FIGSIZE * 0.5))
    handles = []
    # Plot identity (x = y)
    h = plt.plot(
        physical_error_rates, physical_error_rates, linestyle="--", linewidth=1.5, color="#000000B3", label="x=y"
    )
    # handles.extend(h)

    colors_5 = ["#B7D1EC", "#8FB7E1", "#5E97CC", "#3B6FA8", "#2A5687"]
    colors_7 = ["#B7D1EC", "#8FB7E1", "#5E97CC", "#3B6FA8", "#2A5687"]
    # colors_7 = (["#F0B3B0", "#E38E8A", "#C85E59", "#9F3B36", "#7F2E2A"])

    inter_markers = ["", "x", "o", "s", "^"]
    color_list = [colors_7, colors_5]

    d_values = ["7"]

    for i, d in enumerate(d_values):
        errors = defaultdict(dict)
        for ni, n in enumerate(num_inter[::-1]):
            for p in physical_error_rates:
                errors[p] = error_rates[str(n)][d][p][0]

            ys_custom = [errors[p] for p in physical_error_rates]
            h = plt.plot(
                physical_error_rates,
                ys_custom,
                marker=inter_markers[ni],
                linestyle="solid" if d == "5" else "solid",
                color=color_list[i][ni],
                label=r"$n_{inter}$" + f" = {n}",
            )
            handles.extend(h)

    if inter_chiplet_noise == 0.0001:
        ps_inter_text = r"$1e^{-4}$"
    elif inter_chiplet_noise == 0.001:
        ps_inter_text = r"$1e^{-3}$"
    elif inter_chiplet_noise == 0.01:
        ps_inter_text = r"$1e^{-2}$"
    description = r"$p_{inter}$ = " + f"{ps_inter_text}"

    # ax.text(
    #    0, 1.02, description,
    #    transform=ax.transAxes,
    #    #fontsize=9,
    #    fontweight="bold"
    # )
    ax.text(0.05, 1.02, "b) Effect of connectivity on LER", transform=ax.transAxes, fontweight="bold")

    ax.text(
        0.3,
        1.13,
        "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    # plt.ylim(-0.01, 0.9)
    plt.ylim(5e-8, 2e0)
    plt.xlim(1e-4, 1e-2)
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Physical error rate")
    plt.ylabel("LER")
    # plt.legend(loc="lower right", ncol=2)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.21)

    plt.savefig(filename, format="pdf")
    plt.close(fig)

    legend_fig = plt.figure(figsize=(3, 2))
    legend = legend_fig.legend(handles=handles[0:3], loc="center", frameon=False, ncols=5, columnspacing=1.5)
    legend_fig.savefig(filename + "legend1.pdf", bbox_inches="tight", format="pdf")
    plt.close(legend_fig)

    legend_fig = plt.figure(figsize=(3, 2))
    legend = legend_fig.legend(handles=handles[3:], loc="center", frameon=False, ncols=5, columnspacing=1.5)
    legend_fig.savefig(filename + "legend2.pdf", bbox_inches="tight", format="pdf")
    plt.close(legend_fig)


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
        p = s.json_metadata["p"]
        t = str(s.json_metadata["run_name"])
        d = str(s.json_metadata["d"])

        error_rates[t][d][p].append(ler)
        physical_error_rates.add(p)
        d_values.add(d)

    physical_error_rates = sorted(list(physical_error_rates))  # sorted(physical_error_rates)
    # print(physical_error_rates)

    colors_5 = ["#B7D1EC", "#8FB7E1", "#5E97CC", "#3B6FA8", "#2A5687"]
    colors_7 = ["#B7D1EC", "#8FB7E1", "#5E97CC", "#3B6FA8", "#2A5687"]
    # colors_7 = (["#F0B3B0", "#E38E8A", "#C85E59", "#9F3B36", "#7F2E2A"])

    diff_1 = defaultdict(dict)
    diff_2 = defaultdict(dict)
    diff_4 = defaultdict(dict)
    diff_6 = defaultdict(dict)

    for d in d_values:
        for p in physical_error_rates:
            diff_1[d][p] = (
                error_rates["1"][d][p][0] / error_rates["8"][d][p][0]
            )  # (error_rates['1'][d][p][0] / error_rates['8'][d][p][0])
            diff_2[d][p] = error_rates["2"][d][p][0] / error_rates["8"][d][p][0]
            diff_4[d][p] = error_rates["4"][d][p][0] / error_rates["8"][d][p][0]
            diff_6[d][p] = error_rates["6"][d][p][0] / error_rates["8"][d][p][0]

    tex_fonts = {
        # Use LaTeX to write all text
        # "text.usetex": True,
        "font.family": "serif",
        # Font sizes
        "axes.labelsize": FONTSIZE * 1.5,
        "font.size": FONTSIZE * 1.2,
        "legend.fontsize": (FONTSIZE - 2) * 1.3,
        "xtick.labelsize": (FONTSIZE - 1) * 1.3,
        "ytick.labelsize": (FONTSIZE - 1) * 1.3,
        "axes.titlesize": 10,
        # Line and marker styles
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "lines.markeredgewidth": 1.5,
        "lines.markeredgecolor": "black",
        # Error bar cap size
        "errorbar.capsize": 3,
    }

    plt.rcParams.update(tex_fonts)

    # fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*3, WIDTH_FIGSIZE/1.5))
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE * 2.5, WIDTH_FIGSIZE * 0.5))

    """
    ys_custom = [diff_1['5'][p] for p in physical_error_rates]
    plt.plot(physical_error_rates, ys_custom, marker='x', color=colors_5[0], label=r'(d=5, $n_{inter}$=1)')

    ys_custom = [diff_2['5'][p] for p in physical_error_rates]
    plt.plot(physical_error_rates, ys_custom, marker='o', color=colors_5[1], label=r'(d=5, $n_{inter}$=2)')

    ys_custom = [diff_4['5'][p] for p in physical_error_rates]
    plt.plot(physical_error_rates, ys_custom, marker='s', color=colors_5[2], label=r'(d=5, $n_{inter}$=4)')

    ys_custom = [diff_6['5'][p] for p in physical_error_rates]
    plt.plot(physical_error_rates, ys_custom, marker='^', color=colors_5[3], label=r'(d=5, $n_{inter}$=6)')
    """

    ys_custom = [diff_1["7"][p] for p in physical_error_rates]
    plt.plot(physical_error_rates, ys_custom, marker="", color=colors_7[0], label=r"d=7, $n_{inter}$=1")

    ys_custom = [diff_2["7"][p] for p in physical_error_rates]
    plt.plot(physical_error_rates, ys_custom, marker="x", color=colors_7[1], label=r"d=7, $n_{inter}$=2")

    ys_custom = [diff_4["7"][p] for p in physical_error_rates]
    plt.plot(physical_error_rates, ys_custom, marker="o", color=colors_7[2], label=r"d=7, $n_{inter}$=4")

    ys_custom = [diff_6["7"][p] for p in physical_error_rates]
    plt.plot(physical_error_rates, ys_custom, marker="s", color=colors_7[3], label=r"d=7, $n_{inter}$=6")

    if inter_chiplet_noise == 0.0001:
        ps_inter_text = r"$1e^{-4}$"
    elif inter_chiplet_noise == 0.001:
        ps_inter_text = r"$1e^{-3}$"
    elif inter_chiplet_noise == 0.01:
        ps_inter_text = r"$1e^{-2}$"
    description = r"$p_{inter}$ = " + f"{ps_inter_text}"

    ax.text(-0.05, 1.02, "c) Relative effect of connectivity on LER", transform=ax.transAxes, fontweight="bold")

    ax.text(
        0.3,
        1.13,
        "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    plt.ylim(0.9, 130)
    # plt.xlim(1e-4, 1e-1)
    plt.xlim(1e-4, 1e-2)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Physical error rate")
    # plt.ylabel(r"Δ($LER_{Reduced} - LER_{Full}$)")
    plt.ylabel(r"$LER_{Reduced} / LER_{Full}$")
    # plt.legend(loc = "upper right", ncol = 2)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.21)

    plt.savefig(filename, format="pdf")
    plt.close(fig)


def run_exp_distributed_inter_chiplet(reproduce: bool = False) -> None:

    # Number of inter_chiplet_connections
    num_inter_chiplet_connections = [8, 6, 4, 2, 1]

    # Physical noise level
    ps = list(np.logspace(-4, -1, 10))

    # Inter-chiplet noise level
    inter_chiplet_noise = [1e-3]  # [1e-4, 1e-3, 1e-2]

    # Transpilation
    ts = [str(i) for i in num_inter_chiplet_connections]

    # Code size of surface code
    ks = [3]

    # Routing methods
    routing_types = ["default"]  # ["cost", "default"]

    for ps_inter in inter_chiplet_noise:
        if reproduce:
            transpiled_circuits = {}

            def get_circuit(n_icc, p_icc, amp_icc, t: str, routing_type: str, k: int = 1) -> StimCircuit:
                if (n_icc, p_icc, amp_icc, routing_type, k) in transpiled_circuits:
                    # Circuit does not need to be transpiled again
                    print("Utilizing existing backend")
                    return transpiled_circuits[(n_icc, p_icc, amp_icc, routing_type, k)]
                else:
                    circuit, partitions = get_tqec_cnot_rotated(distance_scale=k, n1=1, n2=0)

                    backend = get_backend(n_icc=n_icc, p_icc=p_icc, amp_icc=amp_icc, k=k)

                    # Transpile circuit to backend
                    _, custom_circuit, _, _ = transpile_stim_circuit(
                        circuit,
                        backend,
                        pre_defined_partitions=partitions,
                        routing_type=routing_type,  # "cost",
                        routing_alpha=1.0,
                        routing_beta=1.0,
                    )
                    # Convert circuit to stim
                    custom_circuit_stim = get_stim_circuits_with_detectors(custom_circuit)[0][0]
                    # Add circuit to dictionary, in order to not transpile this circuit configuration again
                    transpiled_circuits[(n_icc, p_icc, amp_icc, routing_type, k)] = custom_circuit_stim

                    plot_circuit_layout(
                        custom_circuit,
                        backend,
                        filename=f"experiments/evaluation/inter_chiplet/backend_mapping/layout_{n_icc}_{k}.png",
                    )

                    plot_circuit_layout_utilization(
                        custom_circuit,
                        backend,
                        filename=f"experiments/evaluation/inter_chiplet/backend_mapping/mapping_{n_icc}_{k}.png",
                    )

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

                    return BackendChipletV2(
                        size=chiplet_size,  # (2, 2, 15, 8),
                        n_inter=n_icc,
                        connectivity="nn",
                        topology="rotated_grid",
                        inter_chiplet_noise=p_icc,
                        inter_chiplet_amplification=amp_icc,
                        inter_chiplet_noise_type="constant",
                    )

            def _get_sinter_task():
                # Construct sinter task for multiple code distances and noise levels
                yield from (
                    sinter.Task(
                        circuit=circuit,
                        json_metadata={"d": 2 * k + 1, "r": 2 * k + 1, "p": p, "run_name": t, "p_inter": p_icc},
                    )
                    for circuit, k, p, t, rt, p_icc in (
                        (
                            get_noise_model(
                                "modsi1000",
                                None,
                                p,
                                None,
                                remote=(
                                    None
                                    if t == "default"
                                    else get_backend(int(t), ps_inter, 1, t, k).inter_chiplet_connections
                                ),
                            ).noisy_circuit(
                                get_circuit(-1 if t == "default" else int(t), ps_inter, 1, t, routing_type=rt, k=k)
                            ),
                            k,
                            p,
                            t,
                            rt,
                            ps_inter,
                        )
                        for t in ts
                        for rt in routing_types
                        for k in ks
                        for p in ps
                    )
                )

            # Run simulation
            stats = run_sinter_simulation(_get_sinter_task, ks, ps)

            # Save simulation results
            output_dir = Path("experiments/evaluation/inter_chiplet")
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / f"inter_chiplet_{ps_inter}_sweep.pkl", "wb") as f:
                pickle.dump(stats, f)

        # Load simulation results
        with open(f"experiments/evaluation/inter_chiplet/inter_chiplet_{ps_inter}_sweep.pkl", "rb") as f:
            stats = pickle.load(f)

        # Plot statistics
        plot_evaluation(
            stats,
            filename=f"experiments/evaluation/inter_chiplet/inter_chiplet_{ps_inter}.pdf",
            inter_chiplet_noise=ps_inter,
            num_inter=num_inter_chiplet_connections,
        )

        # Plot difference to full inter-chiplet connections
        plot_difference(
            stats,
            filename=f"experiments/evaluation/inter_chiplet/inter_chiplet_{ps_inter}_difference.pdf",
            inter_chiplet_noise=ps_inter,
        )


if __name__ == "__main__":
    run_exp_distributed_inter_chiplet(reproduce=True)
