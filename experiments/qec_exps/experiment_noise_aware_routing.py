from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.getcwd(), "."))

from stim import Circuit as StimCircuit
from experiments.exp_utils.circuit_generator import get_tqec_cnot_rotated
from experiments.exp_utils.simulation_utils import *
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.utils import *
from experiments.exp_utils.circuit_noise import get_noise_model
from glue.qiskit_qec.stim_tools import get_stim_circuits_with_detectors
from qeccm.backends.backend_utils import plot_circuit_layout_utilization

# Plotting
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def ci95_bootstrap(values):

    mean = np.mean(values)
    # Create fake replications by sampling own data with replacement
    boot_means = [np.mean(np.random.choice(values, size=len(values), replace=True)) for _ in range(5000)]
    # Find the bounds where 95% of those means fall
    low_perc = np.percentile(boot_means, 2.5)
    high_perc = np.percentile(boot_means, 97.5)

    return mean, mean - low_perc, high_perc - mean


def plot_evaluation(stats, filename, inter_chiplet_noise):
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

    d_values = sorted(d_values)
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

    # fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.6, WIDTH_FIGSIZE))
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE * 2.5, WIDTH_FIGSIZE * 0.5))
    handles = []
    # Plot identity (x = y)
    h = plt.plot(
        physical_error_rates, physical_error_rates, linestyle="--", linewidth=1.5, color="#000000B3", label="x=y"
    )
    # handles.extend(h)

    colors_transpiled = ["#5E97CC", "#3B6FA8", "#2A5687"]
    colors_default = ["#C85E59", "#9F3B36", "#7F2E2A"]
    color_list = [colors_default, colors_transpiled]

    inter_markers = ["x", "o", "s"]
    # plot_label = ["Basic, Low Variance", "Basic, High Variance", "Cost, Low Variance", "Cost, High Variance", "Tradeoff, High Variance", "Tradeoff, High Variance"]
    # for ti, t in enumerate(["basic10", "basic100", "cost_inter10", "cost_inter100", "cost_tradeoff10", "cost_tradeoff100"]):
    plot_label = [
        "Basic, Low Variance",
        "Basic, High Variance",
        "Tradeoff, Low Variance",
        "Tradeoff, High Variance",
        "Focus, Low Variance",
        "Focus, High Variance",
    ]

    for ti, t in enumerate(
        ["basic10", "basic100", "cost_tradeoff10", "cost_tradeoff100", "cost_inter10", "cost_inter100"]
    ):
        for i, d in enumerate(d_values):
            errors = defaultdict(dict)

            for p in physical_error_rates:
                # Create average over samples
                errors[p] = error_rates[t][d][p]

            # Calculate mean and error values
            ys_custom_mean, values_low, values_high = zip(*[ci95_bootstrap(errors[p]) for p in physical_error_rates])
            # Convert from tuple to arrays
            ys_custom_mean = np.array(list(ys_custom_mean))
            values_low = np.array(list(values_low))
            values_high = np.array(list(values_high))

            line_color = "#2A5687" if (t in ["basic10", "cost_inter10", "cost_tradeoff10"]) else "#7F2E2A"
            line_style = "-" if (t in ["basic10", "basic100"]) else "--"
            if t in ["cost_inter10", "cost_inter100"]:
                marker = "x"
            elif t in ["cost_tradeoff10", "cost_tradeoff100"]:
                marker = "o"
            else:
                marker = ""
            # Plot average values
            h = plt.plot(
                physical_error_rates,
                ys_custom_mean,
                linewidth=1,
                marker=marker,
                # markersize = 4,
                # markerfacecolor="none",
                linestyle=line_style,
                color=line_color,
                label=plot_label[ti],
            )
            handles.append(h[0])
            # Plot shaded area around the average value
            plt.fill_between(
                physical_error_rates,
                ys_custom_mean - values_low,
                ys_custom_mean + values_high,
                color=line_color,
                alpha=0.2,
                edgecolor="none",
            )

    if inter_chiplet_noise == 0.0001:
        ps_inter_text = r"$1e^{-4}$"
    elif inter_chiplet_noise == 0.001:
        ps_inter_text = r"$1e^{-3}$"
    elif inter_chiplet_noise == 0.01:
        ps_inter_text = r"$1e^{-2}$"
    description = r"$p_{inter}$ = " + f"{ps_inter_text}, d = 5"

    ax.text(0.05, 1.02, "b) Effect of cost-routing on LER", transform=ax.transAxes, fontweight="bold")

    ax.text(
        0.3,
        1.13,
        "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    # plt.ylim(-0.01, 0.9)
    plt.ylim(1e-6, 1e0)
    plt.xlim(1e-4, 1e-2)
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Physical error rate")
    plt.ylabel("Logical error rate")
    # plt.legend(loc="lower right", ncol=1)
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    # fig.subplots_adjust(left=0.16, right=0.97, top=0.89, bottom=0.13)
    fig.subplots_adjust(left=0.22, right=0.95, top=0.85, bottom=0.21)
    plt.savefig(filename, format="pdf")
    plt.close(fig)

    legend_fig = plt.figure(figsize=(3, 2))
    legend = legend_fig.legend(handles=handles, loc="center", frameon=False, ncols=3, columnspacing=1.5)
    legend_fig.savefig(filename + "legend.pdf", bbox_inches="tight", format="pdf")
    plt.close(legend_fig)


def plot_error_improvement(stats, filename, inter_chiplet_noise, alpha, beta):
    # Difference between basic routing and cost routing (two options there)

    # Calculate error rate and group
    error_rates = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    physical_error_rates = set()
    d_values = set()
    for s in stats:
        ler = s.errors / (s.shots - s.discards)
        p = s.json_metadata["p"]
        t = s.json_metadata["run_name"]
        d = s.json_metadata["d"]

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
            diff_low_cost[d][p] = np.array(error_rates["basic10"][d][p]) / np.array(
                error_rates["cost_inter10"][d][p]
            )  # - error_rates['basic10'][d][p][0]
            diff_low_cost_tradeoff[d][p] = np.array(error_rates["basic10"][d][p]) / np.array(
                error_rates["cost_tradeoff10"][d][p]
            )  #  - error_rates['basic10'][d][p][0]

            diff_high_cost[d][p] = np.array(error_rates["basic100"][d][p]) / np.array(
                error_rates["cost_inter100"][d][p]
            )  # - error_rates['basic100'][d][p][0]
            diff_high_cost_tradeoff[d][p] = np.array(error_rates["basic100"][d][p]) / np.array(
                error_rates["cost_tradeoff100"][d][p]
            )  # - error_rates['basic100'][d][p][0]

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
    alpha_fill = 0.15
    plt.rcParams.update(tex_fonts)
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE * 2.5, WIDTH_FIGSIZE * 0.5))

    ax.axhline(1, color="black", linestyle="--", linewidth=1.5, alpha=0.5)

    for d in d_values:
        ps_rates = sorted(diff_low_cost[d].keys())
        ys_custom_mean, values_low, values_high = map(
            np.array, zip(*[ci95_bootstrap(diff_low_cost[d][p]) for p in ps_rates])
        )

        plt.plot(ps_rates, ys_custom_mean, marker="x", color="#2A5687", linestyle="--", label="Focus, Low Variance")
        plt.fill_between(
            ps_rates,
            ys_custom_mean - values_low,
            ys_custom_mean + values_high,
            color="#2A5687",
            alpha=alpha_fill,
            edgecolor="none",
        )

        ps_rates = sorted(diff_low_cost_tradeoff[d].keys())
        ys_custom_mean, values_low, values_high = map(
            np.array, zip(*[ci95_bootstrap(diff_low_cost_tradeoff[d][p]) for p in ps_rates])
        )
        plt.plot(ps_rates, ys_custom_mean, marker="o", color="#2A5687", linestyle="--", label="Tradeoff, Low Variance")
        plt.fill_between(
            ps_rates,
            ys_custom_mean - values_low,
            ys_custom_mean + values_high,
            color="#2A5687",
            alpha=alpha_fill,
            edgecolor="none",
        )

        ps_rates = sorted(diff_high_cost[d].keys())
        ys_custom_mean, values_low, values_high = map(
            np.array, zip(*[ci95_bootstrap(diff_high_cost[d][p]) for p in ps_rates])
        )
        plt.plot(ps_rates, ys_custom_mean, marker="x", color="#7F2E2A", linestyle="--", label="Focus, High Variance")
        plt.fill_between(
            ps_rates,
            ys_custom_mean - values_low,
            ys_custom_mean + values_high,
            color="#7F2E2A",
            alpha=alpha_fill,
            edgecolor="none",
        )

        ps_rates = sorted(diff_high_cost_tradeoff[d].keys())
        ys_custom_mean, values_low, values_high = map(
            np.array, zip(*[ci95_bootstrap(diff_high_cost_tradeoff[d][p]) for p in ps_rates])
        )
        plt.plot(ps_rates, ys_custom_mean, marker="o", color="#7F2E2A", linestyle="--", label="Tradeoff, High Variance")
        plt.fill_between(
            ps_rates,
            ys_custom_mean - values_low,
            ys_custom_mean + values_high,
            color="#7F2E2A",
            alpha=alpha_fill,
            edgecolor="none",
        )

    description = (
        r"$p_{inter}$ = "
        f"{inter_chiplet_noise}, "
        r"$\alpha_{cost} = $" + str(3 * alpha) + r", $\alpha_{tradeoff} = $" + str(alpha) + r", $\beta = $" + str(beta)
    )

    # ax.text(
    #    0, 1.02, description,
    #    transform=ax.transAxes,
    #    #fontweight="bold"
    # )

    ax.text(-0.1, 1.02, "c) Relative effect of cost-routing on LER", transform=ax.transAxes, fontweight="bold")

    ax.text(
        0.3,
        1.13,
        "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    # plt.ylim(-10e-1, 0.5)
    # plt.xlim(1e-4, 1e-2)
    plt.ylim(0.01, 190)
    plt.xlim(1e-4, 1e-2)
    plt.xscale("log")
    plt.yscale("log")
    # plt.yscale('symlog', linthresh=1e-3)  # linear within ±0.001

    plt.xlabel("Physical error rate")
    # plt.ylabel(r"Δ($LER_{Routing Method} - LER_{Basic}$)")
    plt.ylabel(r"Δ$LER_{Routing}$")
    plt.ylabel(r"$LER_{Cost}/LER_{Basic}$")

    # plt.legend(loc="lower right", ncol=2)
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    # fig.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.2)
    fig.subplots_adjust(left=0.22, right=0.95, top=0.85, bottom=0.21)

    plt.savefig(filename, format="pdf")
    plt.close(fig)


def plot_hyperparameter_search(folder_path: str, filename: str):

    # Load baseline
    with open("experiments/evaluation/qec_routing/sweep/routing_0.0_0.0_0.001_sweep.pkl", "rb") as f:
        baseline_stats = pickle.load(f)

    fig, ax = plt.subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax,
        stats=baseline_stats,
        x_func=lambda stats: stats.json_metadata["p"],
        group_func=lambda stats: stats.json_metadata["run_name"],
    )
    # ax.set_ylim(1e-4, 1e-0)
    # ax.set_xlim(5e-2, 5e-1)
    ax.loglog()
    ax.set_title("Repetition Code Error Rates (Phenomenological Noise)")
    ax.set_xlabel("Phyical Error Rate")
    ax.set_ylabel("Logical Error Rate per Shot")
    ax.grid(which="major")
    ax.grid(which="minor")
    ax.legend()
    fig.set_dpi(120)  # Show it bigger
    plt.savefig(f"experiments/evaluation/qec_routing/sweep/imgs/{0}_{0}_baseline.png")
    plt.close(fig)

    error_rates_baseline = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    physical_error_rates_baseline = set()
    d_values = set()
    for s in baseline_stats:
        ler = s.errors / (s.shots - s.discards)
        p = s.json_metadata["p"]
        t = str(s.json_metadata["run_name"])
        d = str(s.json_metadata["d"])

        error_rates_baseline[t][d][p].append(ler)
        physical_error_rates_baseline.add(p)
        d_values.add(d)

    d_values = sorted(d_values)
    physical_error_rates_baseline = sorted(list(physical_error_rates_baseline))

    # Load sweep and calculate logical error rate and improvement
    data_list = []

    for filename_sweep in os.listdir(folder_path):
        if (
            filename_sweep.endswith(".pkl")
            and filename_sweep.startswith("routing_")
            and "0.0_0.0_" not in filename_sweep
        ):
            parts = filename_sweep.replace(".pkl", "").split("_")
            alpha = float(parts[1])
            beta = float(parts[2])

            file_path = os.path.join(folder_path, filename_sweep)
            sinter_stats = pd.read_pickle(file_path)

            """
            fig, ax = plt.subplots(1, 1)
            sinter.plot_error_rate(
                ax=ax,
                stats=sinter_stats,
                x_func=lambda stats: stats.json_metadata['p'],
                group_func=lambda stats: stats.json_metadata['run_name'],
            )
            #ax.set_ylim(1e-4, 1e-0)
            #ax.set_xlim(5e-2, 5e-1)
            ax.loglog()
            ax.set_title("Repetition Code Error Rates (Phenomenological Noise)")
            ax.set_xlabel("Phyical Error Rate")
            ax.set_ylabel("Logical Error Rate per Shot")
            ax.grid(which='major')
            ax.grid(which='minor')
            ax.legend()
            fig.set_dpi(120)  # Show it bigger
            plt.savefig(f"experiments/evaluation/qec_routing/sweep/imgs/{alpha}_{beta}.png")
            plt.close(fig)
            """

            # Extract run statistics
            error_rates = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            physical_error_rates = set()
            d_values = set()
            for s in sinter_stats:
                ler = s.errors / (s.shots - s.discards)
                p = s.json_metadata["p"]
                t = str(s.json_metadata["run_name"])
                d = str(s.json_metadata["d"])

                error_rates[t][d][p].append(ler)
                physical_error_rates.add(p)
                d_values.add(d)

            d_values = sorted(d_values)
            physical_error_rates = sorted(list(physical_error_rates))

            # Calculate error improvement
            error_diff = defaultdict(dict)
            # Use distance 7 case
            d = "7"
            # Only use the high-variance case
            t = f"{alpha}_{beta}_100"
            t_baseline = "0.0_0.0_100"
            # t = f"{alpha}_{beta}_10"
            # t_baseline = "0.0_0.0_10"

            # Calculate logical error rate difference to baseline run
            for p in [p for p in physical_error_rates if p < 1e-3]:
                # In case of mismatching physical error rate in the baseline, find the closest value
                if error_rates_baseline[t_baseline][d][p] == []:
                    closest_d = min(physical_error_rates_baseline, key=lambda x: abs(x - p))
                else:
                    closest_d = p
                    # error_rates_baseline[t_baseline][closest_d][p][0]
                if error_rates[t][d][p][0] != 0:
                    error_diff[p] = error_rates_baseline[t_baseline][d][closest_d][0] / error_rates[t][d][p][0]

            # mean_error = np.mean(list(error_diff.values()))
            mean_error = np.exp(np.mean(np.log(list(error_diff.values()))))
            print(f"{alpha} {beta} {mean_error}")
            data_list.append([alpha, beta, mean_error])

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

    data_array = np.array(data_list)
    alphas = data_array[:, 0]
    betas = data_array[:, 1]
    errors = data_array[:, 2]
    x_log = np.log10(alphas)
    plt.plot([0, 10], [0, 10], color="black", linestyle="--", linewidth=1, zorder=3)

    hb = plt.hexbin(
        alphas,
        betas,
        C=errors,
        gridsize=20,
        vmax=50,
        cmap="viridis",
        reduce_C_function=np.mean,
        mincnt=1,
    )

    # levels = np.linspace(0, 50, 25)
    # tcf = ax.tricontourf(alphas, betas, errors, levels=levels, cmap='viridis', extend='max')

    # triang = tri.Triangulation(alphas, betas)
    # interp_cubic = tri.CubicTriInterpolator(triang, errors, kind='geom')
    ## UniformTriRefiner subdivides the triangles into a finer grid
    # refiner = tri.UniformTriRefiner(triang)
    # tri_refined, z_refined = refiner.refine_field(errors, subdiv=2)
    # levels = np.linspace(0, 50, 100)
    # tcf = ax.tricontourf(tri_refined, z_refined, levels=levels,
    #                    cmap='viridis', extend='both')

    cb_ticks = [0, 10, 20, 30, 40, 50]
    cb = plt.colorbar(hb, ax=ax, label="Improvement Rate", ticks=cb_ticks)
    # cb = plt.colorbar(tcf, label="Improvement Rate")

    # ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'$10^{{{int(x)}}}$'))
    # plt.xlim(0, 5)
    # plt.ylim(0, 5)
    plt.xlabel(r"$\alpha$")
    ax.set_xticks([0, 5, 10])
    plt.ylabel(r"$\beta$")

    ax.text(0.075, 1.03, "c) Effect of hyperparameters", transform=ax.transAxes, fontweight="bold")

    ax.text(
        0.3,
        1.14,
        "Higher is better ↑",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    fig.subplots_adjust(left=0.128, right=0.98, top=0.85, bottom=0.21)
    plt.savefig(filename, format="pdf")
    plt.close(fig)


def perform_noise_aware_routing_sweep(reproduce: bool = False) -> None:
    # Hyperparameter search space
    num_samples = 500
    routing_alpha_range = [0.1, 2]  # [0.01, 10]
    routing_beta_range = [0, 2]  # [0.5, 10]

    # Sample pairs of (alpha, beta)
    samples = np.random.uniform(
        low=[routing_alpha_range[0], routing_beta_range[0]],
        high=[routing_alpha_range[1], routing_beta_range[1]],
        size=(num_samples, 2),
    )
    # Combine samples with baseline run
    all_samples = np.vstack([[0, 0], samples])

    # Noise level
    ps = list(np.logspace(-4, -1, 10))

    # Inter-chiplet noise level
    inter_chiplet_noise = 1e-3

    # Code distance for surface code
    k = 3

    # Inter chiplet noise variance
    #   - Low variance: [1, 10]*inter_connect_noise
    #   - High variance: [1, 100]*inter_connect_noise
    ic_noise_model = [10, 100]

    if reproduce:
        circuit, partitions = get_tqec_cnot_rotated(distance_scale=k, n1=1, n2=0)

        # Store created backends, as these are only dependend on architecture configurations
        transpiled_backends = {}

        for ra, rb in all_samples:
            # Transpiled circuit depend on both alpha and beta values
            transpiled_circuits = {}

            def get_circuit(inter_noise_factor: int, distance_scale: int) -> StimCircuit:
                if (distance_scale, inter_noise_factor) in transpiled_circuits:
                    print("Found")
                    return transpiled_circuits[(distance_scale, inter_noise_factor)]
                backend = get_backend(inter_noise_factor=inter_noise_factor, d=distance_scale)

                _, custom_circuit, _, _ = transpile_stim_circuit(
                    circuit,
                    backend,
                    pre_defined_partitions=partitions,
                    routing_type="cost",
                    routing_alpha=ra * (1 / inter_chiplet_noise),
                    routing_beta=rb,
                )
                # Convert circuit to stim
                custom_circuit_stim = get_stim_circuits_with_detectors(custom_circuit)[0][0]

                plot_circuit_layout_utilization(
                    custom_circuit,
                    backend,
                    filename=f"experiments/evaluation/qec_routing/sweep/mapping_{distance_scale}.png",
                )

                transpiled_circuits[(distance_scale, inter_noise_factor)] = custom_circuit_stim
                return custom_circuit_stim

            def get_backend(inter_noise_factor: int, d: int) -> BackendChipletV2:
                if (inter_noise_factor, d) in transpiled_backends:
                    return transpiled_backends[(inter_noise_factor, d)]

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

                backend = BackendChipletV2(
                    size=chiplet_size,
                    n_inter=nic,
                    connectivity="nn",
                    topology="rotated_grid",
                    inter_chiplet_noise=inter_chiplet_noise,
                    inter_chiplet_amplification=1,
                    inter_chiplet_rfactor=inter_noise_factor,
                    inter_chiplet_noise_type="random",
                    num_defective_qubits=0,
                )
                transpiled_backends[(inter_noise_factor, d)] = backend

                return backend

            def _get_sinter_task():
                # Construct sinter task for multiple code distances and noise levels
                yield from (
                    sinter.Task(
                        circuit=circuit,
                        # TODO: the naming is incorrect
                        json_metadata={"d": 2 * k + 1, "p": p, "run_name": f"{ra}_{rb}_{icnm}"},
                    )
                    for circuit, k, p, icnm in (
                        (
                            get_noise_model(
                                "modsi1000",
                                None,
                                p,
                                None,
                                remote=(get_backend(inter_noise_factor=icnm, d=k).inter_chiplet_connections),
                            ).noisy_circuit(get_circuit(inter_noise_factor=icnm, distance_scale=k)),
                            k,
                            p,
                            icnm,
                        )
                        for p in ps
                        for icnm in ic_noise_model
                    )
                )

            # Perform simulations with reduced number of shots
            stats = run_sinter_simulation(_get_sinter_task, [k], ps, num_shots=1_000_000, num_t=10 * 2)

            # Save results
            output_dir = Path("experiments/evaluation/qec_routing/sweep")
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / f"routing_{ra}_{rb}_{inter_chiplet_noise}_sweep.pkl", "wb") as f:
                pickle.dump(stats, f)

    # Plot hyperparameter sweep
    plot_hyperparameter_search(
        "experiments/evaluation/qec_routing/sweep/", "experiments/evaluation/qec_routing/routing_sweep.pdf"
    )


def run_noise_aware_routing(reproduce: bool = False) -> None:
    # Alpha values
    routing_alpha = [3]

    # Noise level
    ps = list(np.logspace(-4, -1, 10))

    # Inter-chiplet noise level
    inter_chiplet_noise = [1e-3]  # [1e-4, 1e-3, 1e-2]

    # Code distance for surface code
    k = 3

    # Routing types
    rts = ["basic", "cost_inter", "cost_tradeoff"]

    # Iterations
    n_iter = 4

    # Inter chiplet noise variance
    #   - Low variance: [1, 10]*inter_connect_noise
    #   - High variance: [1, 100]*inter_connect_noise
    ic_noise_model = [10, 100]

    # Generate CNOT lattice surgery circuit
    circuit, partitions = get_tqec_cnot_rotated(distance_scale=k, n1=1, n2=0)

    if reproduce:
        for ra in routing_alpha:
            for ps_inter in inter_chiplet_noise:
                transpiled_circuits = {}

                def get_circuit(
                    routing_type: str, inter_noise_factor: int, distance_scale: int, seed: int
                ) -> StimCircuit:

                    if (routing_type, inter_noise_factor, seed) in transpiled_circuits:
                        # Circuit does not need to be transpiled again
                        print("Found")
                        return transpiled_circuits[(routing_type, inter_noise_factor, seed)]

                    n_icc, backend = get_backend(inter_noise_factor=inter_noise_factor, d=distance_scale, seed=seed)

                    # Transpile circuit to backend
                    if routing_type == "cost_inter":
                        routing_type_u = "cost"
                        routing_alpha = 3 * ra * 1 / ps_inter  # 3*ps_inter
                        routing_beta = 1
                    elif routing_type == "cost_tradeoff":
                        routing_type_u = "cost"
                        routing_alpha = ra * 1 / ps_inter  # 1*ps_inter
                        routing_beta = 1
                    else:
                        routing_type_u = "cost"
                        routing_alpha = 0
                        routing_beta = 0

                    _, custom_circuit, _, _ = transpile_stim_circuit(
                        circuit,
                        backend,
                        pre_defined_partitions=partitions,
                        routing_type=routing_type_u,
                        routing_alpha=routing_alpha,
                        routing_beta=routing_beta,
                    )
                    # Convert circuit to stim
                    custom_circuit_stim = get_stim_circuits_with_detectors(custom_circuit)[0][0]
                    # Add circuit to dictionary, in order to not transpile this circuit configuration again
                    transpiled_circuits[(routing_type, inter_noise_factor, seed)] = custom_circuit_stim

                    return custom_circuit_stim

                def get_backend(inter_noise_factor: int, d: int, seed: int) -> BackendChipletV2:

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

                    backend = BackendChipletV2(
                        size=chiplet_size,
                        n_inter=nic,
                        connectivity="nn",
                        topology="rotated_grid",
                        inter_chiplet_noise=ps_inter,
                        inter_chiplet_amplification=1,
                        inter_chiplet_rfactor=inter_noise_factor,
                        inter_chiplet_noise_type="random",
                        num_defective_qubits=0,
                        rng_seed=seed,
                    )

                    return nic, backend

                def _get_sinter_task():
                    # Construct sinter task for multiple code distances and noise levels
                    yield from (
                        sinter.Task(
                            circuit=circuit,
                            json_metadata={"d": 2 * k + 1, "p": p, "run_name": rt + str(icnm), "iter": iter_seed},
                        )
                        for circuit, k, p, rt, icnm, iter_seed in (
                            (
                                get_noise_model(
                                    "modsi1000",
                                    None,
                                    p,
                                    None,
                                    remote=(
                                        get_backend(inter_noise_factor=icnm, d=k, seed=iter_seed)[
                                            1
                                        ].inter_chiplet_connections
                                    ),
                                ).noisy_circuit(
                                    get_circuit(
                                        routing_type=rt, inter_noise_factor=icnm, distance_scale=k, seed=iter_seed
                                    )
                                ),
                                k,
                                p,
                                rt,
                                icnm,
                                iter_seed,
                            )
                            for p in ps
                            for icnm in ic_noise_model
                            for rt in rts
                            for iter_seed in range(n_iter)
                        )
                    )

                # Run simulation
                stats = run_sinter_simulation(_get_sinter_task, [k], ps)

                # Save results
                output_dir = Path("experiments/evaluation/qec_routing")
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_dir / f"routing_{ra}_{ps_inter}_sweep.pkl", "wb") as f:
                    pickle.dump(stats, f)

    # Plot simulation results
    for ra in routing_alpha:
        for ps in inter_chiplet_noise:
            if ps == 0.0001:
                ps_inter_text = r"$1e^{-4}$"
            elif ps == 0.001:
                ps_inter_text = r"$1e^{-3}$"
            elif ps == 0.01:
                ps_inter_text = r"$1e^{-2}$"

            with open(f"experiments/evaluation/qec_routing/routing_{ra}_{ps}_sweep.pkl", "rb") as f:
                stats = pickle.load(f)

            plot_evaluation(stats, f"experiments/evaluation/qec_routing/routing_{ra}_{ps}.pdf", ps)

            plot_error_improvement(
                stats, f"experiments/evaluation/qec_routing/routing_difference_{ra}_{ps}.pdf", ps_inter_text, ra, 1
            )


if __name__ == "__main__":
    # Complete routing sweep over hyperparameters
    # perform_noise_aware_routing_sweep()

    # Routing for specific configurations
    run_noise_aware_routing(reproduce=True)
