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
from experiments.exp_utils.ablation_utils import (
    check_routed,
    chipmunq_mapping,
    overhead_stats,
    quiet,
    route_from_mapping,
    symmetric_remote_noise,
)

# Plotting
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


# Link noise: ``backend.inter_chiplet_connections`` stores each link in one orientation and the noise model
# matches ``tuple(pair)`` exactly, so a remote gate applied as (b, a) got *intra*-chiplet noise. Roughly a third
# of the remote gates of the Basic/Tradeoff circuits are affected. True applies link noise in both orientations
# (required for a fair SABRE comparison, SABRE orients SWAPs arbitrarily); False reproduces the original numbers.
SYMMETRIC_REMOTE_NOISE = True

# Run name -> router of experiments.exp_utils.ablation_utils.route_from_mapping
ROUTER_OF_RUN = {"basic": "basic", "cost_inter": "focus", "cost_tradeoff": "tradeoff", "sabre": "sabre", "seqc": "seqc"}

# Chipmunq's own routers; the SABRE-SWAP time budget is the slowest of these
CHIPMUNQ_RUNS = ("basic", "cost_inter", "cost_tradeoff")

# Plot styling per run name (without the variance suffix)
RUN_STYLE = {
    "basic": dict(label="Basic", linestyle="-", marker=""),
    "cost_tradeoff": dict(label="Tradeoff", linestyle="--", marker="o"),
    "cost_inter": dict(label="Focus", linestyle="--", marker="x"),
    "sabre": dict(label="SABRE-SWAP", linestyle=":", marker="^"),
    "seqc": dict(label="SEQC", linestyle="-.", marker="D"),
}
VARIANCE_COLOR = {"10": "#2A5687", "100": "#7F2E2A"}
VARIANCE_LABEL = {"10": "Low Variance", "100": "High Variance"}


def remote_noise(backend):
    return symmetric_remote_noise(backend) if SYMMETRIC_REMOTE_NOISE else backend.inter_chiplet_connections


def split_run_name(t: str) -> tuple[str, str]:
    """'cost_tradeoff100' -> ('cost_tradeoff', '100')"""
    for v in ("100", "10"):
        if t.endswith(v):
            return t[: -len(v)], v
    raise ValueError(t)


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

    present = {str(s.json_metadata["run_name"]) for s in stats}
    order = [rt + v for rt in ("basic", "cost_tradeoff", "cost_inter", "sabre", "seqc") for v in ("10", "100")]

    for t in [t for t in order if t in present]:
        rt, var = split_run_name(t)
        style = RUN_STYLE[rt]
        for i, d in enumerate(d_values):
            ys_custom_mean, values_low, values_high = map(
                np.array, zip(*[ci95_bootstrap(error_rates[t][d][p]) for p in physical_error_rates])
            )
            h = plt.plot(
                physical_error_rates,
                ys_custom_mean,
                linewidth=1,
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=VARIANCE_COLOR[var],
                label=f"{style['label']}, {VARIANCE_LABEL[var]}",
            )
            handles.append(h[0])
            plt.fill_between(
                physical_error_rates,
                ys_custom_mean - values_low,
                ys_custom_mean + values_high,
                color=VARIANCE_COLOR[var],
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


def plot_error_improvement(stats, filename, inter_chiplet_noise, alpha, beta, baseline: str = "basic"):
    """Ratio LER_baseline / LER_method (>1: the method beats the baseline).

    ``baseline``: "basic" (Chipmunq without noise-aware routing, as in the original figure), "sabre"
    (SABRE-SWAP from the same Chipmunq mapping) or "seqc" (SEQC's noise-aware routing from the same mapping). Ratios are formed per backend seed (``iter``) and then
    bootstrapped, so a seed's method result is only ever compared with the same seed's baseline.
    """
    ler = defaultdict(dict)  # (run_name, d, p) -> {iter: ler}
    physical_error_rates, d_values = set(), set()
    for s in stats:
        md = s.json_metadata
        ler[(md["run_name"], md["d"], md["p"])][md.get("iter", 0)] = s.errors / (s.shots - s.discards)
        physical_error_rates.add(md["p"])
        d_values.add(md["d"])
    physical_error_rates = sorted(physical_error_rates)

    methods = ["cost_inter", "cost_tradeoff"] + (["basic"] if baseline != "basic" else [])

    tex_fonts = {
        "font.family": "serif",
        "axes.labelsize": FONTSIZE * 1.5,
        "font.size": FONTSIZE * 1.2,
        "legend.fontsize": (FONTSIZE - 2) * 1.3,
        "xtick.labelsize": (FONTSIZE - 1) * 1.3,
        "ytick.labelsize": (FONTSIZE - 1) * 1.3,
        "axes.titlesize": 10,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "lines.markeredgewidth": 1.5,
        "lines.markeredgecolor": "black",
        "errorbar.capsize": 3,
    }
    alpha_fill = 0.15
    plt.rcParams.update(tex_fonts)
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE * 2.5, WIDTH_FIGSIZE * 0.5))
    ax.axhline(1, color="black", linestyle="--", linewidth=1.5, alpha=0.5)

    for d in sorted(d_values):
        for var in ("10", "100"):
            for m in methods:
                xs, mean, lo, hi = [], [], [], []
                for p in physical_error_rates:
                    base = ler.get((baseline + var, d, p), {})
                    meth = ler.get((m + var, d, p), {})
                    ratios = [base[i] / meth[i] for i in base if i in meth and meth[i] > 0 and base[i] > 0]
                    if ratios:
                        mu, l, h = ci95_bootstrap(ratios)
                        xs.append(p), mean.append(mu), lo.append(l), hi.append(h)
                if not xs:
                    continue
                mean, lo, hi = map(np.array, (mean, lo, hi))
                plt.plot(xs, mean, marker=RUN_STYLE[m]["marker"] or "s", color=VARIANCE_COLOR[var],
                         linestyle=RUN_STYLE[m]["linestyle"],
                         label=f"{RUN_STYLE[m]['label']}, {VARIANCE_LABEL[var]}")
                plt.fill_between(xs, mean - lo, mean + hi, color=VARIANCE_COLOR[var], alpha=alpha_fill,
                                 edgecolor="none")

    base_label = RUN_STYLE[baseline]["label"]
    title = "c) Relative effect of cost-routing on LER" if baseline == "basic" else \
        f"c) Cost-routing vs. {RUN_STYLE[baseline]['label']}"
    ax.text(-0.1, 1.02, title, transform=ax.transAxes, fontweight="bold")
    ax.text(0.3, 1.13, "Higher is better ↑", transform=ax.transAxes, fontweight="bold", color=plot_lib_color)

    plt.ylim(0.01, 190)
    plt.xlim(1e-4, 1e-2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Physical error rate")
    plt.ylabel(r"$LER_{" + base_label.replace("-SWAP", "") + r"}/LER_{Method}$")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    fig.subplots_adjust(left=0.22, right=0.95, top=0.85, bottom=0.21)
    plt.savefig(filename, format="pdf")
    plt.close(fig)

    legend_fig = plt.figure(figsize=(3, 2))
    legend_fig.legend(*ax.get_legend_handles_labels(), loc="center", frameon=False, ncols=3)
    legend_fig.savefig(str(filename) + "legend.pdf", bbox_inches="tight", format="pdf")
    plt.close(legend_fig)


def plot_error_improvement_vs_baselines(stats, filename, baselines=("sabre", "seqc"),
                                        methods=("cost_inter", "cost_tradeoff")):
    """c) Chipmunq's cost routing against several baselines in one panel.

    y = LER_baseline / LER_Chipmunq (>1: Chipmunq is better), formed per backend seed (``iter``) so a
    seed's Chipmunq result is only compared with the same seed's baseline, then bootstrapped.
    Colour: link-noise variance. Line style: baseline. Marker: Chipmunq router.
    """
    ler = defaultdict(dict)  # (run_name, d, p) -> {iter: ler}
    physical_error_rates, d_values, present = set(), set(), set()
    for s in stats:
        md = s.json_metadata
        ler[(md["run_name"], md["d"], md["p"])][md.get("iter", 0)] = s.errors / (s.shots - s.discards)
        physical_error_rates.add(md["p"])
        d_values.add(md["d"])
        present.add(split_run_name(str(md["run_name"]))[0])
    physical_error_rates = sorted(physical_error_rates)
    baselines = [b for b in baselines if b in present]
    methods = [m for m in methods if m in present]

    baseline_linestyle = {"sabre": ":", "seqc": "-.", "basic": "-"}

    tex_fonts = {
        "font.family": "serif",
        "axes.labelsize": FONTSIZE * 1.5,
        "font.size": FONTSIZE * 1.2,
        "legend.fontsize": (FONTSIZE - 2) * 1.3,
        "xtick.labelsize": (FONTSIZE - 1) * 1.3,
        "ytick.labelsize": (FONTSIZE - 1) * 1.3,
        "axes.titlesize": 10,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "lines.markeredgewidth": 1.5,
        "lines.markeredgecolor": "black",
        "errorbar.capsize": 3,
    }
    plt.rcParams.update(tex_fonts)
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE * 2.5, WIDTH_FIGSIZE * 0.5))
    ax.axhline(1, color="black", linestyle="--", linewidth=1.5, alpha=0.5)

    for d in sorted(d_values):
        for var in ("10", "100"):
            for b in baselines:
                for m in methods:
                    xs, mean, lo, hi = [], [], [], []
                    for p in physical_error_rates:
                        base = ler.get((b + var, d, p), {})
                        meth = ler.get((m + var, d, p), {})
                        ratios = [base[i] / meth[i] for i in base if i in meth and meth[i] > 0 and base[i] > 0]
                        if ratios:
                            mu, l, h = ci95_bootstrap(ratios)
                            xs.append(p), mean.append(mu), lo.append(l), hi.append(h)
                    if not xs:
                        continue
                    mean, lo, hi = map(np.array, (mean, lo, hi))
                    plt.plot(xs, mean, marker=RUN_STYLE[m]["marker"] or "s", color=VARIANCE_COLOR[var],
                             linestyle=baseline_linestyle.get(b, "-"), linewidth=1,
                             label=f"{RUN_STYLE[m]['label']} vs. {RUN_STYLE[b]['label']}, {VARIANCE_LABEL[var]}")
                    plt.fill_between(xs, mean - lo, mean + hi, color=VARIANCE_COLOR[var], alpha=0.1,
                                     edgecolor="none")

    names = " and ".join(RUN_STYLE[b]["label"] for b in baselines)
    ax.text(-0.1, 1.02, f"c) Chipmunq vs. {names}", transform=ax.transAxes, fontweight="bold")
    ax.text(0.3, 1.13, "Higher is better ↑", transform=ax.transAxes, fontweight="bold", color=plot_lib_color)

    plt.ylim(0.01, 190)
    plt.xlim(1e-4, 1e-2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Physical error rate")
    plt.ylabel(r"$LER_{Baseline}/LER_{Chipmunq}$")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    fig.subplots_adjust(left=0.22, right=0.95, top=0.85, bottom=0.21)
    plt.savefig(filename, format="pdf")
    plt.close(fig)

    # Legend as its own figure (like the other panels): baseline / router / variance explained separately,
    # so it stays small although the panel has up to 8 curves
    from matplotlib.lines import Line2D
    handles = (
        [Line2D([], [], color="black", linestyle=baseline_linestyle.get(b, "-"), label=f"vs. {RUN_STYLE[b]['label']}")
         for b in baselines]
        + [Line2D([], [], color="black", linestyle="", marker=RUN_STYLE[m]["marker"] or "s",
                  markerfacecolor="white", label=f"Chipmunq {RUN_STYLE[m]['label']}") for m in methods]
        + [Line2D([], [], color=VARIANCE_COLOR[v], linewidth=4, label=VARIANCE_LABEL[v]) for v in ("10", "100")]
    )
    legend_fig = plt.figure(figsize=(3, 2))
    legend_fig.legend(handles=handles, loc="center", frameon=False, ncols=3, columnspacing=1.5)
    legend_fig.savefig(str(filename) + "legend.pdf", bbox_inches="tight", format="pdf")
    plt.close(legend_fig)


def plot_routing_overhead(compile_stats: dict, filename: str) -> None:
    """Routing runtime, 2q-gate overhead, depth overhead and inter-chiplet 2q gates per router
    (mean and 95% CI over backend seeds), for both link-noise variances."""
    metrics = [("routing_s", "Routing time [ms]", 1e3), ("2q_overhead", "2q-gate overhead", 1),
               ("depth_overhead", "Depth overhead", 1), ("inter_chiplet_2q", "Inter-chiplet 2q", 1)]
    runs = [rt for rt in ("basic", "cost_tradeoff", "cost_inter", "sabre", "seqc")
            if any(k[0] == rt for k in compile_stats)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(HEIGHT_FIGSIZE * 5, WIDTH_FIGSIZE * 0.45))
    w = 0.38
    for ax, (key, ylabel, scale) in zip(axes, metrics):
        for j, var in enumerate(("10", "100")):
            for i, rt in enumerate(runs):
                vals = [v[key] * scale for k, v in compile_stats.items() if k[0] == rt and str(k[1]) == var]
                if not vals:
                    continue
                mu, lo, hi = ci95_bootstrap(vals)
                ax.bar(i + (j - 0.5) * w, mu, w, yerr=[[lo], [hi]], color=VARIANCE_COLOR[var], edgecolor="black",
                       hatch={"sabre": "//", "seqc": ".."}.get(rt, ""), alpha=0.85,
                       label=VARIANCE_LABEL[var] if (i == 0 and key == "routing_s") else None)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels([RUN_STYLE[r]["label"] for r in runs], rotation=30, ha="right")
        ax.set_title(ylabel, fontsize=FONTSIZE)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    axes[0].legend(fontsize=FONTSIZE - 4)
    fig.text(0.01, 0.98, "Routing cost from the same Chipmunq mapping (SABRE-SWAP: equal time budget; "
                         "SEQC: unconstrained). "
                         "Lower is better ↓", va="top", fontweight="bold", color=plot_lib_color)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(filename, format="pdf")
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

    # SABRE-SWAP from the same mapping, expressed on the same scale (LER_Basic / LER_SABRE)
    sabre_file = os.path.join(folder_path, "routing_sabre_0.001_sweep.pkl")
    if os.path.exists(sabre_file):
        sabre_stats = pd.read_pickle(sabre_file)
        sabre_ler = defaultdict(list)
        for st in sabre_stats:
            if str(st.json_metadata["run_name"]) == "sabre_100" and str(st.json_metadata["d"]) == "7":
                sabre_ler[st.json_metadata["p"]].append(st.errors / (st.shots - st.discards))
        ratios = []
        for p, v in sabre_ler.items():
            if p >= 1e-3 or not v or v[0] == 0:
                continue
            closest = min(physical_error_rates_baseline, key=lambda x: abs(x - p))
            base = error_rates_baseline["0.0_0.0_100"]["7"][closest]
            if base:
                ratios.append(base[0] / v[0])
        if ratios:
            sabre_improvement = float(np.exp(np.mean(np.log(ratios))))
            print(f"SABRE-SWAP improvement over Basic: {sabre_improvement}")
            cb.ax.axhline(min(sabre_improvement, 50), color="white", linewidth=2, linestyle="--")
            cb.ax.text(1.6, min(sabre_improvement, 50), f"SABRE\n×{sabre_improvement:.1f}",
                       transform=cb.ax.get_yaxis_transform(), va="center", fontsize=FONTSIZE - 2)
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
                                remote=remote_noise(get_backend(inter_noise_factor=icnm, d=k)),
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

        # SABRE-SWAP reference: routed from the same Chipmunq mapping on the same backends, with the
        # compile-time budget of the Chipmunq routers. Independent of (alpha, beta), so run once.
        sabre_circuits = {}
        for icnm in ic_noise_model:
            backend = get_backend(inter_noise_factor=icnm, d=k)
            sabre_circuits[icnm], _ = compile_same_mapping(circuit, partitions, backend, inter_chiplet_noise,
                                                           ra=1, seed=0, runs=("sabre",))
        sabre_stats = run_sinter_simulation(
            lambda: (
                sinter.Task(
                    circuit=get_noise_model("modsi1000", None, p, None,
                                            remote=remote_noise(get_backend(inter_noise_factor=icnm, d=k)))
                    .noisy_circuit(sabre_circuits[icnm]["sabre"]["stim"]),
                    json_metadata={"d": 2 * k + 1, "p": p, "run_name": f"sabre_{icnm}"},
                )
                for p in ps
                for icnm in ic_noise_model
            ),
            [k], ps, num_shots=1_000_000, num_t=10 * 2,
        )
        output_dir = Path("experiments/evaluation/qec_routing/sweep")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"routing_sabre_{inter_chiplet_noise}_sweep.pkl", "wb") as f:
            pickle.dump(sabre_stats, f)

    # Plot hyperparameter sweep
    plot_hyperparameter_search(
        "experiments/evaluation/qec_routing/sweep/", "experiments/evaluation/qec_routing/routing_sweep.pdf"
    )


def compile_same_mapping(circuit, partitions, backend, ps_inter, ra, seed=0, runs=tuple(ROUTER_OF_RUN)) -> tuple:
    """Compile ``circuit`` once with Chipmunq's mapping and route it with every router in ``runs``.

    All routers start from the *same* mapping on the *same* backend. SABRE-SWAP gets the
    compile-time budget of the Chipmunq routers (the slowest of Basic/Focus/Tradeoff on this
    backend), spent on independent restarts keeping the fewest-SWAP result. SEQC routes with its
    own noise-aware inter-chiplet routing (Alg. 2 + fidelity-weighted link assignment) and is not
    time-budgeted (its runtime is reported).

    Returns ({run_name: {"stim", "routing_s", "mapping_s", overhead stats...}}, budget_s).
    """
    with quiet():
        circuit_qc = StimCodeCircuit(stim_circuit=circuit).qc
    mapping, t_map = chipmunq_mapping(circuit_qc, backend, partitions)
    compiled = {}
    # Chipmunq routers (and SABRE's budget) always, so SABRE's budget does not depend on ``runs``
    for rt in CHIPMUNQ_RUNS:
        out, info = route_from_mapping(circuit_qc, backend, mapping, ROUTER_OF_RUN[rt], ps_inter, ra=ra)
        compiled[rt] = {"routed": out, **info}
    budget = max(compiled[r]["routing_s"] for r in CHIPMUNQ_RUNS)
    if "sabre" in runs:
        out, info = route_from_mapping(circuit_qc, backend, mapping, "sabre", ps_inter, seed=seed, budget_s=budget)
        compiled["sabre"] = {"routed": out, **info}
    if "seqc" in runs:
        # SEQC's own noise-aware inter-chiplet routing, pinned to the same mapping. Not time-budgeted.
        out, info = route_from_mapping(circuit_qc, backend, mapping, "seqc", ps_inter, seed=seed)
        compiled["seqc"] = {"routed": out, **info}
    compiled = {rt: rec for rt, rec in compiled.items() if rt in runs}
    for rt, rec in compiled.items():
        routed = rec.pop("routed")
        rec["stim"] = check_routed(routed, circuit, backend, stim_circuit=rec.pop("stim", None))
        rec["mapping_s"] = t_map
        rec.update(overhead_stats(routed, circuit_qc, backend))
    return compiled, budget


def run_noise_aware_routing(reproduce: bool = False) -> None:
    # Alpha values
    routing_alpha = [3]

    # Noise level
    ps = list(np.logspace(-4, -1, 10))

    # Inter-chiplet noise level
    inter_chiplet_noise = [1e-3]  # [1e-4, 1e-3, 1e-2]

    # Code distance for surface code
    k = 3

    # Routing types. All start from the same Chipmunq mapping; "sabre" is SABRE-SWAP (LightSABRE routing)
    # with the compile-time budget of the Chipmunq routers, "seqc" SEQC's noise-aware routing.
    rts = ["basic", "cost_inter", "cost_tradeoff", "sabre", "seqc"]

    # Iterations
    n_iter = 4

    # Inter chiplet noise variance
    #   - Low variance: [1, 10]*inter_connect_noise
    #   - High variance: [1, 100]*inter_connect_noise
    ic_noise_model = [10, 100]

    # Generate CNOT lattice surgery circuit
    circuit, partitions = get_tqec_cnot_rotated(distance_scale=k, n1=1, n2=0)

    def get_backend(inter_noise_factor: int, d: int, seed: int, ps_inter: float) -> BackendChipletV2:
        # Depending on the distance, each chiplet needs to be scaled
        chiplet_size, nic = {
            1: ((2, 2, 11, 6), 5),
            2: ((2, 2, 15, 8), 7),
            3: ((2, 2, 19, 10), 9),
            4: ((2, 2, 23, 12), 11),
        }[d]
        return BackendChipletV2(
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

    output_dir = Path("experiments/evaluation/qec_routing")
    if reproduce:
        output_dir.mkdir(parents=True, exist_ok=True)
        for ra in routing_alpha:
            for ps_inter in inter_chiplet_noise:
                # Compile everything up front: one mapping per backend, shared by all routers
                compiled, backends = {}, {}
                for icnm in ic_noise_model:
                    for iter_seed in range(n_iter):
                        backend = get_backend(icnm, k, iter_seed, ps_inter)
                        backends[(icnm, iter_seed)] = backend
                        per_rt, budget = compile_same_mapping(circuit, partitions, backend, ps_inter, ra,
                                                              seed=iter_seed, runs=tuple(rts))
                        for rt, rec in per_rt.items():
                            compiled[(rt, icnm, iter_seed)] = rec
                        print(f"[var {icnm}, seed {iter_seed}] budget {budget * 1e3:.0f} ms: " + ", ".join(
                            f"{rt} {rec['routing_s'] * 1e3:.0f} ms / {rec['swaps']} swaps"
                            for rt, rec in per_rt.items()))

                def _get_sinter_task():
                    yield from (
                        sinter.Task(
                            circuit=get_noise_model(
                                "modsi1000", None, p, None, remote=remote_noise(backends[(icnm, iter_seed)])
                            ).noisy_circuit(compiled[(rt, icnm, iter_seed)]["stim"]),
                            json_metadata={"d": 2 * k + 1, "p": p, "run_name": rt + str(icnm), "iter": iter_seed},
                        )
                        for p in ps
                        for icnm in ic_noise_model
                        for rt in rts
                        for iter_seed in range(n_iter)
                    )

                # Run simulation
                stats = run_sinter_simulation(_get_sinter_task, [k], ps)

                # Save results (Stim circuits are stored as text so the pickle stays portable)
                with open(output_dir / f"routing_{ra}_{ps_inter}_sweep.pkl", "wb") as f:
                    pickle.dump(stats, f)
                compile_stats = {key: {**rec, "stim": str(rec["stim"])} for key, rec in compiled.items()}
                with open(output_dir / f"routing_{ra}_{ps_inter}_compile.pkl", "wb") as f:
                    pickle.dump(compile_stats, f)

    # Plot simulation results
    for ra in routing_alpha:
        for ps in inter_chiplet_noise:
            if ps == 0.0001:
                ps_inter_text = r"$1e^{-4}$"
            elif ps == 0.001:
                ps_inter_text = r"$1e^{-3}$"
            elif ps == 0.01:
                ps_inter_text = r"$1e^{-2}$"

            results_file = output_dir / f"routing_{ra}_{ps}_sweep.pkl"
            if not results_file.exists():
                raise SystemExit(f"No results to plot: {results_file} not found. Run without --plot-only first.")
            with open(results_file, "rb") as f:
                stats = pickle.load(f)

            plot_evaluation(stats, f"experiments/evaluation/qec_routing/routing_{ra}_{ps}.pdf", ps)

            # c) relative to Basic (original figure) and relative to SABRE-SWAP from the same mapping
            plot_error_improvement(
                stats, f"experiments/evaluation/qec_routing/routing_difference_{ra}_{ps}.pdf", ps_inter_text, ra, 1
            )
            # c) Chipmunq (Focus, Tradeoff) against SABRE-SWAP and SEQC in one panel
            if any(str(s.json_metadata["run_name"]).startswith(("sabre", "seqc")) for s in stats):
                plot_error_improvement_vs_baselines(
                    stats, f"experiments/evaluation/qec_routing/routing_difference_baselines_{ra}_{ps}.pdf"
                )

            compile_file = output_dir / f"routing_{ra}_{ps}_compile.pkl"
            if compile_file.exists():
                with open(compile_file, "rb") as f:
                    compile_stats = pickle.load(f)
                plot_routing_overhead(compile_stats, f"experiments/evaluation/qec_routing/routing_overhead_{ra}_{ps}.pdf")
                print("run, variance: routing ms | swaps | 2q overhead | depth overhead | inter-chiplet 2q")
                for rt in rts:
                    for icnm in ic_noise_model:
                        recs = [v for key, v in compile_stats.items() if key[0] == rt and key[1] == icnm]
                        if recs:
                            m = lambda key: np.mean([r[key] for r in recs])  # noqa: E731
                            print(f"{rt:14s} {icnm:4d}: {m('routing_s') * 1e3:7.1f} | {m('swaps'):6.0f} | "
                                  f"{m('2q_overhead'):6.0f} | {m('depth_overhead'):5.0f} | {m('inter_chiplet_2q'):5.0f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Noise-aware routing experiment (Fig. 9)")
    ap.add_argument("--plot-only", action="store_true",
                    help="only replot saved results (experiments/evaluation/qec_routing/), no compilation/simulation")
    ap.add_argument("--sweep", action="store_true",
                    help="run (or, with --plot-only, replot) the alpha/beta hyperparameter sweep instead")
    a = ap.parse_args()

    if a.sweep:
        # Complete routing sweep over hyperparameters
        perform_noise_aware_routing_sweep(reproduce=not a.plot_only)
    else:
        # Routing for specific configurations
        run_noise_aware_routing(reproduce=not a.plot_only)