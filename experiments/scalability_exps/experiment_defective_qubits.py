"""
Fig. 10: defective qubits.

Part 1 -- a) circuit depth, b) #2q gates, c) chiplet utilization:
    Chipmunq vs. LightSABRE on backends with defective qubits (single-/multi-patch chiplets, center /
    size-aware placement).

Part 3 -- d) LER of the two placement strategies:
    LER of the distributed lattice-surgery CNOT after Chipmunq compilation with center and size-aware placement,
    for several code distances and numbers of defective qubits, on the single-patch chiplet backend of a-c
    (one patch per chiplet) and/or the multi-patch backend (several patches share a chiplet). Both strategies
    are compiled on the same backend (same defects) for every defect placement. The LER is plotted relative to
    the untranspiled CNOT circuit (same d, p and noise model); the absolute LER goes to *_abs.pdf. The untranspiled
    reference is simulated only for (d, p) points missing from the saved results. One panel per backend:
        single patch -> combined_overhead_placement_ler.pdf
        multi patch  -> combined_overhead_placement_ler_multi.pdf
    Select the backends with --placement-backends single multi (default: both). Results of backends that are
    not rerun are kept in the pickle, so e.g. `--part placement --placement-backends multi` adds the multi-patch
    results to existing single-patch results.

Part 2 -- c) LER of avoiding vs. deforming (own figure, own legend), plus the utilization tradeoff:
    Defect avoidance (Chipmunq) vs. patch deformation with super-stabilizers (Lin et al., ASPLOS'24 [1]).
    Reviewer request (Comment #6 / B4): compare Chipmunq's regular-patch defect avoidance with prior work that
    deforms patches around defects, and discuss the tradeoff between preserving patch structure and resource
    utilization.

    Setup (one logical qubit per chiplet, as in [1] and in part 1):
      * A chiplet is the qubit set of an L x L rotated surface-code patch (2L^2 - 1 qubits, coordinates as in [1]).
      * k defective qubits are drawn uniformly from the chiplet (qubit defects, as in BackendChipletV2).
      * The program asks for a distance-D patch (D = 5, as in a-c).

      Chipmunq (avoid):   place an intact D x D patch on a defect-free window of the chiplet (window nearest to
                          the centre first). If no defect-free window exists, the chiplet cannot host the patch.
      Lin et al. (deform): use the whole L x L chiplet and deform it around the defects (boundary deformation +
                          super-stabilizers, their code [2]). Distance d_eff varies per chiplet; the patch can fail
                          (percolation) at high defect counts.

      L = D     : tight chiplet (no slack) -- avoidance fails as soon as one defect is present.
      L = D + 2 : chiplet with slack -- avoidance keeps d = D, deformation reaches d_eff <= L.

    Both strategies use the same memory-experiment generator ([2], LogicalQubit.generate_stim) and the noise
    model of [1] (2q: p, 1q: 0.8p, readout: 8/15 p), decoded with PyMatching. Patches with super-stabilizers
    measure X and Z gauges in alternate half-rounds (one generate_stim round = 2 QEC cycles), so the LER is
    reported per QEC cycle.

    [1] S. F. Lin et al., "Codesign of quantum error-correcting codes and modular chiplets in the presence of
        defects", ASPLOS 2024, arXiv:2305.00138.
    [2] https://github.com/SophLin/superstabilizer_demo  (clone into external/baseline/superstabilizer_demo;
        only needed to run part 2)

Sampling: at most 100k shots per circuit by default (parts 2 and 3); override with --max-shots.

All panels are drawn at their printed size for four panels side by side across the full page width. One legend
(combined_overheadlegend.pdf) covers panels a-d, the deformation comparison has its own
(combined_overhead_deformationlegend.pdf); both depend only on the styles below, so they are written on every run,
also with --plot-only and for a single --part.

Run from the repo root:
    python experiments/scalability_exps/experiment_defective_qubits.py                    # all parts
    python experiments/scalability_exps/experiment_defective_qubits.py --plot-only        # replot all
    python experiments/scalability_exps/experiment_defective_qubits.py --part compile     # a-c only
    python experiments/scalability_exps/experiment_defective_qubits.py --part placement [--quick]  # d, both backends
    python experiments/scalability_exps/experiment_defective_qubits.py --part placement --placement-backends multi
    python experiments/scalability_exps/experiment_defective_qubits.py --part deformation [--jobs N] [--quick]
"""
from __future__ import annotations

import argparse
import math
import multiprocessing
import os
import sys
import time

sys.path.append(os.path.join(os.getcwd(), "."))
sys.path.append(os.path.join(os.getcwd(), "./external/baseline/superstabilizer_demo"))

from experiments.exp_utils.circuit_generator import get_tqec_cnot_rotated
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit
from experiments.exp_utils.simulation_utils import *
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.utils import *
from experiments.exp_utils.circuit_noise import get_noise_model

# Plotting
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pymatching
import sinter
import stim
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path

# --------------------------------------------------------------------------------------
# Figure size (all panels a-e): four panels side by side across the full page width. Every panel is drawn
# at its printed size, so include it in LaTeX at its natural width (or width=0.24\textwidth for a 7.0 in
# text width) -- no further scaling, fonts stay 7 pt and identical across the panels.
# --------------------------------------------------------------------------------------
TEXT_WIDTH_IN = 7.0                   # full text width of the paper (two-column IEEE/ACM: ~7.0 in)
N_PANELS = 4                          # panels in one row
PANEL_W = TEXT_WIDTH_IN / N_PANELS    # 1.75 in
PANEL_H = 1.6
FONT_PT = 7
# identical margins for all panels, so the axes line up when placed next to each other
MARGINS = dict(left=0.27, right=0.97, top=0.80, bottom=0.24)

# Maximum shots per simulated circuit (parts 2 and 3); --max-shots overrides it
DEFAULT_MAX_SHOTS = 100_000

# --------------------------------------------------------------------------------------
# Bar styles of panels a-c, shared with the single legend (save_legend). Every entry of the legend must have
# its own colour/hatch, so panels d, e (SERIES) use different colours than a-c.
# --------------------------------------------------------------------------------------
# Chipmunq: single-patch center, single-patch size-aware, multi-patch center, multi-patch size-aware
COLORS_CUSTOM = ["#A7D9ED", "#5B9BD5", "#D9D9D9", "#7F7F7F"]
# LightSABRE: index 0 = single patch, 2 = multi patch (1, 3 unused: LightSABRE has no placement modes)
COLORS_SABRE = ["#F7C6A2", "#E68A5C", "#F7A2A2", "#D65C5C"]
HATCHES = ["...", "//", "xxx", "ooo"]  # one hatch per placement mode (center, size-aware)


# ######################################################################################
# Part 1: Chipmunq vs. LightSABRE on defective backends (Fig. 10 a-c)
# ######################################################################################


def ci95_bootstrap(values, df_values, mode, ks):
    means = []
    err_low = []
    err_high = []
    for df in df_values:
        val = list(values[mode][df][ks].values())
        mean = np.mean(val)
        # Create fake replications by sampling own data with replacement
        boot_means = [np.mean(np.random.choice(val, size=len(val), replace=True)) for _ in range(5000)]
        # Find the bounds where 95% of those means fall
        low_perc = np.percentile(boot_means, 2.5)
        high_perc = np.percentile(boot_means, 97.5)

        means.append(mean)
        err_low.append(mean - low_perc)
        err_high.append(high_perc - mean)

    return means, [err_low, err_high]


def plot_combined_backends(
    custom_depth,
    custom_overhead,
    custom_utilization,
    sabre_depth,
    sabre_overhead,
    sabre_utilization,
    filename: str = "",
):

    # placement modes (outer keys)
    placement_modes = list(custom_depth[0].keys())  # ["default", "size_aware"]

    # defective qubit counts (inner keys)
    df_values = sorted(custom_depth[0][placement_modes[0]].keys())

    ks = list(custom_depth[0][placement_modes[0]][df_values[0]].keys())[0]

    # X-axis (one position per defective-qubit count)
    x = np.arange(len(df_values))  # [0,1,2]

    # Two bars per group
    width = 0.5 / 4  # 0.35/4#0.35/2

    title_left = ""

    # Colors and hatches: module-level COLORS_CUSTOM / COLORS_SABRE / HATCHES (shared with the legend)
    colors_custom = COLORS_CUSTOM
    colors_sabre = COLORS_SABRE
    hatches = HATCHES

    # Panel size, fonts and margins: module-level PANEL_* / FONT_PT / MARGINS (shared with d, e)
    tex_fonts = {
        "font.family": "serif",
        "font.size": FONT_PT,
        "axes.labelsize": FONT_PT,
        "axes.titlesize": FONT_PT,
        "legend.fontsize": FONT_PT,
        "xtick.labelsize": FONT_PT - 1,
        "ytick.labelsize": FONT_PT - 1,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "ytick.minor.size": 1.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.labelpad": 2,
        "axes.linewidth": 0.6,
        "hatch.linewidth": 0.4,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "lines.markeredgewidth": 0.5,
        "lines.markeredgecolor": "black",
        "errorbar.capsize": 1.5,
    }

    plt.rcParams.update(tex_fonts)

    labels = ["center", "size-aware"]

    WIDTH_FIGSIZE = 6
    HEIGHT_FIGSIZE = 2.2

    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))

    for i, mode in enumerate(placement_modes):
        # Single patch
        values_mean_custom, values_err_custom = ci95_bootstrap(custom_depth[0], df_values, mode, ks)
        values_mean_sabre, values_err_sabre = ci95_bootstrap(sabre_depth[0], df_values, mode, ks)

        # Custom
        ax.bar(
            x + i * 1 * width - 2.5 * width,
            values_mean_custom,
            width,
            yerr=values_err_custom,
            capsize=1.5,
            error_kw={"elinewidth": 0.6, "ecolor": "black", "capthick": 0.6},
            linewidth=0.4,
            label=labels[i],
            color=colors_custom[i],
            hatch=hatches[i],
            edgecolor="black",
        )
        if i < 1:
            # SABRE
            ax.bar(
                x + i * 2 * width - 0.5 * width,
                values_mean_sabre,
                width,
                yerr=values_err_sabre,
                capsize=1.5,
                error_kw={"elinewidth": 0.6, "ecolor": "black", "capthick": 0.6},
                linewidth=0.4,
                label=labels[i],
                color=colors_sabre[i],
                # hatch=hatches[i],
                edgecolor="black",
            )

        # Multi patch
        values_mean_custom, values_err_custom = ci95_bootstrap(custom_depth[1], df_values, mode, ks)
        values_mean_sabre, values_err_sabre = ci95_bootstrap(sabre_depth[1], df_values, mode, ks)
        # Custom
        ax.bar(
            x + (2 + i) * width * 2 - 3.5 * width - i * width,
            values_mean_custom,
            width,
            yerr=values_err_custom,
            capsize=1.5,
            error_kw={"elinewidth": 0.6, "ecolor": "black", "capthick": 0.6},
            linewidth=0.4,
            label=labels[i] + "multi",
            color=colors_custom[i + 2],
            hatch=hatches[i],
            edgecolor="black",
        )
        if i < 1:
            # SABRE
            ax.bar(
                x + (2 + i) * width * 2 - 1.5 * width,
                values_mean_sabre,
                width,
                yerr=values_err_sabre,
                capsize=1.5,
                error_kw={"elinewidth": 0.6, "ecolor": "black", "capthick": 0.6},
                linewidth=0.4,
                label=labels[i] + "multi",
                color=colors_sabre[i + 2],
                # hatch=hatches[i + 2],
                edgecolor="black",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(df) for df in df_values])
    ax.set_xlabel("#Defective qubits")
    ax.set_ylabel("Depth overhead")
    # ax.legend(loc='upper left')
    # ax.set_ylim(0, 1250)

    ax.text(0.5, 1.03, "a) Circuit depth", transform=ax.transAxes, fontweight="bold", ha="center", va="bottom")

    ax.text(
        0.5,
        1.16,
        "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
        ha="center",
        va="bottom",
    )

    fig.subplots_adjust(**MARGINS)
    fig.savefig(f"{filename}_depth.pdf", format="pdf")
    plt.close(fig)

    # 2Q Gate Overhead
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))

    for i, mode in enumerate(placement_modes):
        # Single patch
        values_mean_custom, values_err_custom = ci95_bootstrap(custom_overhead[0], df_values, mode, ks)
        values_mean_sabre, values_err_sabre = ci95_bootstrap(sabre_overhead[0], df_values, mode, ks)

        # Custom
        ax.bar(
            x + i * 1 * width - 2.5 * width,
            values_mean_custom,
            width,
            yerr=values_err_custom,
            capsize=1.5,
            error_kw={"elinewidth": 0.6, "ecolor": "black", "capthick": 0.6},
            linewidth=0.4,
            label=labels[i],
            color=colors_custom[i],
            hatch=hatches[i],
            edgecolor="black",
        )
        if i < 1:
            # SABRE
            ax.bar(
                x + i * 2 * width - 0.5 * width,
                values_mean_sabre,
                width,
                yerr=values_err_sabre,
                capsize=1.5,
                error_kw={"elinewidth": 0.6, "ecolor": "black", "capthick": 0.6},
                linewidth=0.4,
                label=labels[i],
                color=colors_sabre[i],
                # hatch=hatches[i],
                edgecolor="black",
            )

        # Multi patch
        values_mean_custom, values_err_custom = ci95_bootstrap(custom_overhead[1], df_values, mode, ks)
        values_mean_sabre, values_err_sabre = ci95_bootstrap(sabre_overhead[1], df_values, mode, ks)
        # Custom
        ax.bar(
            x + (2 + i) * width * 2 - 3.5 * width - i * width,
            values_mean_custom,
            width,
            yerr=values_err_custom,
            capsize=1.5,
            error_kw={"elinewidth": 0.6, "ecolor": "black", "capthick": 0.6},
            linewidth=0.4,
            label=labels[i] + "multi",
            color=colors_custom[i + 2],
            hatch=hatches[i],
            edgecolor="black",
        )
        if i < 1:
            # SABRE
            ax.bar(
                x + (2 + i) * width * 2 - 1.5 * width,
                values_mean_sabre,
                width,
                yerr=values_err_sabre,
                capsize=1.5,
                error_kw={"elinewidth": 0.6, "ecolor": "black", "capthick": 0.6},
                linewidth=0.4,
                label=labels[i] + "multi",
                color=colors_sabre[i + 2],
                # hatch=hatches[i + 2],
                edgecolor="black",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(df) for df in df_values])
    ax.set_xlabel("#Defective qubits")
    ax.set_ylabel("#2q gate overhead")
    # ax.legend(loc='upper left')
    # ax.set_ylim(0, 5500)

    ax.text(0.5, 1.03, "b) #2q gates", transform=ax.transAxes, fontweight="bold", ha="center", va="bottom")

    ax.text(
        0.5,
        1.16,
        "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
        ha="center",
        va="bottom",
    )

    fig.subplots_adjust(**MARGINS)
    fig.savefig(f"{filename}_overhead.pdf", format="pdf")
    plt.close(fig)

    # Backend Utilization
    # fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.6, WIDTH_FIGSIZE*0.7))
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))

    for i, mode in enumerate(placement_modes):
        # Single patch
        values_mean_custom, values_err_custom = ci95_bootstrap(custom_utilization[0], df_values, mode, ks)
        values_mean_sabre, values_err_sabre = ci95_bootstrap(sabre_utilization[0], df_values, mode, ks)

        # Custom
        ax.bar(
            x + i * 1 * width - 2.5 * width,
            values_mean_custom,
            width,
            yerr=values_err_custom,
            capsize=1.5,
            error_kw={"elinewidth": 0.6, "ecolor": "black", "capthick": 0.6},
            linewidth=0.4,
            label="Chipmunq, single patch, " + labels[i],
            color=colors_custom[i],
            hatch=hatches[i],
            edgecolor="black",
        )
        if i < 1:
            # SABRE
            ax.bar(
                x + i * 2 * width - 0.5 * width,
                values_mean_sabre,
                width,
                yerr=values_err_sabre,
                capsize=1.5,
                error_kw={"elinewidth": 0.6, "ecolor": "black", "capthick": 0.6},
                linewidth=0.4,
                label="LightSABRE, single patch",
                color=colors_sabre[i],
                # hatch=hatches[i],
                edgecolor="black",
            )

        # Multi patch
        values_mean_custom, values_err_custom = ci95_bootstrap(custom_utilization[1], df_values, mode, ks)
        values_mean_sabre, values_err_sabre = ci95_bootstrap(sabre_utilization[1], df_values, mode, ks)
        # Custom
        ax.bar(
            x + (2 + i) * width * 2 - 3.5 * width - i * width,
            values_mean_custom,
            width,
            yerr=values_err_custom,
            capsize=1.5,
            error_kw={"elinewidth": 0.6, "ecolor": "black", "capthick": 0.6},
            linewidth=0.4,
            label="Chipmunq, multi patch, " + labels[i],
            color=colors_custom[i + 2],
            hatch=hatches[i],
            edgecolor="black",
        )

        if i < 1:
            # SABRE
            ax.bar(
                x + (2 + i) * width * 2 - 1.5 * width,
                values_mean_sabre,
                width,
                yerr=values_err_sabre,
                capsize=1.5,
                error_kw={"elinewidth": 0.6, "ecolor": "black", "capthick": 0.6},
                linewidth=0.4,
                label="LightSABRE, multi patch",
                color=colors_sabre[i + 2],
                # hatch=hatches[i + 2],
                edgecolor="black",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(df) for df in df_values])
    ax.set_xlabel("#Defective qubits")
    ax.set_ylabel("Utilization")
    # ax.legend(loc='upper left')
    ax.set_ylim(0, 1.2)

    ax.text(0.0, 1.03, "c) Chiplet utilization", transform=ax.transAxes, fontweight="bold", ha="left", va="bottom")

    ax.text(
        0.5,
        1.16,
        "Higher is better ↑",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
        ha="center",
        va="bottom",
    )

    fig.subplots_adjust(**MARGINS)
    fig.savefig(f"{filename}_utilization.pdf", format="pdf")
    plt.close(fig)
    # The legends are written by save_legend() (a-d) and save_deformation_legend()


def calculate_qpu_utilization(circuit, backend):

    # Iterate over circuit
    num_qubits_per_chiplet = backend.n * backend.m
    utilized_chiplets = set()

    num_qubits = backend.num_qubits
    cmap = backend.coupling_map

    qubits = []
    qubit_labels = [""] * num_qubits

    bit_locations = {
        bit: {"register": register, "index": index}
        for register in circuit._layout.initial_layout.get_registers()
        for index, bit in enumerate(register)
    }
    for index, qubit in enumerate(circuit._layout.initial_layout.get_virtual_bits()):
        if qubit not in bit_locations:
            bit_locations[qubit] = {"register": None, "index": index}

    for key, val in circuit._layout.initial_layout.get_virtual_bits().items():
        bit_register = bit_locations[key]["register"]
        if bit_register is None or bit_register.name != "ancilla":
            qubits.append(val)
            qubit_labels[val] = str(bit_locations[key]["index"])

    utilized_qubits = 0
    for qubit in qubits:
        if qubit != "":
            utilized_chiplets.add(backend.node_to_chiplet[int(qubit)])
            utilized_qubits += 1

    print(utilized_chiplets)
    print(f"Calculated utilization of {utilized_qubits / (len(utilized_chiplets) * num_qubits_per_chiplet)}")
    return utilized_qubits / (len(utilized_chiplets) * num_qubits_per_chiplet)


def recursive_dict() -> defaultdict:
    return defaultdict(recursive_dict)


def _num_2q_gates(circuit):
    ops = circuit.count_ops()
    return sum(ops.get(g, 0) for g in ("cx", "cz", "swap"))


def _defective_job(job):
    """One (compiler, backend config, placement, #defects, run) compilation of Fig. 10 a-c. Module level so it
    runs in a spawned worker; the tqec circuit is generated once by the parent (concurrent tqec generation
    writes the same temporary files). Retries backend seeds until the placement succeeds, as before."""
    comp, bc, pp, ks, df, run, circuit_str, partitions, size, n_inter, ps_inter = job
    qc = StimCodeCircuit(stim_circuit=stim.Circuit(circuit_str)).qc
    ic = 0
    while True:
        try:
            backend = BackendChipletV2(
                size=size, n_inter=n_inter, connectivity="nn", topology="rotated_grid",
                inter_chiplet_noise=ps_inter, inter_chiplet_amplification=1, inter_chiplet_noise_type="constant",
                num_defective_qubits=df, rng_seed=run + 42 + ic, sabre_defective=comp == "sabre",
            )
            if comp == "sabre":
                out = sabre_transpilation(qc, backend)
            else:
                out = custom_cost_transpilation(qc, backend, pre_defined_partitions=partitions,
                                                patch_initialization=pp)
            break
        except Exception:
            ic += 1  # placement failed for these defects: retry with the next backend seed
    return ((comp, bc, pp, ks, df, run),
            (calculate_qpu_utilization(out, backend), out.depth() - qc.depth(), _num_2q_gates(out) - _num_2q_gates(qc)))


def _capped_pool(jobs: int):
    """Spawn pool whose workers use one thread each (no oversubscription by Qiskit's / numpy's thread pools)."""
    for var in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    return multiprocessing.get_context("spawn").Pool(jobs)


def run_exp_defective(reproduce: bool = False, jobs: int | None = None) -> None:

    # Backend configuration
    num_inter_chiplet_connections = 8
    ps_inter = 1e-4

    # Compilation configuration
    compilation = ["sabre", "custom"]

    # Placement location of patches on a
    patch_placement = ["center", "size_aware"]

    # Run for two backend configuration:
    # - Backend fits a single patch
    # - Backend fits multiple patches
    backend_config = ["single_patch", "multi_patch"]
    chiplet_dims = {"single_patch": (15, 8), "multi_patch": (23, 14)}

    # Number of defective qubits
    defective_qubits = [0, 1, 2, 3]

    # Number of iterations per configuration
    num_iterations = 10

    # Backend size
    num_dupl = 1

    # Size of surface code
    code_size = [2]

    if reproduce:
        # Generate every circuit once, then compile all configurations in parallel. LightSABRE has no
        # placement modes: it is compiled once and stored under both placement keys (as before, a-c only
        # plot the first one), which halves its compilations.
        sources = {}
        for ks in code_size:
            circuit, partitions = get_tqec_cnot_rotated(distance_scale=ks, n1=1, n2=0)
            sources[ks] = (str(circuit), partitions)
        job_list = []
        for comp in compilation:
            for bc in backend_config:
                size = (num_dupl * 6, num_dupl * 6, *chiplet_dims[bc])
                for pp in (patch_placement[:1] if comp == "sabre" else patch_placement):
                    for ks in code_size:
                        for df in defective_qubits:
                            for run in range(num_iterations):
                                job_list.append((comp, bc, pp, ks, df, run, *sources[ks], size,
                                                 num_inter_chiplet_connections, ps_inter))
        jobs = jobs or os.cpu_count() or 1
        print(f"Fig. 10 a-c: {len(job_list)} compilations on {jobs} workers", flush=True)
        results = {}
        with _capped_pool(jobs) as pool:
            for i, (key, val) in enumerate(pool.imap_unordered(_defective_job, job_list, chunksize=1), 1):
                results[key] = val
                if i % max(1, len(job_list) // 10) == 0 or i == len(job_list):
                    print(f"  {i}/{len(job_list)} compiled", flush=True)

        output_dir = Path("experiments/evaluation/defective_qubits")
        output_dir.mkdir(parents=True, exist_ok=True)
        for comp in compilation:
            for bc in backend_config:
                custom_depth, custom_overhead, custom_utilization = recursive_dict(), recursive_dict(), recursive_dict()
                for pp in patch_placement:
                    src_pp = patch_placement[0] if comp == "sabre" else pp
                    for ks in code_size:
                        for df in defective_qubits:
                            for run in range(num_iterations):
                                util, depth, ovh = results[(comp, bc, src_pp, ks, df, run)]
                                custom_utilization[pp][df][ks][run] = util
                                custom_depth[pp][df][ks][run] = depth
                                custom_overhead[pp][df][ks][run] = ovh
                with open(output_dir / f"{comp}_depth_{bc}.pkl", "wb") as f:
                    pickle.dump(custom_depth, f)
                with open(output_dir / f"{comp}_overhead_{bc}.pkl", "wb") as f:
                    pickle.dump(custom_overhead, f)
                with open(output_dir / f"{comp}_utilization_{bc}.pkl", "wb") as f:
                    pickle.dump(custom_utilization, f)

    # Generate plots
    custom_depth_combined = []
    custom_overhead_combined = []
    custom_utilization_combined = []
    sabre_depth_combined = []
    sabre_overhead_combined = []
    sabre_utilization_combined = []

    for bc in backend_config:
        # SABRE
        with open(f"experiments/evaluation/defective_qubits/sabre_depth_{bc}.pkl", "rb") as f:
            sabre_depth = pickle.load(f)
        with open(f"experiments/evaluation/defective_qubits/sabre_overhead_{bc}.pkl", "rb") as f:
            sabre_overhead = pickle.load(f)
        with open(f"experiments/evaluation/defective_qubits/sabre_utilization_{bc}.pkl", "rb") as f:
            sabre_utilization = pickle.load(f)

        sabre_depth_combined.append(sabre_depth)
        sabre_overhead_combined.append(sabre_overhead)
        sabre_utilization_combined.append(sabre_utilization)

        # Custom
        with open(f"experiments/evaluation/defective_qubits/custom_depth_{bc}.pkl", "rb") as f:
            custom_depth = pickle.load(f)
        with open(f"experiments/evaluation/defective_qubits/custom_overhead_{bc}.pkl", "rb") as f:
            custom_overhead = pickle.load(f)
        with open(f"experiments/evaluation/defective_qubits/custom_utilization_{bc}.pkl", "rb") as f:
            custom_utilization = pickle.load(f)

        custom_depth_combined.append(custom_depth)
        custom_overhead_combined.append(custom_overhead)
        custom_utilization_combined.append(custom_utilization)

    plot_combined_backends(
        custom_depth_combined,
        custom_overhead_combined,
        custom_utilization_combined,
        sabre_depth_combined,
        sabre_overhead_combined,
        sabre_utilization_combined,
        filename="experiments/evaluation/defective_qubits/combined_overhead",
    )


# ######################################################################################
# Part 2: defect avoidance (Chipmunq) vs. patch deformation (Lin et al. [1]) (c: own figure and legend)
# ######################################################################################
# ======================================================================================
# Parameters
# ======================================================================================
D_TARGET = 5                          # distance requested by the program (Fig. 10 a-c use d = 5)
L_VALUES = [D_TARGET, D_TARGET + 2]   # tight chiplet / chiplet with slack
DEFECTS = [0, 1, 2, 3, 5, 8]          # defective qubits per chiplet (0-3 as in Fig. 10 a-c, plus higher)
N_SAMPLES = 20                        # random defect placements per (L, k)
P_PHYS = 1e-3                         # 2q gate error; 1q = 0.8 p, readout = 8/15 p (noise model of [1])
MAX_SHOTS, MAX_ERRORS, BATCH = DEFAULT_MAX_SHOTS, 100, 50_000
OUT_DIR = Path("experiments/evaluation/defective_qubits")
DEFORM_PKL = OUT_DIR / "defect_deformation.pkl"

# ======================================================================================
# Geometry (coordinates of [2]: data at (2x, 2y), syndromes at (2x+1, 2y+1))
# ======================================================================================
def patch_coords(n: int) -> set[tuple[int, int]]:
    """All qubit coordinates of an n x n rotated surface-code patch in the convention of [2] (2n^2 - 1)."""
    s = {(2 * x, 2 * y) for x in range(n) for y in range(n)}
    for x in range(-1, n):
        for y in range(-1, n):
            if (x + y) % 2 == 1 and x not in (-1, n - 1):      # X syndrome
                s.add((2 * x + 1, 2 * y + 1))
            elif (x + y) % 2 == 0 and y not in (-1, n - 1):    # Z syndrome
                s.add((2 * x + 1, 2 * y + 1))
    assert len(s) == 2 * n * n - 1
    return s


def find_window(chip: set, defects: set, L: int, D: int):
    """Chipmunq-style placement: offset (a, b) of a defect-free D x D window inside the L x L chiplet,
    closest to the centre first; None if the patch cannot be placed without touching a defect."""
    base = patch_coords(D)
    c = (L - D) / 2
    offsets = sorted(((a, b) for a in range(L - D + 1) for b in range(L - D + 1)),
                     key=lambda ab: (abs(ab[0] - c) + abs(ab[1] - c), ab))
    for a, b in offsets:
        win = {(u + 2 * a, v + 2 * b) for u, v in base}
        if win <= chip and not (win & defects):
            return a, b
    return None


def _LogicalQubit():
    """Lin et al.'s LogicalQubit [2], imported only when the deformation experiment is run, so Fig. 10 a-c
    (and plotting d, e) work without their code installed."""
    from surface_general_defect import LogicalQubit
    return LogicalQubit


def noise_args(p: float):
    return dict(readout_err=8 / 15 * p, gate1_err=0.8 * p, gate2_err=p)


def deformed_patch(L: int, defects: set, p: float):
    """Lin et al.: deform the L x L chiplet around the defects. Returns None if no valid patch exists."""
    try:
        lq = _LogicalQubit()(L, **noise_args(p), missing_coords=sorted(defects), get_metrics=True)
    except (RuntimeError, AssertionError, KeyError, IndexError):
        return None
    if lq.is_percolated() or lq.terminated_due_to_qubit_loss():
        return None
    return lq


# ======================================================================================
# Sampling
# ======================================================================================
def _sample(task):
    """Sample one circuit until MAX_ERRORS logical errors or MAX_SHOTS shots. Returns (errors, shots)."""
    circ_str, seed, max_shots, max_errors, batch = task
    circ = stim.Circuit(circ_str)
    dem = circ.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = circ.compile_detector_sampler(seed=seed)
    errors = shots = 0
    while shots < max_shots and errors < max_errors:
        n = min(batch, max_shots - shots)
        det, obs = sampler.sample(n, separate_observables=True)
        errors += int(np.any(matcher.decode_batch(det) != obs, axis=1).sum())
        shots += n
    return errors, shots


def per_cycle(errors: int, shots: int, cycles: int) -> float:
    """LER per QEC cycle; zero observed errors -> 0.5/shots (plotted as an upper bound)."""
    P = max(errors, 0.5) / shots
    return 1 - (1 - P) ** (1 / cycles)


def run_deformation(jobs: int, quick: bool, max_shots: int | None = None) -> dict:
    n_samples = 4 if quick else N_SAMPLES
    max_shots = max_shots or (20_000 if quick else MAX_SHOTS)
    rounds = D_TARGET

    # --- Build all circuits (cheap, serial) ------------------------------------------
    rows, circuits = [], {}  # circuits: circuit string -> index into the sampling task list

    def register(circ: stim.Circuit) -> int:
        s = str(circ)
        if s not in circuits:
            circuits[s] = len(circuits)
        return circuits[s]

    # Chipmunq: an intact, defect-free D x D patch (identical circuit for every successful placement)
    lq_ref = _LogicalQubit()(D_TARGET, **noise_args(P_PHYS), get_metrics=True)
    ref_idx = register(lq_ref.generate_stim(rounds))
    ref_cycles = rounds
    ref_qubits = 2 * D_TARGET ** 2 - 1

    for L in L_VALUES:
        chip = sorted(patch_coords(L))
        for k in DEFECTS:
            for s in range(n_samples):
                rng = np.random.default_rng(10_000 * L + 100 * k + s)
                idx = rng.choice(len(chip), size=k, replace=False) if k else []
                defects = {chip[i] for i in idx}

                # Chipmunq: avoid the defects with an intact patch
                win = find_window(set(chip), defects, L, D_TARGET)
                rows.append(dict(method="chipmunq", L=L, k=k, sample=s, ok=win is not None,
                                 d_eff=D_TARGET if win is not None else 0,
                                 util=ref_qubits / len(chip) if win is not None else 0.0,
                                 circ=ref_idx if win is not None else None, cycles=ref_cycles))

                # Lin et al.: deform the whole chiplet around the defects
                lq = deformed_patch(L, defects, P_PHYS)
                if lq is None:
                    rows.append(dict(method="deform", L=L, k=k, sample=s, ok=False, d_eff=0, util=0.0,
                                     circ=None, cycles=None))
                else:
                    cycles = rounds * (2 if len(lq.x_gauges) > 0 else 1)
                    rows.append(dict(method="deform", L=L, k=k, sample=s, ok=True,
                                     d_eff=min(lq.actual_distance_vertical(), lq.actual_distance_horizontal()),
                                     util=len(lq.all_qubits) / len(chip),
                                     circ=register(lq.generate_stim(rounds)), cycles=cycles))

    # --- Sample all distinct circuits in parallel ----------------------------------------
    circ_list = sorted(circuits, key=circuits.get)
    seeds = np.random.SeedSequence(994).generate_state(len(circ_list), dtype=np.uint32)
    tasks = [(c, int(seed), max_shots, MAX_ERRORS, BATCH) for c, seed in zip(circ_list, seeds)]
    print(f"{len(rows)} (method, L, k, sample) points, {len(tasks)} distinct circuits on {jobs} workers, "
          f"<= {max_shots} shots each", flush=True)
    t0 = time.time()
    for var in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    with multiprocessing.get_context("spawn").Pool(jobs) as pool:
        sampled = []
        for i, out in enumerate(pool.imap(_sample, tasks, chunksize=1), 1):
            sampled.append(out)
            if i % max(1, len(tasks) // 10) == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} circuits sampled ({time.time() - t0:.0f} s)", flush=True)

    for r in rows:
        if r["ok"]:
            e, n = sampled[r["circ"]]
            r.update(errors=e, shots=n, ler=per_cycle(e, n, r["cycles"]))
        else:
            r.update(errors=None, shots=None, ler=None)
        r.pop("circ")

    data = dict(rows=rows, params=dict(D=D_TARGET, L=L_VALUES, defects=DEFECTS, samples=n_samples,
                                       p=P_PHYS, rounds=rounds, max_shots=max_shots, max_errors=MAX_ERRORS))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(DEFORM_PKL, "wb") as f:
        pickle.dump(data, f)
    return data


# ======================================================================================
# Summary and plots
# ======================================================================================
def _group(rows, method, L, k):
    return [r for r in rows if r["method"] == method and r["L"] == L and r["k"] == k]


def summarize_deformation(rows):
    print(f"{'method':10s} {'L':>3s} {'k':>3s} {'success':>8s} {'d_eff':>6s} {'util':>6s} {'LER/cycle':>10s}")
    for L in L_VALUES:
        for method in ("chipmunq", "deform"):
            for k in DEFECTS:
                g = _group(rows, method, L, k)
                ok = [r for r in g if r["ok"]]
                succ = len(ok) / len(g)
                if ok:
                    print(f"{method:10s} {L:3d} {k:3d} {succ:8.0%} {np.mean([r['d_eff'] for r in ok]):6.2f} "
                          f"{np.mean([r['util'] for r in ok]):6.2f} {np.mean([r['ler'] for r in ok]):10.2e}")
                else:
                    print(f"{method:10s} {L:3d} {k:3d} {succ:8.0%} {'-':>6s} {'-':>6s} {'-':>10s}")


def _ci95(vals, n_boot=5000, seed=0):
    vals = np.asarray(vals)
    mean = vals.mean()
    if len(vals) < 2:
        return mean, 0.0, 0.0
    rng = np.random.default_rng(seed)
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return mean, mean - np.percentile(boots, 2.5), np.percentile(boots, 97.5) - mean


# (method, L index, label, colour, hatch). Tight: L = D, slack: L = D + 2 (explain in the caption).
# Chipmunq uses purple here: the blues/greys of a-c already stand for Chipmunq's single/multi-patch placements,
# and all panels share one legend. Labels are short so the legend fits the page width in two rows.
SERIES = [
    ("chipmunq", 0, "Avoiding, tight", "#C9BEE6", "++"),
    ("deform", 0, "Deform, tight", "#B5D8B0", "\\\\"),
    ("chipmunq", 1, "Avoiding, slack", "#8E7CC3", "xx"),
    ("deform", 1, "Deform, slack", "#6AA84F", "xx"),
]


def _fonts():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": FONT_PT,
        "axes.labelsize": FONT_PT,
        "axes.titlesize": FONT_PT,
        "legend.fontsize": FONT_PT,
        "xtick.labelsize": FONT_PT - 1,
        "ytick.labelsize": FONT_PT - 1,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "ytick.minor.size": 1.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.labelpad": 2,
        "axes.linewidth": 0.6,
        "hatch.linewidth": 0.4,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "errorbar.capsize": 1.5,
    })


def _panel():
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    fig.subplots_adjust(**MARGINS)
    return fig, ax


def _title(ax, text: str, better: str):
    """Panel title (left-aligned to the axes) with the "better" hint on a second line above it."""
    ax.text(0.0, 1.03, text, transform=ax.transAxes, fontweight="bold", ha="left", va="bottom")
    ax.text(0.5, 1.16, better, transform=ax.transAxes, fontweight="bold", color=plot_lib_color,
            ha="center", va="bottom")


def plot_deformation(data, filename=str(OUT_DIR / "combined_overhead")):
    rows, Ls = data["rows"], data["params"]["L"]
    _fonts()

    # ---------------- d) LER per QEC cycle vs #defects ----------------
    fig, ax = _panel()
    x = np.arange(len(DEFECTS))
    width = 0.84 / len(SERIES)
    all_ok = [r["ler"] for r in rows if r["ok"]]
    floor = min(all_ok) / 3 if all_ok else 1e-7
    for j, (method, li, label, color, hatch) in enumerate(SERIES):
        L = Ls[li]
        xpos = x + (j - (len(SERIES) - 1) / 2) * width
        for i, k in enumerate(DEFECTS):
            g = _group(rows, method, L, k)
            ok = [r["ler"] for r in g if r["ok"]]
            if ok:
                m, lo, hi = _ci95(ok)
                ax.bar(xpos[i], m, width, yerr=[[lo], [hi]], color=color, hatch=hatch, edgecolor="black",
                       linewidth=0.4, error_kw={"elinewidth": 0.6, "ecolor": "black", "capthick": 0.6})
            else:  # no chiplet could host the patch: hollow dashed bar
                ax.bar(xpos[i], floor * 3, width, bottom=floor, color="white", hatch="xxx",
                       edgecolor=color, linewidth=0.8, linestyle="--")
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor)
    ax.set_xticks(x, [str(k) for k in DEFECTS])
    ax.set_xlabel("#Defective qubits")
    ax.set_ylabel("LER")  # per QEC cycle (state in the caption)
    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    _title(ax, "c) Avoid vs. deform", "Lower is better ↓")
    fig.savefig(f"{filename}_deformation_ler.pdf", format="pdf")
    plt.close(fig)

    # ---------------- e) tradeoff: utilization vs LER ----------------
    fig, ax = _panel()
    markers = {0: "o", 1: "s"}
    for method, li, label, color, hatch in SERIES:
        L = Ls[li]
        pts = []
        for k in DEFECTS:
            ok = [r for r in _group(rows, method, L, k) if r["ok"]]
            if ok:
                pts.append((np.mean([r["util"] for r in ok]), np.mean([r["ler"] for r in ok]), k))
        if not pts:
            continue
        u, l, ks = zip(*pts)
        ax.plot(u, l, marker=markers[li], color=color, markeredgecolor="black", markeredgewidth=0.5)
        for ui, lk, kk in pts:
            ax.annotate(str(kk), xy=(ui, lk), xytext=(2, 2), textcoords="offset points", fontsize=FONT_PT - 2.5)
    ax.set_yscale("log")
    ax.set_xlabel("Utilization")  # active code qubits / chiplet qubits (state in the caption)
    ax.set_ylabel("LER")  # per QEC cycle (state in the caption)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
    _title(ax, "f) Utilization tradeoff", "Lower, right is better")
    fig.savefig(f"{filename}_deformation_tradeoff.pdf", format="pdf")
    plt.close(fig)
    # The legends are written by save_legend() (a-d) and save_deformation_legend()


# ######################################################################################
# Part 3: placement strategy (center / size-aware) vs. LER (Fig. 10 d)
# ######################################################################################
PLACEMENT_KS = [1, 2, 3]                       # d = 2k + 1 -> 3, 5, 7
PLACEMENT_DEFECTS = [0, 1, 3]                  # defective qubits per chiplet (0 = compilation overhead only)
PLACEMENT_PS = list(np.logspace(-3, -2, 6))    # 1e-3 ... 1e-2, 6 points
PLACEMENT_SEEDS = 4                            # defect placements (backends) per (d, #defects)
PLACEMENT_MAX_SHOTS, PLACEMENT_MAX_ERRORS = DEFAULT_MAX_SHOTS, 500
PLACEMENT_P_INTER = 1e-4                       # inter-chiplet noise, as in a-c
PLACEMENT_MAX_BACKEND_RETRIES = 20
PLACEMENT_PKL = OUT_DIR / "placement_ler.pkl"
PLACEMENT_MODES = ["center", "size_aware"]
# Backends: single patch (one patch per chiplet) and multi patch (several patches share a chiplet), as in a-c.
# Colours per backend match a-c: single patch = blues, multi patch = greys (same legend entries).
PLACEMENT_BACKENDS = ["single_patch", "multi_patch"]
PLACEMENT_COLORS = {
    "single_patch": {"center": COLORS_CUSTOM[0], "size_aware": COLORS_CUSTOM[1]},
    "multi_patch": {"center": COLORS_CUSTOM[2], "size_aware": COLORS_CUSTOM[3]},
}
# Output file and panel title per backend (single patch keeps the file name of the previous version)
PLACEMENT_FILES = {"single_patch": "_placement_ler.pdf", "multi_patch": "_placement_ler_multi.pdf"}
PLACEMENT_TITLES = {"single_patch": "d) Mapping effect on LER", "multi_patch": "d) Mapping effect on LER"}
PLACEMENT_MARKERS = {3: "v", 5: "o", 7: "s", 9: "D"}
PLACEMENT_DEFECT_LS = {0: ":", 1: "-", 2: "-.", 3: "--", 5: (0, (3, 1, 1, 1))}
IDEAL_COLOR = "black"  # untranspiled reference
IDEAL_LS = (0, (4, 1.5, 1, 1.5, 1, 1.5))  # dash-dot-dot: distinct from the #defects line styles


def _chiplet_dims(k: int, bc: str) -> tuple[int, int]:
    """Chiplet rows x cols for distance scale k (d = 2k + 1).

    single patch: (11 + 4(k-1)) x (6 + 2(k-1)) -> 15 x 8 at d = 5 (a-c), fits one patch.
    multi patch:  single + (4k, 2k + 2)        -> 23 x 14 at d = 5 (a-c); the extra rows/columns scale
                  with d, so several patches fit on one chiplet at every distance.
    """
    rows, cols = 11 + 4 * (k - 1), 6 + 2 * (k - 1)
    if bc == "multi_patch":
        rows, cols = rows + 4 * k, cols + 2 * k + 2
    return rows, cols


def _placement_backend(k: int, n_defects: int, seed: int, bc: str = "single_patch") -> BackendChipletV2:
    """Chiplet backend for distance scale k with one (single_patch) or several (multi_patch) patches per chiplet."""
    return BackendChipletV2(
        size=(6, 6, *_chiplet_dims(k, bc)),
        n_inter=2 * k + 3,
        connectivity="nn",
        topology="rotated_grid",
        inter_chiplet_noise=PLACEMENT_P_INTER,
        inter_chiplet_amplification=1,
        inter_chiplet_noise_type="constant",
        num_defective_qubits=n_defects,
        rng_seed=seed,
    )


def _placement_job(job):
    """Compile one (backend, d, #defects, defect placement) with both strategies on the same backend and build
    the noisy circuits for every physical error rate. Module level: runs in a spawned worker. If both strategies
    give the identical circuit, it is returned once (simulated once, counted for both)."""
    bc, k, n_def, s, circuit_str, partitions, ps = job
    qc = StimCodeCircuit(stim_circuit=stim.Circuit(circuit_str)).qc
    retries = 0
    for attempt in range(PLACEMENT_MAX_BACKEND_RETRIES):
        seed = 1000 * s + 42 + attempt
        backend = _placement_backend(k, n_def, seed, bc)
        try:
            compiled = {pp: str(get_stim_circuits_with_detectors(custom_cost_transpilation(
                qc, backend, pre_defined_partitions=partitions, patch_initialization=pp))[0][0])
                for pp in PLACEMENT_MODES}
            break
        except Exception:
            retries += 1
    else:
        raise RuntimeError(f"No {bc} backend for d={2 * k + 1} with {n_def} defects could be compiled")
    groups = {}  # circuit -> placements producing it
    for pp, circ in compiled.items():
        groups.setdefault(circ, []).append(pp)
    out = []
    for circ, pps in groups.items():
        c = stim.Circuit(circ)
        for p in ps:
            noisy = get_noise_model("modsi1000", None, p, None, remote=backend.inter_chiplet_connections).noisy_circuit(c)
            out.append((pps, p, str(noisy)))
    return dict(bc=bc, k=k, n_def=n_def, s=s, seed=seed, retries=retries, circuits=out)


def _stat_backend(st) -> str:
    """Backend of a sinter stat; results from before the multi-patch option are single-patch."""
    return st.json_metadata.get("backend", "single_patch")


def run_placement(quick: bool = False, jobs: int | None = None, backends=None,
                  max_shots: int | None = None, defects=None) -> list:
    """Compile every (backend, d, #defects, defect placement) with both placement strategies on the same backend
    (in parallel, noisy circuits built in the workers) and simulate the LER. A backend seed is only used if both
    strategies can place all patches; identical circuits of the two strategies are simulated once.

    Saved results of other (backend, #defects) combinations are kept, so backends and defect counts can be run
    separately (e.g. only the 0-defect baseline)."""
    backends = list(backends or PLACEMENT_BACKENDS)
    defects = list(PLACEMENT_DEFECTS if defects is None else defects)
    ks = [1] if quick else PLACEMENT_KS
    n_seeds = 1 if quick else PLACEMENT_SEEDS
    ps = [2e-3, 5e-3] if quick else PLACEMENT_PS
    max_errors = 100 if quick else PLACEMENT_MAX_ERRORS
    max_shots = max_shots or (20_000 if quick else PLACEMENT_MAX_SHOTS)

    sources = {}
    for k in ks:
        circuit, partitions = get_tqec_cnot_rotated(distance_scale=k, n1=1, n2=0)
        sources[k] = (str(circuit), partitions)
    # Without defects all seeds give the same backend: one sample is enough for the baseline
    job_list = [(bc, k, n_def, s, *sources[k], ps) for bc in backends for k in ks
                for n_def in defects for s in range(n_seeds if n_def > 0 else 1)]
    jobs = jobs or os.cpu_count() or 1
    print(f"Fig. 10 d ({', '.join(backends)}; defects {defects}): {len(job_list)} backends "
          f"(x {len(PLACEMENT_MODES)} placements) "
          f"on {min(jobs, len(job_list))} workers, <= {max_shots} shots per circuit", flush=True)
    tasks, shared = [], 0
    with _capped_pool(max(1, min(jobs, len(job_list)))) as pool:
        for r in pool.imap_unordered(_placement_job, job_list, chunksize=1):
            d = 2 * r["k"] + 1
            print(f"[{r['bc']}, d={d}, {r['n_def']} defects, sample {r['s']}] backend seed {r['seed']} "
                  f"({r['retries']} rejected)", flush=True)
            for pps, p, noisy in r["circuits"]:
                shared += len(pps) > 1
                tasks.append(sinter.Task(circuit=stim.Circuit(noisy), json_metadata={
                    "backend": r["bc"], "placements": pps, "placement": pps[0], "d": d, "defects": r["n_def"],
                    "sample": r["s"], "p": p, "backend_seed": r["seed"]}))
    print(f"{len(tasks)} sinter tasks ({shared} shared by both placements: identical circuits)", flush=True)

    # Untranspiled reference (baseline of the relative LER): the original CNOT circuit with the same noise model,
    # without routing and without inter-chiplet links. Independent of backend, placement and defects, so it is
    # only simulated for (d, p) points that are not in the saved results yet.
    old = []
    if PLACEMENT_PKL.exists():
        with open(PLACEMENT_PKL, "rb") as f:
            old = pickle.load(f)
    have_ideal = {(st.json_metadata["d"], st.json_metadata["p"]) for st in old if _stat_backend(st) == "ideal"}
    n_ideal = 0
    for k in ks:
        d = 2 * k + 1
        ideal = get_stim_circuits_with_detectors(StimCodeCircuit(stim_circuit=stim.Circuit(sources[k][0])).qc)[0][0]
        for p in ps:
            if (d, p) in have_ideal:
                continue
            noisy = get_noise_model("modsi1000", None, p, None, remote=[]).noisy_circuit(ideal)
            tasks.append(sinter.Task(circuit=noisy, json_metadata={
                "backend": "ideal", "placements": ["ideal"], "placement": "ideal", "d": d, "defects": -1,
                "sample": 0, "p": p, "backend_seed": None}))
            n_ideal += 1
    print(f"{n_ideal} untranspiled reference tasks", flush=True)

    stats = sinter.collect(num_workers=jobs, tasks=tasks, max_shots=max_shots,
                           max_errors=max_errors, decoders=["pymatching"], print_progress=True)
    stats = list(stats)

    # Keep the saved results of the (backend, #defects) combinations that were not rerun (and the untranspiled
    # reference, which is only ever added for missing (d, p) points)
    if old:
        kept = [st for st in old
                if not (_stat_backend(st) in backends and st.json_metadata["defects"] in defects)]
        if kept:
            combos = sorted({(_stat_backend(st), st.json_metadata["defects"]) for st in kept})
            print(f"Keeping {len(kept)} saved results of " + ", ".join(f"{b}/{n} defects" for b, n in combos),
                  flush=True)
        stats = kept + stats
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PLACEMENT_PKL, "wb") as f:
        pickle.dump(stats, f)
    return stats


def _placement_ler(stats, bc: str) -> dict:
    """{(placement, d, #defects, p): [LER per defect placement]} for one backend."""
    ler = defaultdict(list)
    for st in stats:
        if _stat_backend(st) != bc:
            continue
        n = st.shots - st.discards
        if n > 0:
            m = st.json_metadata
            for pp in m.get("placements", [m["placement"]]):  # identical circuits count for both placements
                ler[(pp, m["d"], m["defects"], m["p"])].append(st.errors / n)
    return ler


def _centered_title(fig, ax, text: str, y: float = 1.03, margin_in: float = 0.03):
    """Bold panel title centred over the axes, shifted just enough to stay inside the figure when it is wider
    than the axes (the 1.75 in panels leave little room on the right)."""
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(fig.transFigure, ax.transAxes)
    pos = ax.get_position()
    t = ax.text((pos.x0 + pos.x1) / 2, y, text, transform=trans, fontweight="bold", ha="center", va="bottom")
    fig.canvas.draw()
    half = t.get_window_extent().width / fig.bbox.width / 2   # half title width, figure fraction
    margin = margin_in / fig.get_figwidth()
    t.set_x(min(max(t.get_position()[0], half + margin), 1 - half - margin))
    return t


def plot_placement(stats, filename=str(OUT_DIR / "combined_overhead"), backends=None) -> None:
    """d) LER relative to the untranspiled circuit vs. physical error rate, one panel per backend:
    colour = placement, marker = code distance, line style = #defects (0 defects = compilation overhead only).

    Each point is mean LER(placement, d, #defects, p) over the defect placements divided by the LER of the
    untranspiled CNOT circuit (same d, p and noise model, no routing / inter-chiplet links); 1 = no LER increase
    due to compilation and defects. Points without observed errors are not shown. The absolute LER, including the
    untranspiled reference in black, is also written (file name with "_abs")."""
    ideal = _placement_ler(stats, "ideal")
    ideal = {(d, p): np.mean(v) for (_, d, _, p), v in ideal.items()}
    available = [b for b in PLACEMENT_BACKENDS if any(_stat_backend(st) == b for st in stats)]
    for bc in (b for b in (backends or PLACEMENT_BACKENDS) if b not in available):
        print(f"No placement results for {bc}: run --part placement --placement-backends "
              f"{bc.split('_')[0]} first")
    if not ideal:
        print("No untranspiled reference in the results, so no relative LER (rerun --part placement; only the "
              "missing reference points are simulated in addition); writing the absolute LER only")
    from matplotlib.ticker import NullFormatter
    for bc in [b for b in (backends or PLACEMENT_BACKENDS) if b in available]:
        ler = _placement_ler(stats, bc)
        ps = sorted({k[3] for k in ler})
        ds = sorted({k[1] for k in ler})
        defects = sorted({k[2] for k in ler if k[2] > 0})  # 0-defect results are kept but not drawn

        for relative in ([True, False] if ideal else [False]):
            _fonts()
            fig, ax = _panel()
            for pp in PLACEMENT_MODES:
                color = PLACEMENT_COLORS[bc][pp]
                for d in ds:
                    for nd in defects:
                        pts = []
                        for p in ps:
                            vals = ler.get((pp, d, nd, p))
                            if not vals or np.mean(vals) <= 0:
                                continue
                            v = np.mean(vals)
                            if relative:
                                base = ideal.get((d, p), 0)
                                if base <= 0:
                                    continue
                                v /= base
                            pts.append((p, v))
                        if not pts:
                            continue
                        xs, ys = zip(*pts)
                        ax.plot(xs, ys, color=color, linestyle=PLACEMENT_DEFECT_LS.get(nd, "-"),
                                marker=PLACEMENT_MARKERS.get(d, "P"), markerfacecolor=color,
                                markeredgecolor="black", markeredgewidth=0.4, linewidth=1.1)
            if not relative:  # untranspiled reference
                for d in ds:
                    pts = [(p, ideal[(d, p)]) for p in ps if ideal.get((d, p), 0) > 0]
                    if pts:
                        xs, ys = zip(*pts)
                        ax.plot(xs, ys, color=IDEAL_COLOR, linestyle=IDEAL_LS, marker=PLACEMENT_MARKERS.get(d, "P"),
                                markerfacecolor="white", markeredgecolor=IDEAL_COLOR, markeredgewidth=0.6,
                                linewidth=0.8, zorder=5)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(min(PLACEMENT_PS) / 1.15, max(PLACEMENT_PS) * 1.15)
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_minor_formatter(NullFormatter())  # no overlapping labels on ranges below one decade
            ax.set_xlabel("Physical error rate")
            if relative:
                ax.axhline(1.0, color=IDEAL_COLOR, linewidth=0.6, linestyle=":", zorder=1)  # = untranspiled LER
                ax.set_ylabel("LER / LER$_{\\mathrm{untranspiled}}$")
            else:
                ax.set_ylabel("LER")  # per logical CNOT (state in the caption)
            ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
            _centered_title(fig, ax, PLACEMENT_TITLES[bc])
            ax.text(0.5, 1.16, "Lower is better ↓", transform=ax.transAxes, fontweight="bold",
                    color=plot_lib_color, ha="center", va="bottom")
            out = PLACEMENT_FILES[bc] if relative else PLACEMENT_FILES[bc].replace(".pdf", "_abs.pdf")
            fig.savefig(f"{filename}{out}", format="pdf")
            plt.close(fig)


# ######################################################################################
# Layouts: mapping and routing solutions of the placement experiment (inspection plots, not a paper panel)
# ######################################################################################
# For every (backend, d, #defects) the backend of defect placement `sample` is rebuilt with exactly the seeds of
# the placement experiment (same retry rule), compiled with center and size-aware placement, and drawn side by
# side: chiplet grid, coupling edges, inter-chiplet links, defective qubits, the qubits of every patch (colour =
# partition) and the SWAP edges inserted by routing (line width ~ #SWAPs on that edge). Only compilation, no
# sampling, so this is cheap. One PDF per (backend, d, #defects) in experiments/evaluation/defective_qubits/layouts.
LAYOUT_DIR = OUT_DIR / "layouts"
LAYOUT_PKL = LAYOUT_DIR / "layouts.pkl"
LAYOUT_CHIPLET_GAP = 2  # empty grid units between neighbouring chiplets in the drawing


def _backend_defects(backend, all_nodes, edges) -> set:
    """Defective physical qubits. BackendChipletV2's attribute name is not fixed here, so try the usual names and
    fall back to qubits without any coupling edge (defective qubits are cut out of the coupling map)."""
    for attr in ("defective_qubits", "defect_qubits", "defects", "defective_nodes"):
        val = getattr(backend, attr, None)
        if val is not None and not callable(val):
            try:
                return {int(q) for q in val}
            except TypeError:
                pass
    connected = {q for e in edges for q in e}
    return {q for q in all_nodes if q not in connected}


def _layout_extract(qc, backend, partitions, size) -> dict:
    """Picklable summary of one compiled circuit: patch -> physical qubits, SWAP edge counts, depth."""
    lay = qc.layout.initial_index_layout(filter_ancillas=True)
    patches = [dict(type=p.get("type", ""), phys=sorted(int(lay[i]) for i in p["indices"])) for p in partitions]
    swaps = defaultdict(int)
    for inst in qc.data:
        if inst.operation.name == "swap":
            a, b = sorted(qc.find_bit(q).index for q in inst.qubits)
            swaps[(a, b)] += 1
    return dict(patches=patches, swaps=dict(swaps), n_swaps=int(sum(swaps.values())), depth=qc.depth(),
                twoq=_num_2q_gates(qc))


def _layout_job(job):
    """Compile one (backend, d, #defects, sample) with both placements; same seeds / retries as _placement_job."""
    bc, k, n_def, s, circuit_str, partitions = job
    qc = StimCodeCircuit(stim_circuit=stim.Circuit(circuit_str)).qc
    for attempt in range(PLACEMENT_MAX_BACKEND_RETRIES):
        seed = 1000 * s + 42 + attempt
        backend = _placement_backend(k, n_def, seed, bc)
        try:
            compiled = {pp: custom_cost_transpilation(qc, backend, pre_defined_partitions=partitions,
                                                      patch_initialization=pp) for pp in PLACEMENT_MODES}
            break
        except Exception:
            continue
    else:
        return dict(bc=bc, k=k, n_def=n_def, s=s, error="no backend seed could be compiled")
    edges = sorted({tuple(sorted(map(int, e))) for e in backend.coupling_map.get_edges() if e[0] != e[1]})
    chiplet_of = {int(q): int(c) for q, c in backend.node_to_chiplet.items()}
    return dict(bc=bc, k=k, n_def=n_def, s=s, seed=seed, size=(6, 6, *_chiplet_dims(k, bc)),
                chiplet_of=chiplet_of, edges=edges, defects=sorted(_backend_defects(backend, chiplet_of, edges)),
                layouts={pp: _layout_extract(c, backend, partitions, None) for pp, c in compiled.items()})


def run_layouts(backends=None, defects=None, sample: int = 0, jobs: int | None = None, quick: bool = False) -> list:
    backends = list(backends or PLACEMENT_BACKENDS)
    defects = list(PLACEMENT_DEFECTS if defects is None else defects)
    ks = [1] if quick else PLACEMENT_KS
    sources = {}
    for k in ks:
        circuit, partitions = get_tqec_cnot_rotated(distance_scale=k, n1=1, n2=0)
        sources[k] = (str(circuit), partitions)
    job_list = [(bc, k, n, sample, *sources[k]) for bc in backends for k in ks for n in defects]
    jobs = max(1, min(jobs or os.cpu_count() or 1, len(job_list)))
    print(f"Layouts: {len(job_list)} backends (x {len(PLACEMENT_MODES)} placements) on {jobs} workers", flush=True)
    with _capped_pool(jobs) as pool:
        out = list(pool.imap_unordered(_layout_job, job_list, chunksize=1))
    # Keep saved layouts of other configurations
    key = lambda r: (r["bc"], r["k"], r["n_def"], r["s"])  # noqa: E731
    if LAYOUT_PKL.exists():
        with open(LAYOUT_PKL, "rb") as f:
            old = {key(r): r for r in pickle.load(f)}
        old.update({key(r): r for r in out})
        out = list(old.values())
    LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LAYOUT_PKL, "wb") as f:
        pickle.dump(out, f)
    return out


def _grid_positions(r) -> dict:
    """Drawing position of every physical qubit: chiplets on their (row, col) grid, qubits inside a chiplet on a
    rows x cols grid in index order. Rotated-grid couplings are drawn on this square grid (diagonal edges)."""
    R, C, n, m = r["size"]
    by_chip = defaultdict(list)
    for q, c in r["chiplet_of"].items():
        by_chip[c].append(q)
    pos = {}
    for c, qs in by_chip.items():
        qs.sort()
        cr, cc = divmod(c, C)
        cols = m if len(qs) == n * m else math.ceil(math.sqrt(len(qs)))
        for i, q in enumerate(qs):
            lr, lc = divmod(i, cols)
            pos[q] = (cc * (m + LAYOUT_CHIPLET_GAP) + lc, -(cr * (n + LAYOUT_CHIPLET_GAP) + lr))
    return pos


def plot_layouts(results, backends=None) -> None:
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle
    LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    for r in sorted(results, key=lambda r: (r["bc"], r["k"], r["n_def"], r["s"])):
        if backends and r["bc"] not in backends:
            continue
        d = 2 * r["k"] + 1
        if "error" in r:
            print(f"[layout] {r['bc']} d={d} {r['n_def']} defects: {r['error']}")
            continue
        pos = _grid_positions(r)
        R, C, n, m = r["size"]
        chiplet_of = r["chiplet_of"]
        defects = set(r["defects"])

        # Crop to the chiplets used by either placement (+1 chiplet margin)
        used = {chiplet_of[q] for lay in r["layouts"].values() for p in lay["patches"] for q in p["phys"]}
        used |= {chiplet_of[q] for lay in r["layouts"].values() for e in lay["swaps"] for q in e}
        rows = [c // C for c in used]
        cols = [c % C for c in used]
        r0, r1 = max(min(rows) - 1, 0), min(max(rows) + 1, R - 1)
        c0, c1 = max(min(cols) - 1, 0), min(max(cols) + 1, C - 1)
        shown = {q for q, c in chiplet_of.items() if r0 <= c // C <= r1 and c0 <= c % C <= c1}

        intra = [(pos[a], pos[b]) for a, b in r["edges"]
                 if a in shown and b in shown and chiplet_of[a] == chiplet_of[b]]
        inter = [(pos[a], pos[b]) for a, b in r["edges"]
                 if a in shown and b in shown and chiplet_of[a] != chiplet_of[b]]
        w = (c1 - c0 + 1) * (m + LAYOUT_CHIPLET_GAP)
        h = (r1 - r0 + 1) * (n + LAYOUT_CHIPLET_GAP)
        scale = 7.0 / max(w * 2, 1)
        fig, axes = plt.subplots(1, 2, figsize=(14, max(3.0, h * scale + 1.0)))
        for ax, pp in zip(axes, PLACEMENT_MODES):
            lay = r["layouts"][pp]
            # chiplet outlines
            for c in {chiplet_of[q] for q in shown}:
                cr, cc = divmod(c, C)
                ax.add_patch(Rectangle((cc * (m + LAYOUT_CHIPLET_GAP) - 0.5, -(cr * (n + LAYOUT_CHIPLET_GAP) + n) + 0.5),
                                       m, n, fill=False, edgecolor="#B0B0B0", linewidth=0.6))
            ax.add_collection(LineCollection(intra, colors="#D8D8D8", linewidths=0.3, zorder=1))
            ax.add_collection(LineCollection(inter, colors="#E68A5C", linewidths=0.8, linestyles="--", zorder=2))
            xs, ys = zip(*(pos[q] for q in shown))
            ax.scatter(xs, ys, s=3, c="#C8C8C8", zorder=3, linewidths=0)
            cmap = plt.get_cmap("tab20")
            for i, p in enumerate(lay["patches"]):
                pts = [pos[q] for q in p["phys"] if q in pos]
                if pts:
                    ax.scatter(*zip(*pts), s=10, color=cmap((2 * i + (i // 10) % 2) % 20), zorder=4, linewidths=0,
                               label=f"{i}: {p['type']}")
            if lay["swaps"]:
                mx = max(lay["swaps"].values())
                segs = [(pos[a], pos[b]) for a, b in lay["swaps"]]
                lw = [0.8 + 2.2 * cnt / mx for cnt in lay["swaps"].values()]
                ax.add_collection(LineCollection(segs, colors="black", linewidths=lw, zorder=5))
            dq = [pos[q] for q in defects if q in shown]
            if dq:
                ax.scatter(*zip(*dq), marker="x", s=30, color="red", linewidths=1.2, zorder=6)
            ax.set_title(f"{pp.replace('_', '-')}: {lay['n_swaps']} SWAPs, depth {lay['depth']}, "
                         f"{lay['twoq']} 2q gates", fontsize=9)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.autoscale_view()
        handles = [Line2D([], [], color="#E68A5C", ls="--", label="inter-chiplet link"),
                   Line2D([], [], color="black", lw=2, label="SWAP edge (width ~ #SWAPs)"),
                   Line2D([], [], marker="x", color="red", ls="none", label="defective qubit"),
                   Line2D([], [], marker="o", color="#C8C8C8", ls="none", ms=3, label="unused qubit")]
        fig.legend(handles=handles, loc="lower center", ncols=4, frameon=False)
        fig.suptitle(f"{r['bc'].replace('_', ' ')}, d={d}, {r['n_def']} defective qubits "
                     f"(sample {r['s']}, backend seed {r['seed']}; colour = patch)", fontsize=10)
        fig.tight_layout(rect=(0, 0.05, 1, 0.95))
        out = LAYOUT_DIR / f"layout_{r['bc']}_d{d}_def{r['n_def']}_s{r['s']}.pdf"
        fig.savefig(out, format="pdf")
        plt.close(fig)
        print(f"[layout] {out}  center: {r['layouts']['center']['n_swaps']} SWAPs, "
              f"size-aware: {r['layouts']['size_aware']['n_swaps']} SWAPs")


# ######################################################################################
# Legends: one for panels a-d, one for the deformation comparison (avoid vs. deform)
# ######################################################################################
def _patch(label, color, hatch=""):
    return Patch(facecolor=color, hatch=hatch, edgecolor="black", linewidth=0.4, label=label)


def _blank():
    return Line2D([], [], linestyle="none", label=" ")


def _two_rows(top: list, bottom: list) -> list:
    """Order handles for a two-row legend (matplotlib fills legends column by column)."""
    n = max(len(top), len(bottom))
    top = top + [_blank() for _ in range(n - len(top))]
    bottom = bottom + [_blank() for _ in range(n - len(bottom))]
    return [h for col in zip(top, bottom) for h in col]


def save_legend(filename=str(OUT_DIR / "combined_overhead")) -> None:
    """Single legend for panels a-d, full page width, two rows.

    Columns: Chipmunq single patch (center / size-aware), Chipmunq multi patch (center / size-aware),
    LightSABRE (single / multi) for a-c; code distances (markers) and #defects (line styles) for d) -- the
    placement colours of d) are the Chipmunq colours of the same backend (single patch: blues, multi patch:
    greys). Built from the style constants only (no data), so it is written whatever part is run or replotted.
    """
    _fonts()
    grey = "#606060"
    dist = [Line2D([], [], color=grey, marker=PLACEMENT_MARKERS[2 * k + 1], linestyle="none",
                   markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.4, label=f"d={2 * k + 1}")
            for k in PLACEMENT_KS]
    defects = [Line2D([], [], color=grey, linestyle=PLACEMENT_DEFECT_LS.get(n, "-"),
                      label=f"{n} defect" + ("s" if n != 1 else "")) for n in PLACEMENT_DEFECTS if n > 0]
    top = [_patch("Chipmunq single, center", COLORS_CUSTOM[0], HATCHES[0]),
           _patch("Chipmunq multi, center", COLORS_CUSTOM[2], HATCHES[0]),
           _patch("LightSABRE single", COLORS_SABRE[0])] + dist  # first row: all code distances
    bottom = [_patch("Chipmunq single, size-aware", COLORS_CUSTOM[1], HATCHES[1]),
              _patch("Chipmunq multi, size-aware", COLORS_CUSTOM[3], HATCHES[1]),
              _patch("LightSABRE multi", COLORS_SABRE[2])] + defects  # second row: #defects
    handles = _two_rows(top, bottom)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    legend_fig = plt.figure(figsize=(TEXT_WIDTH_IN, 0.4))
    legend_fig.legend(handles=handles, loc="center", frameon=False, ncols=len(handles) // 2,
                      columnspacing=0.8, handlelength=1.6, handletextpad=0.35)
    legend_fig.savefig(f"{filename}legend.pdf", bbox_inches="tight", format="pdf")
    plt.close(legend_fig)


def save_deformation_legend(filename=str(OUT_DIR / "combined_overhead")) -> None:
    """Own legend of the deformation comparison (c: avoid vs. deform; also the utilization tradeoff): avoiding /
    deforming on tight and slack chiplets, and the hollow bar for configurations without a valid patch."""
    _fonts()
    by_method = {m: [_patch(label, color, hatch) for mm, _, label, color, hatch in sorted(SERIES, key=lambda s: s[1])
                     if mm == m] for m in ("chipmunq", "deform")}  # tight above slack
    none_bar = Patch(facecolor="white", hatch="xxx", edgecolor="#808080", linewidth=0.8, linestyle="--",
                     label="no valid patch")
    top = [by_method["chipmunq"][0], by_method["deform"][0], none_bar]
    bottom = [by_method["chipmunq"][1], by_method["deform"][1]]
    handles = _two_rows(top, bottom)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    legend_fig = plt.figure(figsize=(TEXT_WIDTH_IN / 2, 0.4))
    legend_fig.legend(handles=handles, loc="center", frameon=False, ncols=len(handles) // 2,
                      columnspacing=0.8, handlelength=1.6, handletextpad=0.35)
    legend_fig.savefig(f"{filename}_deformationlegend.pdf", bbox_inches="tight", format="pdf")
    plt.close(legend_fig)


# ######################################################################################
# Entry point
# ######################################################################################
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fig. 10: defective qubits (a-c compilation, d placement LER, "
                                             "c deformation)")
    ap.add_argument("--plot-only", action="store_true", help="only replot the saved results")
    ap.add_argument("--part", choices=["all", "compile", "placement", "deformation", "layouts"], default="all",
                    help="compile = a-c (Chipmunq vs. LightSABRE), placement = d (center vs. size-aware LER), "
                         "deformation = c (avoid vs. deform, Lin et al.), layouts = draw the mapping/routing "
                         "solutions of the placement experiment (not part of 'all')")
    ap.add_argument("--layout-sample", type=int, default=0,
                    help="layouts part: which defect placement (sample index of the placement experiment) to draw")
    ap.add_argument("--placement-backends", nargs="+", choices=["single", "multi"], default=["single", "multi"],
                    help="placement part: chiplet backends to run / plot (single = one patch per chiplet, "
                         "multi = several patches per chiplet; default: both)")
    ap.add_argument("--placement-defects", nargs="*", type=int, default=None,
                    help=f"placement part: #defective qubits to run (default {PLACEMENT_DEFECTS}). Saved results of "
                         f"other defect counts are kept. Without values only the missing untranspiled reference "
                         f"points are simulated")
    ap.add_argument("--max-shots", type=int, default=None,
                    help=f"maximum shots per simulated circuit, parts 2 and 3 (default {DEFAULT_MAX_SHOTS}; "
                         f"--quick: 20000)")
    ap.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 1, help="parallel workers (compilation of a-c and d, sampling of c and d)")
    ap.add_argument("--quick", action="store_true",
                    help="parts 2, 3: few samples / shots (pipeline smoke test)")
    a = ap.parse_args()
    placement_backends = [f"{b}_patch" for b in a.placement_backends]

    if a.part in ("all", "compile"):
        run_exp_defective(reproduce=not a.plot_only, jobs=a.jobs)

    if a.part in ("all", "placement"):
        if a.plot_only:
            if not PLACEMENT_PKL.exists():
                raise SystemExit(f"No results to plot: {PLACEMENT_PKL} not found. Run without --plot-only first.")
            with open(PLACEMENT_PKL, "rb") as f:
                placement_stats = pickle.load(f)
        else:
            placement_stats = run_placement(a.quick, jobs=a.jobs, backends=placement_backends,
                                            max_shots=a.max_shots, defects=a.placement_defects)
        plot_placement(placement_stats, backends=placement_backends)

    if a.part in ("all", "deformation"):
        if a.plot_only:
            if not DEFORM_PKL.exists():
                raise SystemExit(f"No results to plot: {DEFORM_PKL} not found. Run without --plot-only first.")
            with open(DEFORM_PKL, "rb") as f:
                data = pickle.load(f)
        else:
            data = run_deformation(a.jobs, a.quick, max_shots=a.max_shots)
        summarize_deformation(data["rows"])
        plot_deformation(data)

    if a.part == "layouts":
        if a.plot_only:
            if not LAYOUT_PKL.exists():
                raise SystemExit(f"No layouts to plot: {LAYOUT_PKL} not found. Run without --plot-only first.")
            with open(LAYOUT_PKL, "rb") as f:
                layouts = pickle.load(f)
        else:
            layouts = run_layouts(placement_backends, a.placement_defects, a.layout_sample, a.jobs, a.quick)
        plot_layouts(layouts, backends=placement_backends)

    # Legends: panels a-d, and the deformation comparison (written from the style constants on every run)
    save_legend()
    save_deformation_legend()