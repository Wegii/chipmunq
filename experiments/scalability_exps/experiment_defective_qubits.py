"""
Fig. 10: defective qubits.

Part 1 -- a) circuit depth, b) #2q gates, c) chiplet utilization:
    Chipmunq vs. LightSABRE on backends with defective qubits (single-/multi-patch chiplets, center /
    size-aware placement).

Part 3 -- d) LER of the two placement strategies:
    LER of the distributed lattice-surgery CNOT after Chipmunq compilation with center and size-aware placement on
    the single-patch chiplet backend of a-c, for several code distances and numbers of defective qubits. Both
    strategies are compiled on the same backend (same defects) for every defect placement.

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

All panels are drawn at their printed size for four panels side by side across the full page width. One legend
(combined_overheadlegend.pdf) covers panels a-d, the deformation comparison has its own
(combined_overhead_deformationlegend.pdf); both depend only on the styles below, so they are written on every run,
also with --plot-only and for a single --part.

Run from the repo root:
    python experiments/scalability_exps/experiment_defective_qubits.py                    # both parts
    python experiments/scalability_exps/experiment_defective_qubits.py --plot-only        # replot both
    python experiments/scalability_exps/experiment_defective_qubits.py --part compile     # a-c only
    python experiments/scalability_exps/experiment_defective_qubits.py --part placement [--quick]  # d only
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
MAX_SHOTS, MAX_ERRORS, BATCH = 2_000_000, 100, 50_000
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


def run_deformation(jobs: int, quick: bool) -> dict:
    n_samples = 4 if quick else N_SAMPLES
    max_shots = 200_000 if quick else MAX_SHOTS
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
    print(f"{len(rows)} (method, L, k, sample) points, {len(tasks)} distinct circuits on {jobs} workers",
          flush=True)
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
PLACEMENT_DEFECTS = [1, 3]                     # defective qubits per chiplet
PLACEMENT_PS = list(np.logspace(-3, -2, 6))    # 1e-3 ... 1e-2, 6 points
PLACEMENT_SEEDS = 4                            # defect placements (backends) per (d, #defects)
PLACEMENT_MAX_SHOTS, PLACEMENT_MAX_ERRORS = 1_000_000, 500
PLACEMENT_P_INTER = 1e-4                       # inter-chiplet noise, as in a-c
PLACEMENT_MAX_BACKEND_RETRIES = 20
PLACEMENT_PKL = OUT_DIR / "placement_ler.pkl"
PLACEMENTS = [("center", COLORS_CUSTOM[0]), ("size_aware", COLORS_CUSTOM[1])]  # colours of a-c (single patch)
PLACEMENT_MARKERS = {3: "v", 5: "o", 7: "s", 9: "D"}
PLACEMENT_DEFECT_LS = {1: "-", 2: "-.", 3: "--", 5: ":"}


def _placement_backend(k: int, n_defects: int, seed: int) -> BackendChipletV2:
    """Single-patch chiplets (one patch per chiplet) as in a-c, sized for distance scale k."""
    return BackendChipletV2(
        size=(6, 6, 11 + 4 * (k - 1), 6 + 2 * (k - 1)),
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
    """Compile one (d, #defects, defect placement) with both strategies on the same backend and build the noisy
    circuits for every physical error rate. Module level: runs in a spawned worker. If both strategies give the
    identical circuit, it is returned once (simulated once, counted for both)."""
    k, n_def, s, circuit_str, partitions, ps = job
    qc = StimCodeCircuit(stim_circuit=stim.Circuit(circuit_str)).qc
    retries = 0
    for attempt in range(PLACEMENT_MAX_BACKEND_RETRIES):
        seed = 1000 * s + 42 + attempt
        backend = _placement_backend(k, n_def, seed)
        try:
            compiled = {pp: str(get_stim_circuits_with_detectors(custom_cost_transpilation(
                qc, backend, pre_defined_partitions=partitions, patch_initialization=pp))[0][0])
                for pp, _ in PLACEMENTS}
            break
        except Exception:
            retries += 1
    else:
        raise RuntimeError(f"No backend for d={2 * k + 1} with {n_def} defects could be compiled")
    groups = {}  # circuit -> placements producing it
    for pp, circ in compiled.items():
        groups.setdefault(circ, []).append(pp)
    out = []
    for circ, pps in groups.items():
        c = stim.Circuit(circ)
        for p in ps:
            noisy = get_noise_model("modsi1000", None, p, None, remote=backend.inter_chiplet_connections).noisy_circuit(c)
            out.append((pps, p, str(noisy)))
    return dict(k=k, n_def=n_def, s=s, seed=seed, retries=retries, circuits=out)


def run_placement(quick: bool = False, jobs: int | None = None) -> list:
    """Compile every (d, #defects, defect placement) with both placement strategies on the same backend (in
    parallel, noisy circuits built in the workers) and simulate the LER. A backend seed is only used if both
    strategies can place all patches; identical circuits of the two strategies are simulated once."""
    ks = [1] if quick else PLACEMENT_KS
    n_seeds = 1 if quick else PLACEMENT_SEEDS
    ps = [2e-3, 5e-3] if quick else PLACEMENT_PS
    max_shots, max_errors = (20_000, 100) if quick else (PLACEMENT_MAX_SHOTS, PLACEMENT_MAX_ERRORS)

    sources = {}
    for k in ks:
        circuit, partitions = get_tqec_cnot_rotated(distance_scale=k, n1=1, n2=0)
        sources[k] = (str(circuit), partitions)
    job_list = [(k, n_def, s, *sources[k], ps) for k in ks for n_def in PLACEMENT_DEFECTS for s in range(n_seeds)]
    jobs = min(jobs or os.cpu_count() or 1, len(job_list))
    print(f"Fig. 10 d: {len(job_list)} backends (x {len(PLACEMENTS)} placements) on {jobs} workers", flush=True)
    tasks, shared = [], 0
    with _capped_pool(jobs) as pool:
        for r in pool.imap_unordered(_placement_job, job_list, chunksize=1):
            d = 2 * r["k"] + 1
            print(f"[d={d}, {r['n_def']} defects, sample {r['s']}] backend seed {r['seed']} "
                  f"({r['retries']} rejected)", flush=True)
            for pps, p, noisy in r["circuits"]:
                shared += len(pps) > 1
                tasks.append(sinter.Task(circuit=stim.Circuit(noisy), json_metadata={
                    "placements": pps, "placement": pps[0], "d": d, "defects": r["n_def"], "sample": r["s"],
                    "p": p, "backend_seed": r["seed"]}))
    print(f"{len(tasks)} sinter tasks ({shared} shared by both placements: identical circuits)", flush=True)

    stats = sinter.collect(num_workers=multiprocessing.cpu_count(), tasks=tasks, max_shots=max_shots,
                           max_errors=max_errors, decoders=["pymatching"], print_progress=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PLACEMENT_PKL, "wb") as f:
        pickle.dump(stats, f)
    return stats


def plot_placement(stats, filename=str(OUT_DIR / "combined_overhead")) -> None:
    """d) LER vs. physical error rate: colour = placement, marker = code distance, line style = #defects.
    Each point is the mean LER over the defect placements; points without observed errors are not shown."""
    ler = defaultdict(list)
    for st in stats:
        n = st.shots - st.discards
        if n > 0:
            m = st.json_metadata
            for pp in m.get("placements", [m["placement"]]):  # identical circuits count for both placements
                ler[(pp, m["d"], m["defects"], m["p"])].append(st.errors / n)
    ps = sorted({k[3] for k in ler})
    ds = sorted({k[1] for k in ler})
    defects = sorted({k[2] for k in ler})
    _fonts()
    fig, ax = _panel()
    for pp, color in PLACEMENTS:
        for d in ds:
            for nd in defects:
                pts = [(p, np.mean(ler[(pp, d, nd, p)])) for p in ps if ler.get((pp, d, nd, p))]
                pts = [(p, v) for p, v in pts if v > 0]
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=color, linestyle=PLACEMENT_DEFECT_LS.get(nd, "-"),
                        marker=PLACEMENT_MARKERS.get(d, "P"), markerfacecolor=color, markeredgecolor="black",
                        markeredgewidth=0.4, linewidth=1.1)
    from matplotlib.ticker import NullFormatter
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min(PLACEMENT_PS) / 1.15, max(PLACEMENT_PS) * 1.15)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_formatter(NullFormatter())  # no overlapping labels on ranges below one decade
    ax.set_xlabel("Physical error rate")
    ax.set_ylabel("LER")  # per logical CNOT (state in the caption)
    ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.text(0.5, 1.03, "d) Placement vs. LER", transform=ax.transAxes, fontweight="bold", ha="center",
            va="bottom")
    ax.text(0.5, 1.16, "Lower is better ↓", transform=ax.transAxes, fontweight="bold", color=plot_lib_color,
            ha="center", va="bottom")
    fig.savefig(f"{filename}_placement_ler.pdf", format="pdf")
    plt.close(fig)


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
    placement colours of d) are the Chipmunq single-patch colours. Built from the style constants only (no
    data), so it is written whatever part is run or replotted.
    """
    _fonts()
    grey = "#606060"
    dist = [Line2D([], [], color=grey, marker=PLACEMENT_MARKERS[2 * k + 1], linestyle="none",
                   markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.4, label=f"d={2 * k + 1}")
            for k in PLACEMENT_KS]
    defects = [Line2D([], [], color=grey, linestyle=PLACEMENT_DEFECT_LS.get(n, "-"),
                      label=f"{n} defect" + ("s" if n != 1 else "")) for n in PLACEMENT_DEFECTS]
    top = [_patch("Chipmunq single, center", COLORS_CUSTOM[0], HATCHES[0]),
           _patch("Chipmunq multi, center", COLORS_CUSTOM[2], HATCHES[0]),
           _patch("LightSABRE single", COLORS_SABRE[0])] + dist[0::2] + defects[0::2]
    bottom = [_patch("Chipmunq single, size-aware", COLORS_CUSTOM[1], HATCHES[1]),
              _patch("Chipmunq multi, size-aware", COLORS_CUSTOM[3], HATCHES[1]),
              _patch("LightSABRE multi", COLORS_SABRE[2])] + dist[1::2] + defects[1::2]
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
    ap.add_argument("--part", choices=["all", "compile", "placement", "deformation"], default="all",
                    help="compile = a-c (Chipmunq vs. LightSABRE), placement = d (center vs. size-aware LER), "
                         "deformation = c (avoid vs. deform, Lin et al.)")
    ap.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 1, help="parallel workers (compilation of a-c and d, sampling of c)")
    ap.add_argument("--quick", action="store_true",
                    help="parts 2, 3: few samples / shots (pipeline smoke test)")
    a = ap.parse_args()

    if a.part in ("all", "compile"):
        run_exp_defective(reproduce=not a.plot_only, jobs=a.jobs)

    if a.part in ("all", "placement"):
        if a.plot_only:
            if not PLACEMENT_PKL.exists():
                raise SystemExit(f"No results to plot: {PLACEMENT_PKL} not found. Run without --plot-only first.")
            with open(PLACEMENT_PKL, "rb") as f:
                placement_stats = pickle.load(f)
        else:
            placement_stats = run_placement(a.quick, jobs=a.jobs)
        plot_placement(placement_stats)

    if a.part in ("all", "deformation"):
        if a.plot_only:
            if not DEFORM_PKL.exists():
                raise SystemExit(f"No results to plot: {DEFORM_PKL} not found. Run without --plot-only first.")
            with open(DEFORM_PKL, "rb") as f:
                data = pickle.load(f)
        else:
            data = run_deformation(a.jobs, a.quick)
        summarize_deformation(data["rows"])
        plot_deformation(data)

    # Legends: panels a-d, and the deformation comparison (written from the style constants on every run)
    save_legend()
    save_deformation_legend()