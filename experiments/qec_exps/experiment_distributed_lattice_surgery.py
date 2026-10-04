"""Distributed lattice-surgery CNOT: LER of the ideal (uncompiled) circuit, Chipmunq and LightSABRE.

Code distances d = 5, 7, 9, 11, 13. Chipmunq ("compiled", Basic routing: alpha = beta = 0) and LightSABRE
("sabre") each compile in their own process with a wall-clock limit of COMPILE_TIMEOUT_S; a configuration
that hits the limit is skipped (not simulated) and recorded in compile_status_<p_inter>.pkl.
Sampling: at most MAX_SHOTS shots and MAX_ERRORS logical errors per (method, d, p).

Run from the repo root:
    python experiments/qec_exps/experiment_distributed_lattice_surgery.py              # full
    python experiments/qec_exps/experiment_distributed_lattice_surgery.py --plot-only  # replot
    python experiments/qec_exps/experiment_distributed_lattice_surgery.py --quick      # smoke test
"""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.getcwd(), "."))

from stim import Circuit as StimCircuit
from experiments.exp_utils.circuit_generator import get_tqec_cnot_rotated
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit
from experiments.exp_utils.simulation_utils import *
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.utils import *
from qeccm.backends.backend_utils import plot_circuit_layout_utilization, plot_circuit_layout
from experiments.exp_utils.circuit_noise import get_noise_model

# Plotting
import argparse
import multiprocessing
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from pathlib import Path

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
DISTANCE_SCALES = [2, 3, 4, 5, 6]  # d = 2k + 1 -> 5, 7, 9, 11, 13
MAX_SHOTS = 10_000_000              # per (method, d, p)
MAX_ERRORS = 1000                   # per (method, d, p)
COMPILE_TIMEOUT_S = 1000           # wall-clock limit per compilation (Chipmunq Basic and LightSABRE)
OUTPUT_DIR = Path("experiments/evaluation/qec_evaluation")

# Plot styles: one colour per method, one marker per code distance
METHODS = [("default", "Ideal", "#000000"), ("compiled", "Chipmunq", "#3B6FA8"), ("sabre", "LightSABRE", "#C85E59")]
MARKERS = ["x", "o", "s", "^", "D", "v", "P"]


def get_backend(p_icc: float, amp_icc: float, d: int = -1):
    """Backend for distance scale d (code distance 2d + 1); each patch fits on one chiplet.
    Chiplet size (11 + 4(d-1)) x (6 + 2(d-1)) with 2d + 3 inter-chiplet links, as for d = 1..4 before."""
    chiplet_size = (6, 6, 11 + 4 * (d - 1), 6 + 2 * (d - 1))
    nic = 2 * d + 3
    backend = BackendChipletV2(
        size=chiplet_size,
        n_inter=nic,
        connectivity="nn",
        topology="rotated_grid",
        inter_chiplet_noise=p_icc,
        inter_chiplet_amplification=amp_icc,
        inter_chiplet_noise_type="constant",
        num_defective_qubits=0,
    )
    return nic, backend


def compile_circuit(t: str, distance_scale: int, p_icc: float, amp_icc: float):
    """Compile the CNOT with Chipmunq ("compiled") or LightSABRE ("sabre"). Module level so it can run in a
    killable child process. Returns (Stim circuit as string, compiled QuantumCircuit)."""
    _, backend = get_backend(p_icc=p_icc, amp_icc=amp_icc, d=distance_scale)
    circuit, partitions = get_tqec_cnot_rotated(distance_scale=distance_scale, n1=1, n2=0)
    if t == "compiled":
        _, custom_circuit, _, _ = transpile_stim_circuit(
            circuit,
            backend,
            pre_defined_partitions=partitions,
            routing_type="cost",
            routing_alpha=0,
            routing_beta=0,
        )
    elif t == "sabre":
        custom_circuit = sabre_transpilation(StimCodeCircuit(stim_circuit=circuit).qc, backend)
    else:
        raise ValueError(t)
    return str(get_stim_circuits_with_detectors(custom_circuit)[0][0]), custom_circuit


def _rates(stats):
    """{method: {d: {p: LER}}} (only configurations that were simulated)."""
    error_rates = defaultdict(lambda: defaultdict(dict))
    for s in stats:
        n = s.shots - s.discards
        if n > 0:
            md = s.json_metadata
            error_rates[str(md["run_name"])][str(md["d"])][md["p"]] = s.errors / n
    return error_rates


def plot_evaluation(stats, filename, inter_chiplet_noise):
    error_rates = _rates(stats)
    physical_error_rates = sorted({s.json_metadata["p"] for s in stats})
    d_values = sorted({str(s.json_metadata["d"]) for s in stats}, key=int)
    marker = {d: MARKERS[i % len(MARKERS)] for i, d in enumerate(d_values)}

    tex_fonts = {
        "font.family": "serif",
        "axes.labelsize": FONTSIZE * 1.5,
        "font.size": FONTSIZE * 1.2,
        "legend.fontsize": (FONTSIZE - 2) * 1.3,
        "xtick.labelsize": (FONTSIZE - 1) * 1.3,
        "ytick.labelsize": (FONTSIZE - 1) * 1.3,
        "axes.titlesize": 10,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "lines.markeredgewidth": 1.5,
        "lines.markeredgecolor": "black",
        "errorbar.capsize": 3,
    }
    plt.rcParams.update(tex_fonts)

    # ---------------- a) LER of every method and distance ----------------
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE * 2.5, WIDTH_FIGSIZE * 0.5))
    plt.plot(physical_error_rates, physical_error_rates, linestyle="--", linewidth=1.2, color="#000000B3")
    for t, _, color in METHODS:
        for d in d_values:
            pts = [(p, error_rates[t][d][p]) for p in physical_error_rates
                   if p in error_rates[t][d] and error_rates[t][d][p] > 0]  # points without errors are not shown
            if not pts:
                continue  # compilation timed out, or no errors observed at any p
            xs, ys = zip(*pts)
            plt.plot(xs, ys, linewidth=1.5, marker=marker[d], markerfacecolor="none", linestyle="solid",
                     color=color, markeredgecolor=color)

    if inter_chiplet_noise == 0.0001:
        ps_inter_text = r"$1e^{-4}$"
    elif inter_chiplet_noise == 0.001:
        ps_inter_text = r"$1e^{-3}$"
    elif inter_chiplet_noise == 0.01:
        ps_inter_text = r"$1e^{-2}$"

    ax.text(-0.01, 1.025, "a) Compilation effects", transform=ax.transAxes, fontweight="bold")
    ax.text(0.3, 1.15, "Lower is better ↓", transform=ax.transAxes, fontweight="bold", color=plot_lib_color)
    plt.ylim(5e-9, 2e0)
    plt.xlim(1e-4, 1e-2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Physical error rate")
    plt.ylabel("LER")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.subplots_adjust(left=0.16, right=0.95, top=0.83, bottom=0.21)
    plt.savefig(filename + ".pdf", format="pdf")
    plt.close(fig)

    # ---------------- b) LER_LightSABRE / LER_Chipmunq ----------------
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE * 2.5, WIDTH_FIGSIZE * 0.5))
    for d in d_values:
        sab, chip, ideal = error_rates["sabre"][d], error_rates["compiled"][d], error_rates["default"][d]
        ps_both = [p for p in physical_error_rates if sab.get(p, 0) > 0 and chip.get(p, 0) > 0]
        if not ps_both:
            continue
        ratio = [sab[p] / chip[p] for p in ps_both]
        avg = [r for p, r in zip(ps_both, ratio) if p < 1e-2]
        avg_ideal = [sab[p] / ideal[p] for p in ps_both if p < 1e-2 and ideal.get(p, 0) > 0]
        print(f"Relative increase for compiled {d} for icc {inter_chiplet_noise}: {np.mean(avg) if avg else np.nan}")
        print(f"Relative increase for sabre {d} for icc {inter_chiplet_noise}: "
              f"{np.mean(avg_ideal) if avg_ideal else np.nan}")
        plt.plot(ps_both, ratio, linewidth=1.5, marker=marker[d], markersize=5, markerfacecolor="none",
                 linestyle="--", color="#3B6FA8", markeredgecolor="#3B6FA8")

    plt.xscale("log")
    plt.xlabel("Physical error rate")
    plt.ylabel(r"$LER_{LightSABRE} / LER_{Chipmunq}$")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.yscale("log")
    ax.text(-0.01, 1.025, "b) LER comparison", transform=ax.transAxes, fontweight="bold")
    ax.text(0.3, 1.15, "Lower is better ↓", transform=ax.transAxes, fontweight="bold", color=plot_lib_color)
    fig.subplots_adjust(left=0.2, right=0.95, top=0.83, bottom=0.21)  # wider left margin for the long y-label
    plt.savefig(filename + "_relative.pdf", format="pdf")
    plt.close()

    # ---------------- compact legend: colour = method, marker = code distance ----------------
    handles = [Line2D([], [], color=color, linewidth=1.5, label=label) for _, label, color in METHODS]
    handles += [Line2D([], [], color="#606060", marker=marker[d], markerfacecolor="none", markeredgecolor="#606060",
                       linestyle="none", label=f"d={d}") for d in d_values]
    legend_fig = plt.figure(figsize=(3, 2))
    legend_fig.legend(handles=handles, loc="center", frameon=False, ncols=len(handles), columnspacing=1.2,
                      handlelength=1.5)
    legend_fig.savefig(filename + "legend.pdf", bbox_inches="tight", format="pdf")
    plt.close(legend_fig)


def run_exp_distributed_lattice_surgery(reproduce: bool = False, quick: bool = False) -> None:
    # Physical noise level
    ps = list(np.logspace(-4, -2, 6)) if not quick else [1e-3, 5e-3]  # 1e-4 ... 1e-2, 6 points

    # Inter-chiplet noise levels
    inter_chiplet_noise = [1e-3]

    # Compilation methods
    # - default: No compilation
    # - compiled: Our method (Basic routing)
    # - sabre: LightSABRE method
    ts = ["default", "compiled", "sabre"]

    # Code size of surface code: d = 2k + 1
    ks = DISTANCE_SCALES if not quick else [1, 2]
    timeout = COMPILE_TIMEOUT_S if not quick else 120
    output_dir = OUTPUT_DIR / ("quick" if quick else "")

    for ps_inter in inter_chiplet_noise:
        if reproduce:
            # ---- compile every (method, k) once; Chipmunq and LightSABRE under the time limit ----
            circuits, status = {}, {}
            for k in ks:
                circuit, _ = get_tqec_cnot_rotated(distance_scale=k, n1=1, n2=0)
                circuits[("default", k)] = get_stim_circuits_with_detectors(StimCodeCircuit(circuit).qc)[0][0]
                status[("default", k)] = dict(status="ok", runtime_s=0.0)
                for t in ("compiled", "sabre"):
                    res, runtime, st = run_with_timeout(compile_circuit, t, k, ps_inter, 1, timeout=timeout)
                    status[(t, k)] = dict(status=st, runtime_s=runtime)
                    print(f"[d={2 * k + 1}] {t}: {st} after {runtime:.0f} s", flush=True)
                    if st != "ok":
                        continue  # timed out / failed: not simulated
                    stim_str, custom_circuit = res
                    circuits[(t, k)] = StimCircuit(stim_str)
                    _, backend = get_backend(p_icc=ps_inter, amp_icc=1, d=k)
                    for plot_fn, prefix in ((plot_circuit_layout, "layout"),
                                            (plot_circuit_layout_utilization, "mapping")):
                        try:
                            plot_fn(custom_circuit, backend,
                                    filename=str(output_dir / "backend_mapping" / f"{prefix}_{t}_{k}.png"))
                        except Exception as e:  # visualisation only
                            print(f"{prefix} plot skipped: {e}")

            def _get_sinter_task():
                for (t, k), circ in circuits.items():
                    remote = None if t == "default" else get_backend(d=k, p_icc=ps_inter, amp_icc=1)[1].inter_chiplet_connections
                    for p in ps:
                        yield sinter.Task(
                            circuit=get_noise_model("modsi1000", None, p, None, remote=remote).noisy_circuit(circ),
                            json_metadata={"d": 2 * k + 1, "r": 2 * k + 1, "p": p, "run_name": t},
                        )

            # Run simulation (at most MAX_SHOTS shots / MAX_ERRORS errors per task)
            stats = sinter.collect(
                num_workers=max(1, multiprocessing.cpu_count() // 2),
                tasks=_get_sinter_task(),
                max_shots=MAX_SHOTS,
                max_errors=MAX_ERRORS,
                decoders=["pymatching"],
                print_progress=True,
                hint_num_tasks=len(circuits) * len(ps),
                count_observable_error_combos=True,
            )

            # Save simulation results and compile status (timeouts)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / f"single_cnot_rotated_{ps_inter}.pkl", "wb") as f:
                pickle.dump(stats, f)
            with open(output_dir / f"compile_status_{ps_inter}.pkl", "wb") as f:
                pickle.dump(status, f)

        # Load simulation results
        with open(output_dir / f"single_cnot_rotated_{ps_inter}.pkl", "rb") as f:
            stats = pickle.load(f)
        status_file = output_dir / f"compile_status_{ps_inter}.pkl"
        if status_file.exists():
            with open(status_file, "rb") as f:
                for (t, k), st in pickle.load(f).items():
                    if st["status"] != "ok":
                        print(f"d={2 * k + 1} {t}: {st['status']} (not simulated)")

        # Plot statistics
        plot_evaluation(
            stats,
            filename=str(output_dir / f"single_cnot_rotated_{ps_inter}"),
            inter_chiplet_noise=ps_inter,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--quick", action="store_true", help="d=3,5, two error rates, 120 s compile limit")
    a = ap.parse_args()
    run_exp_distributed_lattice_surgery(reproduce=not a.plot_only, quick=a.quick)