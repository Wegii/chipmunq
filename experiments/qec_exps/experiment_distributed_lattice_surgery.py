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
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


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
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "lines.markeredgewidth": 1.5,
        "lines.markeredgecolor": "black",
        # Error bar cap size
        "errorbar.capsize": 3,
    }
    plt.rcParams.update(tex_fonts)

    # fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE*2.6, WIDTH_FIGSIZE))
    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE * 2.5, WIDTH_FIGSIZE * 0.5))
    # Plot identity (x = y)
    plt.plot(physical_error_rates, physical_error_rates, linestyle="--", linewidth=1.2, color="#000000B3", label="x=y")

    colors_sabre = ["#E38E8A", "#C85E59", "#9F3B36"]
    colors_transpiled = ["#5E97CC", "#3B6FA8", "#2A5687"]
    colors_default = ["#000", "#000", "#000"]  # ([ "#C85E59", "#9F3B36", "#7F2E2A"])
    color_list = [colors_default, colors_transpiled, colors_sabre]

    # TODO: Add sabre to the plots
    inter_markers = ["x", "o", "s"]
    handles = []
    d_values_reduced = ["5", "7"]

    for ti, t in enumerate(["default", "compiled", "sabre"]):
        for i, d in enumerate(d_values_reduced):
            errors = defaultdict(dict)

            for p in physical_error_rates:
                errors[p] = error_rates[t][d][p][0]

            if t == "default":
                l = "Ideal"
            elif t == "compiled":
                l = "Chipmunq"
            elif t == "sabre":
                l = "LightSABRE"
            ys_custom = [errors[p] for p in physical_error_rates]
            h = plt.plot(
                physical_error_rates,
                ys_custom,
                linewidth=1.5 if t == "default" else 1.5,
                marker=inter_markers[i],
                markerfacecolor="none",
                linestyle="solid",  # "--" if t == "default" else "solid",
                color=color_list[ti][i],
                label=f"{l}, d={d}",
            )
            handles.extend(h)

    if inter_chiplet_noise == 0.0001:
        ps_inter_text = r"$1e^{-4}$"
    elif inter_chiplet_noise == 0.001:
        ps_inter_text = r"$1e^{-3}$"
    elif inter_chiplet_noise == 0.01:
        ps_inter_text = r"$1e^{-2}$"
    description = r"$p_{inter}$ = " + f"{ps_inter_text}"

    ax.text(-0.0, 1.04, "a) Effect of compilation on the LER", transform=ax.transAxes, fontweight="bold")

    ax.text(
        0.3,
        1.15,
        "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    # plt.ylim(-0.01, 0.9)
    plt.ylim(5e-9, 2e0)
    plt.xlim(1e-4, 1e-2)
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Physical error rate")
    plt.ylabel("LER")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.subplots_adjust(left=0.24, right=0.95, top=0.8, bottom=0.21)
    plt.savefig(filename + ".pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE * 2.6, WIDTH_FIGSIZE))
    for i, d in enumerate(d_values):
        errors = defaultdict(dict)
        errors_sabre = defaultdict(dict)
        errors_average = []
        errors_average_sabre = []

        for p in physical_error_rates:
            errors[p] = error_rates["compiled"][d][p][0] / error_rates["default"][d][p][0]
            errors_sabre[p] = error_rates["sabre"][d][p][0] / error_rates["default"][d][p][0]

            if p < 1e-2:
                errors_average.append(errors[p])
                errors_average_sabre.append(errors_sabre[p])

        print(f"Relative increase for compiled {d} for icc {inter_chiplet_noise}: {np.mean(errors_average)}")
        print(f"Relative increase for sabre {d} for icc {inter_chiplet_noise}: {np.mean(errors_average_sabre)}")

        ys_custom = [errors[p] for p in physical_error_rates]
        plt.plot(
            physical_error_rates,
            ys_custom,
            linewidth=1.5,
            marker=inter_markers[i],
            markersize=5,
            markerfacecolor="none",
            linestyle="--",
            color=colors_transpiled[i],
            label=f"({d})",
        )

    ax.text(0, 1.02, description, transform=ax.transAxes, fontweight="bold")

    ax.text(
        0.57,
        1.08,
        "Lower is better ↓",
        transform=ax.transAxes,
        fontweight="bold",
        color=plot_lib_color,
    )

    # plt.ylim(-0.01, 0.9)
    # plt.ylim(5e-9, 1e0)
    plt.xscale("log")
    # plt.yscale('log')

    plt.xlabel("Physical error rate")
    plt.ylabel(r"$LER_{Compiled} / LER_{Default}$")
    # plt.legend(loc="upper right", ncol=1)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.subplots_adjust(left=0.24, right=0.95, top=0.95, bottom=0.1)
    plt.savefig(filename + "_relative.pdf", format="pdf")
    plt.close()

    legend_fig = plt.figure(figsize=(3, 2))
    legend = legend_fig.legend(handles=handles, loc="center", frameon=False, ncols=3, columnspacing=1.5)
    legend_fig.savefig(filename + "legend.pdf", bbox_inches="tight", format="pdf")
    plt.close(legend_fig)


def run_exp_distributed_lattice_surgery(reproduce: bool = False) -> None:
    # Physical noise level
    ps = list(np.logspace(-4, -1, 10))

    # Inter-chiplet noise levels
    inter_chiplet_noise = [1e-3]  # [1e-4, 1e-3, 1e-2]

    # Compilation methods
    # - default: No compilation
    # - compiled: Our method
    # - sabre: LightSABRE method
    ts = ["default", "compiled", "sabre"]

    # Code size of surface code
    ks = [1, 2, 3]

    for ps_inter in inter_chiplet_noise:
        if reproduce:
            transpiled_circuits = {}

            def get_circuit(distance_scale: int, p_icc, amp_icc, t: str) -> StimCircuit:

                if t == "default":
                    # Reference circuit
                    circuit, partitions = get_tqec_cnot_rotated(distance_scale=distance_scale, n1=1, n2=0)
                    normal_circuit_stim = get_stim_circuits_with_detectors(StimCodeCircuit(circuit).qc)[0][0]
                    return normal_circuit_stim
                else:
                    n_icc, backend = get_backend(p_icc=p_icc, amp_icc=amp_icc, d=distance_scale)

                    if (distance_scale, n_icc, p_icc, amp_icc, t) in transpiled_circuits:
                        # Circuit does not need to be transpiled again
                        print("Utilizing existing backend")
                        return transpiled_circuits[(distance_scale, n_icc, p_icc, amp_icc, t)]
                    else:
                        circuit, partitions = get_tqec_cnot_rotated(distance_scale=distance_scale, n1=1, n2=0)

                        # Transpile circuit to backend
                        if t == "compiled":
                            # Custom compilation
                            _, custom_circuit, _, _ = transpile_stim_circuit(
                                circuit,
                                backend,
                                pre_defined_partitions=partitions,
                                routing_type="cost",
                                routing_alpha=0,
                                routing_beta=0,
                            )
                        elif t == "sabre":
                            # Compilation using SABRRE
                            stim_code_circuit = StimCodeCircuit(stim_circuit=circuit)
                            custom_circuit = sabre_transpilation(stim_code_circuit.qc, backend)

                        # Convert circuit to stim
                        custom_circuit_stim = get_stim_circuits_with_detectors(custom_circuit)[0][0]
                        # Add circuit to dictionary, in order to not transpile this circuit configuration again
                        transpiled_circuits[(distance_scale, n_icc, p_icc, amp_icc, t)] = custom_circuit_stim

                        plot_circuit_layout(
                            custom_circuit,
                            backend,
                            filename=f"experiments/evaluation/qec_evaluation/backend_mapping/layout_{t}_{distance_scale}.png",
                        )

                        plot_circuit_layout_utilization(
                            custom_circuit,
                            backend,
                            filename=f"experiments/evaluation/qec_evaluation/backend_mapping/mapping_{t}_{distance_scale}.png",
                        )

                        return custom_circuit_stim

            def get_backend(p_icc: float, amp_icc: float, t: str = "", d: int = -1) -> BackendChipletV2:
                if t == "default":
                    return None
                else:
                    # Depending on the distance, each chiplet needs to be scaled

                    if d == 1:
                        chiplet_size = (6, 6, 11, 6)
                        nic = 5
                    elif d == 2:
                        chiplet_size = (6, 6, 15, 8)
                        nic = 7
                    elif d == 3:
                        chiplet_size = (6, 6, 19, 10)
                        nic = 9
                    elif d == 4:
                        chiplet_size = (6, 6, 23, 12)
                        nic = 11

                    backend = BackendChipletV2(
                        size=chiplet_size,  # size = (6, 6, 15, 8),
                        n_inter=nic,
                        connectivity="nn",
                        topology="rotated_grid",
                        inter_chiplet_noise=p_icc,
                        inter_chiplet_amplification=amp_icc,
                        inter_chiplet_noise_type="constant",  # "constant"
                        num_defective_qubits=0,
                    )

                    return nic, backend

            def _get_sinter_task():
                # Construct sinter task for multiple code distances and noise levels
                yield from (
                    sinter.Task(
                        circuit=circuit,
                        json_metadata={
                            "d": 2 * k + 1,
                            "r": 2 * k + 1,
                            "p": p,
                            "run_name": t,
                        },
                    )
                    for circuit, k, p, t in (
                        (
                            get_noise_model(
                                "modsi1000",
                                None,
                                p,
                                None,
                                remote=(
                                    None
                                    if t == "default"
                                    else get_backend(d=k, p_icc=ps_inter, amp_icc=1, t=t)[1].inter_chiplet_connections
                                ),
                            ).noisy_circuit(get_circuit(distance_scale=k, p_icc=ps_inter, amp_icc=1, t=t)),
                            k,
                            p,
                            t,
                        )
                        for t in ts
                        for k in ks
                        for p in ps
                    )
                )

            # Run simulation
            stats = run_sinter_simulation(_get_sinter_task, ks, ps)

            # Save simulation results
            output_dir = Path("experiments/evaluation/qec_evaluation")
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / f"single_cnot_rotated_{ps_inter}.pkl", "wb") as f:
                pickle.dump(stats, f)

        # Load simulation results
        with open(f"experiments/evaluation/qec_evaluation/single_cnot_rotated_{ps_inter}.pkl", "rb") as f:
            stats = pickle.load(f)

        # Plot statistics
        plot_evaluation(
            stats,
            filename=f"experiments/evaluation/qec_evaluation/single_cnot_rotated_{ps_inter}",
            inter_chiplet_noise=ps_inter,
        )


if __name__ == "__main__":
    run_exp_distributed_lattice_surgery(reproduce=True)
