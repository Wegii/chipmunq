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


def plot_evaluation(stat, filename, with_transpilation = False):
    fig, ax = plt.subplots()

    if with_transpilation:
        grp_fc = lambda stat: (
            stat.json_metadata["d"],
            stat.json_metadata["run_name"],
            )
    else:
        grp_fc = lambda stat: (
            stat.json_metadata["d"]
            )
    sinter.plot_error_rate(
        ax=ax,
        stats=stat,
        x_func=lambda stat: stat.json_metadata["p"],
        group_func=grp_fc,
    )
    #plot_observable_as_inset(ax, zx_graph, correlation_surfaces[i])
    ax.grid(axis="both")
    ax.legend()
    ax.loglog()
    ax.set_title("Logical Error Rate")
    ax.set_xlabel("Physical Error Rate")
    ax.set_ylabel("Logical Error Rate")
    fig.savefig(filename)

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

    diff_1 = defaultdict(dict)
    diff_4 = defaultdict(dict)

    for d in d_values:
        for p in physical_error_rates:
            diff_1[d][p] = error_rates['1'][d][p][0] - error_rates['8'][d][p][0]
            diff_4[d][p] = error_rates['4'][d][p][0] - error_rates['8'][d][p][0]

    fig, ax = plt.subplots(figsize=(8, 3))
  
    ys_custom = [diff_1['5'][p] for p in physical_error_rates]
    plt.plot(physical_error_rates, ys_custom, marker='x', linewidth=2, color='#ff8c00', label=f'(d=5, n_inter = 1)')

    ys_custom = [diff_4['5'][p] for p in physical_error_rates]
    plt.plot(physical_error_rates, ys_custom, marker='x', linewidth=2, color="#5c79bd", label=f'(d=5, n_inter = 4)')

    description = (r"$p_{inter}$ = " + f"{inter_chiplet_noise}")

    ax.text(
        0, 1.02, description,
        transform=ax.transAxes,
        fontsize=9,
        #fontweight="bold"
    )
    #ax.text(
    #    0, 1.02, description,
    #    transform=ax.transAxes,
    #    fontsize=9,
    #    fontweight="bold"
    #)

    ax.text(
        0.81, 1.02, "Lower is better ↓",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color="#5c79bd",
    )
    
    plt.ylim(-0.01, 0.9)
    plt.xscale('log')
    #plt.yscale('log')
    plt.yscale('symlog', linthresh=1e-3)  # linear within ±0.001

    plt.xlabel("Physical Error Rate")
    plt.ylabel(r"Δ($LER_{Reduced} - LER_{Full}$)")
    plt.legend(loc="lower right")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()



def run_exp_distributed_inter_chiplet() -> None:

    # Reference circuit
    circuit, partitions = get_tqec_cnot_rotated()
    normal_circuit_stim = get_stim_circuits_with_detectors(StimCodeCircuit(circuit).qc)[0][0]

    # Number of inter_chiplet_connections
    num_inter_chiplet_connections = [8, 4, 1]
    
    # Noise level
    ps = list(np.logspace(-4, -1, 10))

    # Inter-chiplet noise level
    inter_chiplet_noise = [1e-4, 1e-3, 1e-2]
    for ps_inter in inter_chiplet_noise:

        # Transpilation
        ts = [str(i) for i in num_inter_chiplet_connections]
        ks = [2]
        routing_types = ["default"] #["cost", "default"]

        transpiled_circuits = {}
        """
        
        def get_circuit(n_icc, p_icc, amp_icc, t: str , routing_type: str) -> StimCircuit:
            if t == "default":
                return normal_circuit_stim
            else:
                backend = get_backend(n_icc = n_icc,
                                    p_icc = p_icc,
                                    amp_icc = amp_icc)

                if (n_icc, p_icc, amp_icc, routing_type) in transpiled_circuits:
                    # Circuit does not need to be transpiled again
                    print("Found")
                    return transpiled_circuits[(n_icc, p_icc, amp_icc, routing_type)]
                else:
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
                    transpiled_circuits[(n_icc, p_icc, amp_icc, routing_type)] = custom_circuit_stim

                    return custom_circuit_stim

        def get_backend(n_icc: int, p_icc: float, amp_icc: float, t: str = "") -> BackendChipletV2:
            if t == "default":
                return None
            else:
                return BackendChipletV2(size = (2, 2, 15, 8),
                                        n_inter = n_icc,
                                        connectivity = "nn",
                                        topology = "rotated_grid",
                                        inter_chiplet_noise = p_icc,
                                        inter_chiplet_amplification = amp_icc,
                                        inter_chiplet_noise_type = "constant"#"random"#"constant"
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
                                            get_backend(int(t), ps_inter, 1, t).inter_chiplet_connections)
                                    ).noisy_circuit(
                                        get_circuit(-1 if t == "default" else int(t), ps_inter, 1, t, routing_type=rt)
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
                        with_transpilation = True)
        
        plot_difference(stats,
                        filename = f"experiments/evaluation/inter_chiplet/inter_chiplet_{ps_inter}_difference.png",
                        inter_chiplet_noise = ps_inter)
        
        
        
    #plot_interconnect_sweep(stats,
    #                        filename = f"experiments/evaluation/chiplet_evaluation/single_cnot_rotated_inter_chiplet_sweep.png")


if __name__ == "__main__":
    run_exp_distributed_inter_chiplet()