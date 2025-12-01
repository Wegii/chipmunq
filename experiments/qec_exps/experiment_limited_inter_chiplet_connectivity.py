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

def plot_interconnect_sweep(stats, filename):
    from collections import defaultdict
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Collect raw data points grouped by physical error rate
    points_by_p = defaultdict(list)   # p → list of (p_inter, ler, d, config)
    points_by_group = defaultdict(list)  # (p, run_name) -> list of (p_inter, ler)


    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection="3d")

    for s in stats:
        ler = s.errors / (s.shots - s.discards)

        p        = s.json_metadata["p"]
        p_inter  = s.json_metadata["p_inter"]
        d        = s.json_metadata["d"]
        config   = s.json_metadata["run_name"]

        points_by_p[p].append((p_inter, ler, d, config))
        points_by_group[(p, config)].append((p_inter, ler))


        ax.scatter(p, p_inter, ler, marker='o')

    for (p, run_name), plist in points_by_group.items():
        # Sort so that connecting lines follow p_inter
        plist_sorted = sorted(plist, key=lambda x: x[0])

        p_inters = [pt[0] for pt in plist_sorted]
        lers     = [pt[1] for pt in plist_sorted]
        ps       = [p] * len(plist_sorted)

        # Connect points
        ax.plot(ps, p_inters, lers)#, label=f"{run_name}")



    ax.set_xlabel("Physical Error Rate (p)")
    ax.set_ylabel("Inter-Chiplet Error Rate (p_inter)")
    ax.set_zlabel("Logical Error Rate (LER)")
    ax.set_title("3D Scatter Plot with Lines Grouped by Physical Error Rate")
    #ax.zaxis.set_ticks_position('left')
    #ax.zaxis.set_label_position('left')

    ax.legend()

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def simulate_single_cnot_from_tqec() -> None:

    # Reference circuit
    circuit, partitions = get_tqec_cnot_rotated()
    normal_circuit_stim = get_stim_circuits_with_detectors(StimCodeCircuit(circuit).qc)[0][0]

    # Number of inter_chiplet_connections
    num_inter_chiplet_connections = [8, 1]#[8, 4, 2, 1]
    
    # Noise level
    ps = list(np.logspace(-4, -1, 10))#list(np.logspace(-4, -1, 10))
    # Inter-chiplet noise level
    ps_inter = [1e-2]#[1e-3, 1e-2]

    # Transpilation
    ts = [str(i) for i in num_inter_chiplet_connections]
    ks = [2]
    routing_types = ["cost", "default"]#, ["default"]

    transpiled_circuits = {}

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
                                                                routing_alpha = 1e4,
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
                                    inter_chiplet_noise_type = "random"#"constant"
                                    )

    def _get_sinter_task():
        # Construct sinter task for multiple code distances and noise levels
        yield from (
            sinter.Task(
                circuit=circuit,
                json_metadata={"d": 2 * k + 1, "r": 2 * k + 1, "p": p, "run_name": t + rt, "p_inter": p_icc},
            )
            for circuit, k, p, t, rt, p_icc in (
                (get_noise_model("modsi1000",
                                 None,
                                 p,
                                 None, 
                                 remote = (None if t == "default" else
                                           get_backend(int(t), p_icc, 1, t).inter_chiplet_connections)
                                 ).noisy_circuit(
                                     get_circuit(-1 if t == "default" else int(t), p_icc, 1, t, routing_type=rt)
                                     ), k, p, t, rt, p_icc)
                
                for p_icc in ps_inter
                for t in ts
                for rt in routing_types
                for k in ks
                for p in ps
            )
        )

    stats = run_sinter_simulation(_get_sinter_task, ks, ps)

    plot_evaluation(stats,
                    filename = f"experiments/evaluation/chiplet_evaluation/single_cnot_rotated_inter_chiplet{ps_inter[0]}.png",
                    with_transpilation = True)
    
    #plot_interconnect_sweep(stats,
    #                        filename = f"experiments/evaluation/chiplet_evaluation/single_cnot_rotated_inter_chiplet_sweep.png")


if __name__ == "__main__":
    simulate_single_cnot_from_tqec()