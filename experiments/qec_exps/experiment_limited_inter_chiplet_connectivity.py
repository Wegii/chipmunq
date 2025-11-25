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
            stat.json_metadata["configuration"]
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


def simulate_single_cnot_from_tqec() -> None:

    # Reference circuit
    circuit, partitions = get_tqec_cnot_rotated()
    normal_circuit_stim = get_stim_circuits_with_detectors(StimCodeCircuit(circuit).qc)[0][0]

    # Number of inter_chiplet_connections
    num_inter_chiplet_connections = [8, 4]#, 2, 1]
    
    # Noise level
    ps = list(np.logspace(-4, -1, 10))
    # Inter-chiplet noise level
    ps_inter = [1e-3]

    # Transpilation
    ts = ["default"] + [str(i) for i in num_inter_chiplet_connections]
    ks = [2]
    routing_types = ["cost", "default"]

    transpiled_circuits = {}

    def get_circuit(n_icc, p_icc, amp_icc, t: str, routing_type: str) -> StimCircuit:
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
                                                                routing_alpha = 1e3,
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
                json_metadata={"d": 2 * k + 1, "r": 2 * k + 1, "p": p, "configuration": t + rt},
            )
            for circuit, k, p, t, rt in (
                (get_noise_model("si1000",
                                 None,
                                 p,
                                 None, 
                                 remote = (None if t == "default" else
                                           get_backend(n_icc, p_icc, 1, t).inter_chiplet_connections)
                                 ).noisy_circuit(
                                     get_circuit(n_icc, p_icc, 1, t, routing_type=rt)
                                     ), k, p, t, rt)
                
                for k in ks
                for p in ps
                for p_icc in ps_inter
                for t in ts
                for rt in routing_types
                for n_icc in num_inter_chiplet_connections
            )
        )

    stats = run_sinter_simulation(_get_sinter_task, ks, ps)

    plot_evaluation(stats,
                    filename = f"experiments/evaluation/chiplet_evaluation/single_cnot_rotated_inter_chiplet.png",
                    with_transpilation = True)


if __name__ == "__main__":
    simulate_single_cnot_from_tqec()