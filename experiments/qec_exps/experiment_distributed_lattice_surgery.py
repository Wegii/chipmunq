# How does the logical error rate of lattice surgery operations change when distributed to multiple chiplets?

# TODO: Show with simple routing, and with improved routing


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
from qeccm.backends.backend_utils import plot_gate_map

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


def run_exp_distributed_lattice_surgery() -> None:
    
    # Noise level
    ps = list(np.logspace(-4, -1, 10))

    # Inter-chiplet noise level
    inter_chiplet_noise = [1e-4, 1e-3, 1e-2]
    for ps_inter in inter_chiplet_noise:

        # Transpilation
        ts = ["default", "compiled"]# + [str(i) for i in num_inter_chiplet_connections]
        ks = [1, 2, 3]#[1, 2, 3]
        
        transpiled_circuits = {}

        def get_circuit(distance_scale: int, p_icc, amp_icc, t: str) -> StimCircuit:
            # Reference circuit
            
            if t == "default":
                circuit, partitions = get_tqec_cnot_rotated(distance_scale = distance_scale,
                                                            n1 = 1,
                                                            n2 = 0)
                normal_circuit_stim = get_stim_circuits_with_detectors(StimCodeCircuit(circuit).qc)[0][0]
                return normal_circuit_stim
            else:
                n_icc, backend = get_backend(p_icc = p_icc,
                                    amp_icc = amp_icc, 
                                    d = distance_scale)

                if (distance_scale, n_icc, p_icc, amp_icc) in transpiled_circuits:
                    # Circuit does not need to be transpiled again
                    print("Found")
                    return transpiled_circuits[(distance_scale, n_icc, p_icc, amp_icc)]
                else:
                    circuit, partitions = get_tqec_cnot_rotated(distance_scale = distance_scale,
                                                            n1 = 1,
                                                            n2 = 0)
                    
                    # Transpile circuit to backend
                    _, custom_circuit, _, _ = transpile_stim_circuit(circuit,
                                                                    backend,
                                                                    pre_defined_partitions = partitions,
                                                                    routing_type = "cost",
                                                                    routing_alpha = 0,
                                                                    routing_beta = 0)
                    # Convert circuit to stim
                    custom_circuit_stim = get_stim_circuits_with_detectors(custom_circuit)[0][0]
                    # Add circuit to dictionary, in order to not transpile this circuit configuration again
                    transpiled_circuits[(distance_scale, n_icc, p_icc, amp_icc)] = custom_circuit_stim

                    plot_circuit_layout(custom_circuit,
                                        backend,
                                        filename=f"experiments/evaluation/qec_evaluation/backend_mapping/layout_{distance_scale}.png")

                    return custom_circuit_stim

        def get_backend(p_icc: float, amp_icc: float, t: str = "", d: int = -1) -> BackendChipletV2:
            if t == "default":
                return None
            else:
                # Depending on the distance, each chiplet needs to be scaled
                print(d)

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

                backend = BackendChipletV2(size=chiplet_size,#size = (6, 6, 15, 8),
                                            n_inter = nic,
                                            connectivity = "nn",
                                            topology = "rotated_grid",
                                            inter_chiplet_noise = p_icc,
                                            inter_chiplet_amplification = amp_icc,
                                            inter_chiplet_noise_type = "random",#"constant"
                                            num_defective_qubits=0,
                                            )

                #plot_gate_map(backend = backend,
                #              filename = f"experiments/evaluation/qec_evaluation/backend_mapping/{chiplet_size}.png")
                return nic, backend

        def _get_sinter_task():
            # Construct sinter task for multiple code distances and noise levels
            yield from (
                sinter.Task(
                    circuit=circuit,
                    # TODO: the naming is incorrect
                    json_metadata={"d": 2 * k + 1, "r": 2 * k + 1, "p": p, "run_name": t,},
                )
                for circuit, k, p, t in (
                    (get_noise_model("modsi1000",
                                    None,
                                    p,
                                    None, 
                                    remote = (None if t == "default" else
                                            get_backend(d = k,
                                                        p_icc = ps_inter,
                                                        amp_icc = 1,
                                                        t = t)[1].inter_chiplet_connections)
                                    ).noisy_circuit(
                                        get_circuit(distance_scale = k,
                                                    p_icc = ps_inter,
                                                    amp_icc = 1,
                                                    t = t)
                                        ), k, p, t)
                    
                    for t in ts
                    for k in ks
                    for p in ps
                )
            )

        stats = run_sinter_simulation(_get_sinter_task, ks, ps)

        plot_evaluation(stats,
                        filename = f"experiments/evaluation/qec_evaluation/single_cnot_rotated_{ps_inter}.png",
                        with_transpilation = True)
    

if __name__ == "__main__":
    run_exp_distributed_lattice_surgery()