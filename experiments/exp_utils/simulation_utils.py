import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import sinter
from tqec.utils.noise_model import NoiseModel

from experiments.exp_utils.transpilation_utils import *
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit
from glue.qiskit_qec.stim_tools import get_stim_circuits_with_detectors


def transpile_stim_circuit(
    circuit: StimCodeCircuit,
    backend: BackendChipletV2,
    pre_defined_partitions: list = None,
    routing_type: str = "",
    routing_alpha: float = 0.0,
    routing_beta: float = 0.0,
) -> tuple[StimCodeCircuit, QuantumCircuit, StimCodeCircuit, QuantumCircuit]:
    """Transpilation of circuit to backend using custom and sabre transpilation passes

    :param circuit: _description_
    :type circuit: StimCodeCircuit
    :param backend: _description_
    :type backend: BackendChipletV2
    :param pre_defined_partitions: _description_, defaults to None
    :type pre_defined_partitions: list, optional
    :param routing_type: _description_, defaults to ""
    :type routing_type: str, optional
    :return: _description_
    :rtype: tuple[StimCodeCircuit, QuantumCircuit, StimCodeCircuit, QuantumCircuit]
    """
    # Stim to qiskit
    stim_code_circuit = StimCodeCircuit(stim_circuit=circuit)

    # Transpile using custom implementation
    if routing_type == "default":
        custom_circuit = custom_partitioned_transpilation(
            stim_code_circuit.qc, backend, pre_defined_partitions=pre_defined_partitions
        )
    else:
        custom_circuit = custom_cost_transpilation(
            stim_code_circuit.qc,
            backend,
            pre_defined_partitions=pre_defined_partitions,
            routing_alpha=routing_alpha,
            routing_beta=routing_beta,
        )

    # Transpile using standard SABRE
    sabre_circuit = None  # sabre_transpilation(stim_code_circuit.qc, backend)

    # Qiskit to stim
    custom_circuit_stim = get_stim_circuits_with_detectors(custom_circuit)[0][0]
    sabre_circuit_stim = None  # get_stim_circuits_with_detectors(sabre_circuit)[0][0]

    return custom_circuit_stim, custom_circuit, sabre_circuit_stim, sabre_circuit


def run_sinter_simulation(tasks_fct, ks, ps, num_shots: int = 100_000_000, num_t=None):
    if num_t == None:
        num_t = int(multiprocessing.cpu_count() / 2)

    stats = sinter.collect(
        num_workers=num_t,  # multiprocessing.cpu_count(),
        tasks=(tasks_fct()),
        save_resume_filepath=None,
        progress_callback=None,
        max_shots=num_shots,  # 10_000_000,
        max_errors=5_000,
        decoders=["pymatching"],
        print_progress=True,
        hint_num_tasks=len(ks) * len(ps),
        count_observable_error_combos=True,
    )

    return stats


def run_simulation_transpiled_circuit(circuit_generator, backend):

    # Code distance to consider
    ks = [1]

    circuits = {
        # TODO: add observable to measure
        k: (
            # _transpile_to_tqec(lattice_surgery_circuit.single_cnot(distance_scale = k)[1], backend)#[1]
            transpile_stim_circuit(circuit_generator, backend)
        )
        for k in ks
    }

    return sinter_simulation(circuits, ks)


def sinter_simulation(circuits, ks):
    # Noise level
    ps = list(np.logspace(-4, -1, 10))
    # TODO: Change to noise model that takes remote gates into consideration
    tqec_noise_model = NoiseModel.uniform_depolarizing

    # Transpilation
    ts = ["custom", "sabre"]

    def _get_sinter_task():
        # Construct sinter task for multiple code distances and noise levels
        yield from (
            sinter.Task(
                circuit=circuit,
                json_metadata={"d": 2 * k + 1, "r": 2 * k + 1, "p": p, "transpilation": t},
            )
            for circuit, k, p, t in (
                (tqec_noise_model(p).noisy_circuit(circuit[0 if t == "custom" else 2]), k, p, t)
                for k, circuit in circuits.items()
                for p in ps
                for t in ts
            )
        )

    transpiled_stat = run_sinter_simulation(_get_sinter_task, ks, ps)

    return transpiled_stat


def plot_sinter_stats(stat, filename, with_transpilation=False):
    fig, ax = plt.subplots()

    if with_transpilation:
        grp_fc = lambda stat: (stat.json_metadata["d"], stat.json_metadata["transpilation"])
    else:
        grp_fc = lambda stat: stat.json_metadata["d"]
    sinter.plot_error_rate(
        ax=ax,
        stats=stat,
        x_func=lambda stat: stat.json_metadata["p"],
        group_func=grp_fc,
    )
    # plot_observable_as_inset(ax, zx_graph, correlation_surfaces[i])
    ax.grid(axis="both")
    ax.legend()
    ax.loglog()
    ax.set_title("Logical Error Rate")
    ax.set_xlabel("Physical Error Rate")
    ax.set_ylabel("Logical Error Rate")
    fig.savefig(filename)
