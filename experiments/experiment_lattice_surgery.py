from __future__ import annotations

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))
sys.path.append(os.path.join(os.getcwd(), "glue/eccentric_bench/"))


# Enable debug logging for Qiskit
#import logging
#logging.basicConfig(level=logging.DEBUG)
import time
import numpy as np

# Custom utils
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.circuit_generator import QECCircuit
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit
from glue.qiskit_qec.stim_tools import get_stim_circuits_with_detectors
from glue.eccentric_bench.noise import get_noise_model
from glue.eccentric_bench.backends import QubitTracking


# tqec
from tqec.utils.enums import Basis
from tqec.utils.noise_model import NoiseModel
import sinter
import multiprocessing

# Plotting
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def _transpile_to_tqec(circuit, backend) -> tuple[float, float]:
    """Transpilation of circuit to backend using custom and sabre transpilation passes

    :param circuit: _description_
    :type circuit: qiskit.QuantumCircuit
    :param backend: _description_
    :type backend: BackendChipletV2
    :return: _description_
    :rtype: tuple[float, float]
    """

    # Stim to qiskit
    stim_code_circuit = StimCodeCircuit(stim_circuit = circuit)

    custom_circuit = custom_partitioned_transpilation(stim_code_circuit.qc, backend)
    sabre_circuit = sabre_transpilation(stim_code_circuit.qc, backend)

    # Qiskit to stim
    custom_circuit_stim = get_stim_circuits_with_detectors(custom_circuit)[0][0]
    sabre_circuit_stim = get_stim_circuits_with_detectors(sabre_circuit)[0][0]


    #print(stim_code_circuit)
    #print(sabre_circuit_stim)


    return custom_circuit_stim, custom_circuit, sabre_circuit_stim, sabre_circuit


def _get_circuit(type, distance_scale: int = 1):
    # Distance must be multiples of 3

    # Get circuit with non-interacting patches
    lattice_surgery_circuit = QECCircuit()

    if type == "cnot":
        qiskit_circuit, stim_circuit = lattice_surgery_circuit.single_cnot(distance_scale = distance_scale)
    elif type == "three_cnot":
        qiskit_circuit, stim_circuit = lattice_surgery_circuit.three_cnot(distance_scale = distance_scale)
    elif type == "steane":
        qiskit_circuit, stim_circuit = lattice_surgery_circuit.steane_encoding(distance_scale = distance_scale)
  
    return qiskit_circuit, stim_circuit


def _run_sinter_simulation(tasks_fct, ks, ps):
    stats = sinter.collect(
        num_workers = int(multiprocessing.cpu_count()/2),#multiprocessing.cpu_count(),
        tasks=(tasks_fct()),
        save_resume_filepath = None,
        progress_callback=None,
        max_shots=1_000_000,
        max_errors=5_000,
        decoders=["pymatching"],
        print_progress=True,
        hint_num_tasks=len(ks) * len(ps),
        count_observable_error_combos=True,
    )

    return stats


def plot_sinter_stats(stat, filename, with_transpilation = False):
    fig, ax = plt.subplots()

    if with_transpilation:
        grp_fc = lambda stat: (
            stat.json_metadata["d"],
            stat.json_metadata["transpilation"]
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
    ax.set_title("Logical CNOT Error Rate")
    ax.set_xlabel("Physical Error Rate")
    ax.set_ylabel("Logical Error Rate")
    fig.savefig(filename)


def get_transpiled_circuit_as_sinter_task(backend, circuit_type) -> sinter.TaskStats:
    """Calculate logical error rate for transpiled circuit

    :param backend: _description_
    :type backend: _type_
    :param circuit_type: _description_
    :type circuit_type: _type_
    :return: _description_
    :rtype: sinter.TaskStats
    :yield: _description_
    :rtype: Iterator[sinter.TaskStats]
    """

    # Code distance to consider
    ks = [1, 2, 3]
    
    # Noise level
    ps = list(np.logspace(-4, -1, 10))

    # Transpilation
    ts = ["custom", "sabre"]

    circuits = {
        # TODO: add observable to measure
        k: (
            #_transpile_to_tqec(lattice_surgery_circuit.single_cnot(distance_scale = k)[1], backend)#[1]
            _transpile_to_tqec(_get_circuit(type = circuit_type, distance_scale = k)[1], backend)
        )
        for k in ks
    }

    def _get_sinter_task():
        # Construct sinter task for multiple code distances and noise levels
        yield from (
            sinter.Task(
                circuit=circuit,
                json_metadata={"d": 2 * k + 1, "r": 2 * k + 1, "p": p, "transpilation": t},
            )
            for circuit, k, p, t in (
                # Add noise to circuit using eccentric_bench noisy_circuit.
                # Note: This needs the QubitTracking
                ((get_noise_model("constant",
                                  QubitTracking(backend, circuit[1 if t == "custom" else 3]),
                                  p,
                                  backend)).noisy_circuit(circuit[0 if t == "custom" else 2]), k, p, t)
                for k, circuit in circuits.items()
                for p in ps
                for t in ts
            )
        )

    stat = _run_sinter_simulation(_get_sinter_task, ks, ps)

    return stat


def get_circuit_as_sinter_task(circuit_type) -> sinter.TaskStats:
    """Calculate logical error rate for circuit

    :param circuit_type: _description_
    :type circuit_type: _type_
    :return: _description_
    :rtype: sinter.TaskStats
    :yield: _description_
    :rtype: Iterator[sinter.TaskStats]
    """


    # Code distance to consider
    ks = [1, 2, 3]

    # Noise model
    noise_model_factory = NoiseModel.uniform_depolarizing
    # Noise level
    ps = list(np.logspace(-4, -1, 10))
    # Construct circuit and noise models
    noise_models = {p: noise_model_factory(p) for p in ps}

    circuits = {
        k: (_get_circuit(type = circuit_type, distance_scale = k)[1])
        for k in ks
    }

    def _get_sinter_task():
        # Construct sinter task for multiple code distances and noise levels
        yield from (
            sinter.Task(
                circuit=circuit,
                json_metadata={"d": 2 * k + 1, "r": 2 * k + 1, "p": p},
            )
            for circuit, k, p in (
                (nm.noisy_circuit(circuit), k, p)
                for k, circuit in circuits.items()
                for p, nm in noise_models.items()
            )
        )

    stat = _run_sinter_simulation(_get_sinter_task, ks, ps)

    return stat


if __name__ == "__main__":

    # Single CNOT
    n_inter = 5
    small_backend = BackendChipletV2((2, 2, 10, 10), n_inter)
    small_backend = BackendChipletV2((4, 4, 10, 10), n_inter)

    # Types of simple gates
    gates = ["three_cnot"]#["cnot", "three_cnot"]#, "steane"]

    for gate in gates:

        transpiled_stat = get_transpiled_circuit_as_sinter_task(small_backend, gate)
        plot_sinter_stats(transpiled_stat,
                        filename = f"experiments/data/tqec/figures/transpiled_{gate}_logical_error.png",
                        with_transpilation = True)

        non_transpiled_stat = get_circuit_as_sinter_task(gate)
        plot_sinter_stats(non_transpiled_stat,
                        filename = f"experiments/data/tqec/figures/{gate}_logical_error.png",
                        with_transpilation = False)

