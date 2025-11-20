# How is the logical error rate influenced by a limited number of inter-chiplet connections?
from __future__ import annotations

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))
sys.path.append(os.path.join(os.getcwd(), "glue/eccentric_bench/"))


# Enable debug logging for Qiskit
import time
import numpy as np
from collections import defaultdict


# Custom utils
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.circuit_generator import QECCircuit
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit
from glue.qiskit_qec.stim_tools import get_stim_circuits_with_detectors
from glue.eccentric_bench.noise import get_noise_model
from glue.eccentric_bench.backends import QubitTracking

# Stim and sinter
import math
from typing import Callable, TypeVar, List, Any, Iterable, Optional, TYPE_CHECKING, Dict, Union, Literal, Tuple
from typing import Sequence
from typing import cast
from sinter._probability_util import fit_binomial, shot_error_rate_to_piece_error_rate, Fit

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

    return custom_circuit_stim, custom_circuit, sabre_circuit_stim, sabre_circuit


def _get_circuit(type, distance_scale: int = 1):
    # Distance must be multiples of 3

    # Get circuit with non-interacting patches
    lattice_surgery_circuit = QECCircuit()


    qiskit_circuit, stim_circuit = lattice_surgery_circuit.single_cnot(distance_scale = distance_scale)
  
    #print(stim_circuit)
    return qiskit_circuit, stim_circuit


def _run_sinter_simulation(tasks_fct, ks, ps):
    stats = sinter.collect(
        num_workers = int(multiprocessing.cpu_count()/2),#multiprocessing.cpu_count(),
        tasks=(tasks_fct()),
        save_resume_filepath = None,
        progress_callback=None,
        max_shots=10_000_000, #10_000_000,
        max_errors=5_000,
        decoders=["pymatching"],
        print_progress=True,
        hint_num_tasks=len(ks) * len(ps),
        count_observable_error_combos=True,
    )

    return stats
    

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

    print("Start sinter tasks")

    # Code distance to consider
    ks = [1]#, 2, 3]
    
    # Noise level
    ps = list(np.logspace(-4, -1, 10))
    # TODO: Change to noise model that takes remote gates into consideration
    tqec_noise_model = NoiseModel.uniform_depolarizing

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
                (tqec_noise_model(p).noisy_circuit(circuit[0 if t == "custom" else 2]), k, p, t)
                # Add noise to circuit using eccentric_bench noisy_circuit.
                # Note: This needs the QubitTracking
                #((get_noise_model("constant",
                #                  None, #QubitTracking(backend, circuit[1 if t == "custom" else 3]),
                #                  p,
                #                  None)#backend)
                #                  ).noisy_circuit(circuit[0 if t == "custom" else 2]), k, p, t)
                for k, circuit in circuits.items()
                for p in ps
                for t in ts
            )
        )
    print("Starting simulation")
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
    ks = [1]#, 2, 3]

    # Noise level
    ps = list(np.logspace(-4, -1, 10))
    
    circuits = {
        k: (_get_circuit(type = circuit_type, distance_scale = k)[1])
        for k in ks
    }
    tqec_noise_model = NoiseModel.uniform_depolarizing
    
    def _get_sinter_task():
        # Construct sinter task for multiple code distances and noise levels
        yield from (
            sinter.Task(
                circuit=circuit,
                json_metadata={"d": 2 * k + 1, "r": 2 * k + 1, "p": p, "transpilation": "none"},
            )
            for circuit, k, p in (
                (tqec_noise_model(p).noisy_circuit(circuit), k, p)
                #((get_noise_model("constant",
                #                  None,
                #                  p,
                #                  None)).noisy_circuit(circuit), k, p)
                for k, circuit in circuits.items()
                for p in ps
            )
        )

    stat = _run_sinter_simulation(_get_sinter_task, ks, ps)

    return stat


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
    ax.set_title("Logical Error Rate")
    ax.set_xlabel("Physical Error Rate")
    ax.set_ylabel("Logical Error Rate")
    fig.savefig(filename)



if __name__ == "__main__":
    # Single CNOT
    n_inter = 6
    #small_backend = BackendChipletV2((2, 2, 10, 10), n_inter)
    small_backend = BackendChipletV2((2, 2, 6, 6), n_inter)

    # Types of simple gates
    gate = "cnot"

    # Calculate logical error rate
    print(f"Running {gate}")
    
    # Combine transpiled and non_transpiled stats
    transpiled_stat_with_interconnects = get_transpiled_circuit_as_sinter_task(small_backend, gate)
    non_transpiled_stat = get_circuit_as_sinter_task(gate)
    
    all_stats = transpiled_stat_with_interconnects.__add__(non_transpiled_stat)

    plot_sinter_stats(all_stats,
                      filename = f"experiments/evaluation/error_rate_limited_inter_chiplet.png",
                      with_transpilation = True)