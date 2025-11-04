# Explore how the mapping influences circuit for quantum memory





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

    # Noise level
    ps = list(np.logspace(-4, -1, 10))
    
    circuits = {
        k: (_get_circuit(type = circuit_type, distance_scale = k)[1])
        for k in ks
    }
    
    def _get_sinter_task():
        # Construct sinter task for multiple code distances and noise levels
        yield from (
            sinter.Task(
                circuit=circuit,
                json_metadata={"d": 2 * k + 1, "r": 2 * k + 1, "p": p, "transpilation": "none"},
            )
            for circuit, k, p in (
                #(nm.noisy_circuit(circuit), k, p)
                ((get_noise_model("constant",
                                  None,
                                  p,
                                  None)).noisy_circuit(circuit), k, p)
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


def plot_error_improvement(stats, filename):
    # TODO: Show improvement of custom and sabre over transpilation="none"
    #error_by_t = {task.t: task.errors / task.shots for task in stats}
    from collections import defaultdict
    import matplotlib.pyplot as plt

    # Calculate error rate and group
    error_rates = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    physical_error_rates = set()
    d_values = set()
    for s in stats:
        ler = s.errors / (s.shots - s.discards)
        p = s.json_metadata['p']
        t = s.json_metadata['transpilation']
        d = s.json_metadata['d']

        error_rates[t][p][d].append(ler)
        physical_error_rates.add(p)
        d_values.add(d)

    print(error_rates)
    # Compute difference in error_rate
    custom_diff_by_d = defaultdict(dict)
    sabre_diff_by_d = defaultdict(dict)

    for d in d_values:
        for p in physical_error_rates:
            if p in error_rates['none'] and d in error_rates['none'][p]:
                # Custom vs none
                if 'custom' in error_rates and d in error_rates['custom'][p]:
                    # take only value for this p,d pair
                    custom_diff_by_d[d][p] = error_rates['custom'][p][d][0] - error_rates['none'][p][d][0]

                # Sabre vs none
                if 'sabre' in error_rates and d in error_rates['sabre'][p]:
                    sabre_diff_by_d[d][p] = error_rates['sabre'][p][d][0] - error_rates['none'][p][d][0]

    plt.figure(figsize=(8,6))

    # Plot error rate difference for each transpilation method and distance
    for d in sorted(d_values):
        ps_custom = sorted(custom_diff_by_d[d].keys())
        ys_custom = [custom_diff_by_d[d][p] for p in ps_custom]
        plt.plot(ps_custom, ys_custom, marker='o', label=f'custom, d={d}')

        ps_sabre = sorted(sabre_diff_by_d[d].keys())
        ys_sabre = [sabre_diff_by_d[d][p] for p in ps_sabre]
        plt.plot(ps_sabre, ys_sabre, marker='x', label=f'sabre, d={d}')

    plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel("Physical Error Rate (p)")
    plt.ylabel("Delta Logical Error Rate")
    plt.title("Logical Error Rate Differences by d")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    

def plot_gate_overhead(stats, filename):
    """Plot two-qubit gate overhead.

    Plot the two-qubit gate overhead over multiple circuits for custom and sabre transpilation.
    Extract the gate overhead of the lattice surgery circuit using code distance = 7.

    :param stats: _description_
    :type stats: _type_
    :param filename: _description_
    :type filename: _type_
    """

    base_palette = sns.color_palette("pastel", n_colors=2*3)
    colors = {}
    for c in range(len(stats.keys())):
        colors[c] = base_palette[c+1]

    fig, ax = plt.subplots(figsize=(10, 6))

    width = 0.35
    x = np.arange(len(stats.keys()))

    for i, gate in enumerate(stats.keys()):
        # Extract gate overhead for distance = 2*3 + 1 = 7 
        custom_gates = ((stats[gate])["3"])["custom"]
        sabre_gates = ((stats[gate])["3"])["sabre"]

        if i == 0:
            ax.bar(x[i] - width/2, sabre_gates, width, hatch='/', color=colors[0], edgecolor='black', label="SABRE")
            ax.bar(x[i] + width/2, custom_gates, width, hatch='o', color=colors[1], edgecolor='black', label="Custom")
        else:
            ax.bar(x[i] - width/2, sabre_gates, width, hatch='/', color=colors[0], edgecolor='black')
            ax.bar(x[i] + width/2, custom_gates, width, hatch='o', color=colors[1], edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(stats.keys(), fontsize=12)
    ax.set_ylabel('Two-Qubit Gate Overhead', fontsize=12)
    ax.set_xlabel('Lattice Surgery Circuit', fontsize=12)
    #ax.set_yscale("log")
    ax.text(-0.025, 1.05, 'Lower is better ↓', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
    ax.legend(title="Method",
              fontsize=10,
              loc='center left',
              bbox_to_anchor=(1.02, 0.5))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":

    # Single CNOT
    n_inter = 5
    #small_backend = BackendChipletV2((2, 2, 10, 10), n_inter)
    small_backend = BackendChipletV2((2, 2, 10, 10), n_inter)

    # Types of simple gates
    gates = ["cnot", "three_cnot", "steane"]

    # Calculate logical error rate
    for gate in gates:
        print(f"Running {gate}")
        
        # Combine transpiled and non_transpiled stats
        transpiled_stat = get_transpiled_circuit_as_sinter_task(small_backend, gate)
        non_transpiled_stat = get_circuit_as_sinter_task(gate)
        
        transpiled_stat = transpiled_stat.__add__(non_transpiled_stat)

        plot_sinter_stats(transpiled_stat,
                          filename = f"experiments/data/tqec/figures/{gate}_logical_error.png",
                          with_transpilation = True)
        
        # Plot improvement (or worsening) of logical error rate
        plot_error_improvement(transpiled_stat, f"experiments/data/tqec/figures/{gate}_logical_error_diff.png")
        

    """
    # Count gate overhead before and after transpilation
    ks_gates = {}
    for gate in gates:
        # Calculate gate overhead for distance = 2*ks + 1 = 7 circuit
        ks = [3]#[1, 2, 3]
        ks_results = {}
        
        for k in ks:
            circuit = _get_circuit(type = gate, distance_scale = k)[0]
            custom_circuit_stim, custom_circuit, sabre_circuit_stim, sabre_circuit = _transpile_to_tqec(
                _get_circuit(type = gate, distance_scale = k)[1], small_backend)
            
            gates = sum(1 for instr, qargs, cargs in (circuit.qc).data if len(qargs) == 2)
            custom_gates = sum(1 for instr, qargs, cargs in (custom_circuit).data if len(qargs) == 2)
            sabre_gates = sum(1 for instr, qargs, cargs in (sabre_circuit).data if len(qargs) == 2)

            ks_results[str(k)] = {"custom": custom_gates - gates, "sabre": sabre_gates - gates}

        ks_gates[gate] = ks_results

    plot_gate_overhead(ks_gates, filename = f"experiments/data/tqec/figures/all_circuit_gate_overhead.png")
    """
