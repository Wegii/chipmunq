# Explore how the mapping influences circuit for quantum memory




from __future__ import annotations

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))
sys.path.append(os.path.join(os.getcwd(), "glue/eccentric_bench/"))

# Custom utils
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.circuit_generator import QECMemory, GenericCircuit
#from experiments.exp_utils.circuit_statistics import QECCircuitStats
from glue.eccentric_bench.backends import QubitTracking

# Plotting
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Sinter and stim
import sinter
import multiprocessing
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit
from glue.qiskit_qec.stim_tools import get_stim_circuits_with_detectors
from glue.eccentric_bench.noise import get_noise_model


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
    base_palette = sns.color_palette("Set2", n_colors=len(d_values))
    colors = {}
    for c in range(len(d_values)):
        colors[c] = base_palette[c]

    for i, d in enumerate(sorted(d_values)):
        ps_custom = sorted(custom_diff_by_d[d].keys())
        ys_custom = [custom_diff_by_d[d][p] for p in ps_custom]
        plt.plot(ps_custom,
                 ys_custom,
                 marker='o',
                 color=colors[i],
                 linewidth=2,
                 label=f'({d}, custom)')

        ps_sabre = sorted(sabre_diff_by_d[d].keys())
        ys_sabre = [sabre_diff_by_d[d][p] for p in ps_sabre]
        plt.plot(ps_sabre,
                 ys_sabre,
                 marker='x',
                 color=colors[i],
                 linestyle='--',
                 linewidth=2,
                 label=f'({d}, sabre)')

    plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel("Physical Error Rate")
    plt.ylabel("Δ Logical Error Rate")
    #plt.title("Logical Error Rate Differences by d")
    # plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def _run_sinter_simulation(tasks_fct, ks, ps):
    stats = sinter.collect(
        num_workers = int(multiprocessing.cpu_count()*0.75),#multiprocessing.cpu_count(),
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

def _transpile_to_stim(circuit, backend) -> tuple[float, float]:
    """Transpilation of circuit to backend using custom and sabre transpilation passes

    :param circuit: _description_
    :type circuit: qiskit.QuantumCircuit
    :param backend: _description_
    :type backend: BackendChipletV2
    :return: _description_
    :rtype: tuple[float, float]
    """

    #print(circuit)

    # Stim to qiskit
    stim_code_circuit = StimCodeCircuit(stim_circuit = circuit)

    custom_circuit = custom_partitioned_transpilation(stim_code_circuit.qc, backend)
    sabre_circuit = sabre_transpilation(stim_code_circuit.qc, backend)

    # Qiskit to stim
    custom_circuit_stim = get_stim_circuits_with_detectors(custom_circuit)[0][0]
    sabre_circuit_stim = get_stim_circuits_with_detectors(sabre_circuit)[0][0]

    with open("experiments/data/stim/data/cnot.txt", "w") as f:
        print(circuit, file=f)

    with open("experiments/data/stim/data/cnot_transpiled_sabre.txt", "w") as f:
        print(sabre_circuit_stim, file=f)

    with open("experiments/data/stim/data/cnot_transpiled_custom.txt", "w") as f:
        print(custom_circuit_stim, file=f)


    return custom_circuit_stim, custom_circuit, sabre_circuit_stim, sabre_circuit


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
            _transpile_to_stim(_get_circuit(type = circuit_type, num_patches=1, distance_scale = k), backend)
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
                ((get_noise_model("si1000",
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
        k: _get_circuit(type = circuit_type, num_patches=1, distance_scale = k)
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
                ((get_noise_model("si1000",
                                  None,
                                  p,
                                  None)).noisy_circuit(circuit), k, p)
                for k, circuit in circuits.items()
                for p in ps
            )
        )

    stat = _run_sinter_simulation(_get_sinter_task, ks, ps)

    return stat


def _get_circuit(type, num_patches, distance_scale=1):
    # Get circuit with non-interacting patches

    distance = 9
    # calculate num_qubits for distance
    num_qubits = 100

    circuit_generator = QECMemory(num_qubits)
    # Selects the maximum code distance given the number of qubits
    # Note: This is a stim circuit!
    circuit = circuit_generator.generate_code_memory('surface', num_patches, distance_scale)

    return circuit


def _transpile(circuit: qiskit.QuantumCircuit, backend: BackendChipletV2) -> tuple[float, float]:
    """Time transpilation of circuit to backend using custom and sabre transpilation passes

    :param circuit: _description_
    :type circuit: qiskit.QuantumCircuit
    :param backend: _description_
    :type backend: BackendChipletV2
    :return: _description_
    :rtype: tuple[float, float]
    """

    custom_circuit = custom_partitioned_transpilation(circuit, backend)
    sabre_circuit = sabre_transpilation(circuit, backend)

    return custom_circuit, sabre_circuit


if __name__ == "__main__":
    n_inter = 5

    small_backend = BackendChipletV2((2, 2, 10, 10), n_inter)
    medium_backend = BackendChipletV2((8, 8, 10, 10), n_inter)
    big_backend = BackendChipletV2((16, 16, 10, 10), n_inter)

    """
    # Small backend
    small_generic_patch_circuit = _get_circuit("", 2*2)    
    custom_small, sabre_small = _transpile(small_generic_patch_circuit.qc, small_backend)

    print(custom_small.count_ops())
    print(sabre_small.count_ops())

    custom_small_stats = QECCircuitStats(transpiled_circuit = custom_small,
                                         stim_circuit = small_generic_patch_circuit,
                                         backend = small_backend)
    custom_small_error_rate = custom_small_stats.get_logical_error_rate(num_samples = 10000)

    sabre_small_stats = QECCircuitStats(transpiled_circuit = sabre_small,
                                         stim_circuit = small_generic_patch_circuit,
                                         backend = small_backend)

    sabre_small_error_rate = sabre_small_stats.get_logical_error_rate(num_samples = 10000)

    print(custom_small_error_rate)
    print(sabre_small_error_rate)
    """

    transpiled_stat = get_transpiled_circuit_as_sinter_task(small_backend, "surface")
    non_transpiled_stat = get_circuit_as_sinter_task("surface")    
    transpiled_stat = transpiled_stat.__add__(non_transpiled_stat)

    plot_sinter_stats(transpiled_stat,
                      filename = f"experiments/data/stim/figures/surface_memory_logical_error.png",
                      with_transpilation = True)
    
    plot_error_improvement(transpiled_stat,
                           filename="experiments/data/stim/figures/surface_memory_logical_error_difference.png")

    """
    sabre_small_stats = QECCircuitStats(transpiled_circuit = sabre_small, backend = small_backend)
    #basic_small_stats = QECCircuitStats(transpiled_circuit = basic_small, backend = small_backend)

    # Medium backend
    medium_generic_patch_circuit = _get_circuit("", 8*8)
    custom_medium, sabre_medium = _transpile(medium_generic_patch_circuit, medium_backend)
    custom_medium_stats = QECCircuitStats(transpiled_circuit = custom_medium, backend = medium_backend)
    sabre_medium_stats = QECCircuitStats(transpiled_circuit = sabre_medium, backend = medium_backend)
    #basic_medium_stats = QECCircuitStats(transpiled_circuit = basic_medium, backend = medium_backend)

    # Big backend
    big_generic_patch_circuit = _get_circuit("", 16*16)
    custom_big, sabre_big = _transpile(big_generic_patch_circuit, big_backend)
    custom_big_stats = QECCircuitStats(transpiled_circuit = custom_big, backend = big_backend)
    sabre_big_stats = QECCircuitStats(transpiled_circuit = sabre_big, backend = big_backend)
    #basic_big_stats = QECCircuitStats(transpiled_circuit = basic_big, backend = big_backend)
    
    """
