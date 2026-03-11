import os
import sys

sys.path.append(os.path.join(os.getcwd(), "."))

# MECH
sys.path.append(os.path.join(os.getcwd(), "./external/baseline/MECH"))
import networkx as nx
import numpy as np

# Qiskit
import qiskit
import stim
from qiskit.transpiler import CouplingMap
from qiskit.visualization import plot_coupling_map

from external.baseline.MECH.Chiplet import *
from external.baseline.MECH.Circuit import *
from external.baseline.MECH.HighwayOccupancy import *
from external.baseline.MECH.MECHBenchmarks import *
from external.baseline.MECH.Router import *
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit

from pathlib import Path


def generate_simple_backend(x_num: int, y_num: int, icc_num: int = None) -> tuple[nx.Graph, int, int]:
    """Generate simple backend using MECH library.

    The backend generated with MECH can be used by Qiskit, QECC-Synth and by MECH itself. This is the fastest solution for now,
    to generate a backend that contains the highway structure (necessary for MECH to work).

    Args:
        x_num (int): _description_
        y_num (int): _description_

    Returns:
        tuple[nx.Graph, int, int]: _description_
    """
    structure = "square"
    chip_col_num = 2
    chip_row_num = 2

    if icc_num == None:
        icc_num = x_num

    # Generate layout of chip by connecting smaller chiplets
    G = gen_chiplet_array(structure, chip_col_num, chip_row_num, x_num, y_num, cross_link_sparsity=icc_num)
    # Add highway (for MECH)
    gen_highway_layout(G)

    qubit_num = len(G.nodes)
    data_qubit_num = len(G.nodes) - len(get_highway_qubits(G))

    print(
        f"Generated backend! \n #qubits: {len(G.nodes)} \n #qubits considering highway (only relevant for MECH): {data_qubit_num}"
    )

    return G, qubit_num, data_qubit_num


def coupling_to_adjacency(coupling_map: list) -> np.array:
    """Generate adjacency matrix from coupling map

    Args:
        coupling_map (list): Coupling map for backend as list containing (qubit_index, neighbour_index)

    Returns:
        np.array: Adjacency matrix
    """
    # Extract unique qubits and sort them
    qubits = sorted({q for pair in coupling_map for q in pair})
    qubit_index = {q: i for i, q in enumerate(qubits)}
    # Initialize adjacency matrix with zeros
    size = len(qubits)
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    # Fill in the adjacency matrix
    for a, b in coupling_map:
        i, j = qubit_index[a], qubit_index[b]
        matrix[i][j] = 1
        matrix[j][i] = 1  # Assuming undirected graph

    return np.array(matrix)


def idx_dict_to_list(mapping_dict: dict) -> list:
    """Construct qubit index list

    Args:
        mapping_dict (dict): Qubit indices

    Returns:
        list: Indices
    """
    # Step 1: Group by row
    from collections import defaultdict

    row_groups = defaultdict(list)
    for (row, col), val in mapping_dict.items():
        row_groups[row].append(col)

    # Step 2: Sort columns within each row
    for row in row_groups:
        row_groups[row].sort()

    # Step 3: Convert to [row_index, sequential_index] list
    result = []
    for row in sorted(row_groups.keys()):
        for seq_index, col in enumerate(row_groups[row]):
            result.append([row - builtins.min(row_groups.keys()), seq_index])

    return result


def generate_qecc_synth_backend_from_mech(G: nx.graph) -> tuple[np.array, list]:
    """Generate backend for QECC-Synth given MECH backend

    Args:
        G (nx.graph): Backend

    Returns:
        tuple[np.array, list]: Adjacency graph and qubit index list
    """
    qubit_idx_dict = gen_qubit_idx_dict(G)
    regular_coupling = list([qubit_idx_dict[n1], qubit_idx_dict[n2]] for n1, n2 in G.edges)
    regular_coupling += list([qubit_idx_dict[n2], qubit_idx_dict[n1]] for n1, n2 in G.edges)

    CG = coupling_to_adjacency(regular_coupling)
    qubit_idx_dict = idx_dict_to_list(qubit_idx_dict)

    return (CG, qubit_idx_dict)


def generate_qiskit_backend_from_mech(G: nx.graph) -> CouplingMap:
    """Generate backend to be used by qiskit

    Args:
        G (nx.graph): _description_

    Returns:
        CouplingMap: _description_
    """
    qubit_idx_dict = gen_qubit_idx_dict(G)
    regular_coupling = list([qubit_idx_dict[n1], qubit_idx_dict[n2]] for n1, n2 in G.edges)
    regular_coupling += list([qubit_idx_dict[n2], qubit_idx_dict[n1]] for n1, n2 in G.edges)

    return CouplingMap(regular_coupling)


def display_simple_backend(backend: nx.Graph, filename: str) -> None:
    """Generate figure of coupling graph for specified backend

    Args:
        backend (nx.Graph): _description_
        filename (str): _description_
    """
    qubit_idx_dict = gen_qubit_idx_dict(backend)
    regular_coupling = list([qubit_idx_dict[n1], qubit_idx_dict[n2]] for n1, n2 in backend.edges)
    regular_coupling += list([qubit_idx_dict[n2], qubit_idx_dict[n1]] for n1, n2 in backend.edges)

    cm = CouplingMap(couplinglist=regular_coupling)

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    plot_coupling_map(cm.size(), None, cm.get_edges(), filename=filename)


def get_surface_code_stim(d, T=None):
    if T == None:
        T = d

    stim_circuit = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=T, distance=d)
    return StimCodeCircuit(stim_circuit=stim_circuit)


def calc_circuit_qiskit_stats(
    transpiled_circuit: qiskit.circuit, G: nx.graph, qeccsynth_result=None, initial_circuit=None
) -> dict:
    """_summary_

    Args:
        transpiled_circuit (qiskit.circuit): _description_
        G (nx.graph): _description_

    Returns:
        dict: _description_
    """
    # Parameter
    cross_chip_gate_weight = 7.4

    # Decompose swap gates
    filter_function = lambda gate: gate.operation.num_qubits >= 2
    swap_decomposed_circuit = transpiled_circuit.decompose("swap")

    # Calculate depth
    swap_decomposed_depth = 0
    swap_decomposed_depth += swap_decomposed_circuit.depth(filter_function)

    # Count instructions
    within_chip_cnots = 0
    cross_chip_cnots = 0
    for instr, qargs, cargs in swap_decomposed_circuit.data:
        # Only care about 2-qubit gates
        if instr.num_qubits < 2:
            continue

        # Extract physical qubit indices
        q1 = swap_decomposed_circuit.qubits.index(qargs[0])
        q2 = swap_decomposed_circuit.qubits.index(qargs[1])

        idx_qubit_dict = gen_idx_qubit_dict(G)
        edge = (idx_qubit_dict[q1], idx_qubit_dict[q2])

        if instr.name in ["cx", "cp"] and G.edges[edge]["type"] == "on_chip":
            within_chip_cnots += 1
        elif instr.name == "swap" and G.edges[edge]["type"] == "on_chip":
            within_chip_cnots += 3
        elif instr.name in ["cx", "cp"] and G.edges[edge]["type"] == "cross_chip":
            cross_chip_cnots += 1
        elif instr.name == "swap" and G.edges[edge]["type"] == "cross_chip":
            cross_chip_cnots += 3

    # Calculate effective number of CNOT gates (for calculation, see section 7.1 of "MECH: Multi-Entry Communication Highway for Superconducting Quantum Chiplets")
    norm_cnots = within_chip_cnots + cross_chip_cnots * cross_chip_gate_weight

    if qeccsynth_result == None and initial_circuit != None:
        # Calculate gate overhead using initial circuit

        def num_2q_gates(circuit):
            ops = circuit.count_ops()
            two_qubit_gate_names = ["cx", "cz", "swap"]
            return sum(ops.get(g, 0) for g in two_qubit_gate_names)

        two_qubit_overhead = (within_chip_cnots + cross_chip_cnots) - num_2q_gates(initial_circuit)

        result_qiskit = {
            "2q_gates_overhead": two_qubit_overhead,
            "depth": swap_decomposed_depth,
            "eff_gate_num": norm_cnots,
            "on-chip": within_chip_cnots,
            "cross-chip": cross_chip_cnots,
        }
    else:
        # Calculate gate overhead using results file. Only applicable ot QECC-Synth

        # If qeccsynth results are available, add 2q-gate overhead
        # Calculation of CNOT overhead taken from QECC-Synth - CodeStitch.py
        cnotNum = 0
        for k in range(qeccsynth_result["chunkNum"]):
            for s in qeccsynth_result["Swap_layer"][k]:
                cnotNum += 3
        for k in range(qeccsynth_result["chunkNum"]):
            for k, stab in enumerate(qeccsynth_result["Stab"][k]):
                cnotNum += len(stab["Ancilla"]) * 2 - 2

        two_qubit_overhead = cnotNum
        result_qiskit = {
            "2q_gates_overhead": two_qubit_overhead,
            "depth": swap_decomposed_depth,
            "eff_gate_num": norm_cnots,
            "on-chip": within_chip_cnots,
            "cross-chip": cross_chip_cnots,
        }

    return result_qiskit


def calc_circuit_mech_stats(router: Router, initial_circuit) -> dict:
    """_summary_

    Args:
        router (Router): _description_

    Returns:
        dict: _description_
    """
    # Parameter
    cross_chip_gate_weight = 7.4
    meas_weight = 2.2

    on_chip_gate_num = 0
    cross_chip_gate_num = 0
    meas_num = 0

    for idx in range(router.circuit.depth):
        for line in range(len(router.circuit.circuit_lines)):
            if router.circuit.take_role(line, idx) == "q":
                meas_num += 1
            if router.circuit.take_role(line, idx) in ["t", "mt"]:
                node = router.circuit.take_node(line, idx)
                # print(node.control)
                # print(isinstance(node, Circuit.MOpNode))
                # if isinstance(node, OpNode):
                # print(isinstance(node, Circuit.OpNode))
                # print()
                # print(type(node).__module__)
                # print(MOpNode)
                # print(MOpNode.__module__)

                if type(node).__name__ == "OpNode":
                    control_line = node.control
                if type(node).__name__ == "MOpNode":
                    control_line = node.shared
                control_qubit, target_qubit = (
                    router.highway_manager.idx_qubit_dict[control_line],
                    router.highway_manager.idx_qubit_dict[line],
                )

                if (
                    router.chip.has_edge(control_qubit, target_qubit)
                    and router.chip.edges[(control_qubit, target_qubit)]["type"] == "cross_chip"
                ):
                    cross_chip_gate_num += 1
                else:
                    on_chip_gate_num += 1

    def num_2q_gates(circuit):
        ops = circuit.count_ops()
        two_qubit_gate_names = ["cx", "cz", "swap"]
        return sum(ops.get(g, 0) for g in two_qubit_gate_names)

    initial_2q_gates = num_2q_gates(initial_circuit)

    # Calculate effective number of CNOT gates (for calculation, see section 7.1 of "MECH: Multi-Entry Communication Highway for Superconducting Quantum Chiplets")
    eff_gate_num = on_chip_gate_num + cross_chip_gate_num * cross_chip_gate_weight + meas_num * meas_weight

    # Collect statistics
    result_mech = {
        "initial_2q_gates": initial_2q_gates,
        "depth": router.circuit.depth,
        "2q_gates_overhead": (on_chip_gate_num + cross_chip_gate_num) - initial_2q_gates,
        "eff_gate_num": eff_gate_num,
        "on-chip": on_chip_gate_num,
        "cross-chip": cross_chip_gate_num,
        "meas_num": meas_num,
        "shuttle_num": len(router.highway_manager.shuttle_stack),
    }

    return result_mech
