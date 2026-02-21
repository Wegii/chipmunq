import os
import sys

sys.path.append(os.path.join(os.getcwd(), "."))

import itertools
import logging
import pickle

import numpy as np
from tqdm import tqdm

# MECH
sys.path.append(os.path.join(os.getcwd(), "./external/baseline/MECH"))
import networkx as nx

from external.baseline.MECH.Chiplet import *
from external.baseline.MECH.Circuit import *
from external.baseline.MECH.HighwayOccupancy import *
from external.baseline.MECH.MECHBenchmarks import *
from external.baseline.MECH.Router import *
from external.baseline.MECH.transpile_mech import transpile_circuit_MECH

# QECC-Synth
sys.path.append(os.path.join(os.getcwd(), "./external/baseline/QECC_Synth/SurfStitch/MyCode/src"))
from external.baseline.QECC_Synth.SurfStitch.MyCode.src.transpile_qeccsynth import transpile_circuit_QECCSynth

# SABRE
sys.path.append(os.path.join(os.getcwd(), "./src/baseline/SABRE"))
from external.baseline.SABRE.transpile_sabre import transpile_circuit_SABRE

# Eccentric_bench
sys.path.append(os.path.join(os.getcwd(), "../eccentric_bench/"))
sys.path.append(os.path.join(os.getcwd(), "../eccentric_bench/external/qiskit_qec/src"))
from codes import get_code, get_max_d
from qiskit.transpiler import CouplingMap

# Qiskit
from qiskit.visualization import plot_coupling_map


def generate_simple_dqc_backend(x_num: int, y_num: int, cross_link_sparsity: int) -> tuple[nx.Graph, int, int]:
    """Generate simple backend using MECH library.

    The backend generated with MECH can be used by Qiskit, QECC-Synth and by MECH itself. This is the fastest solution for now,
    to generate a backend that contains the highway structure (necessary for MECH to work).

    Args:
        x_num (int): _description_
        y_num (int): _description_
        cross_link_sparsity (int): _description_

    Returns:
        tuple[nx.Graph, int, int]: _description_
    """
    structure = "square"
    chip_col_num = 2
    chip_row_num = 2

    # Generate layout of chip by connecting smaller chiplets
    G = gen_chiplet_array(structure, chip_col_num, chip_row_num, x_num, y_num, cross_link_sparsity=cross_link_sparsity)
    # Add highway (for MECH)
    gen_highway_layout(G)

    qubit_num = len(G.nodes)
    data_qubit_num = len(G.nodes) - len(get_highway_qubits(G))

    print(f"Generated backend! \n #qubits: {len(G.nodes)} \n #qubits considering highway (only relevant for MECH): {data_qubit_num}")

    return G, qubit_num, data_qubit_num


def generate_qiskit_backend_from_mech(G: nx.graph) -> CouplingMap:
    """Generate backend to be used by qiskit

    Args:
        G (nx.graph): _description_

    Returns:
        CouplingMap: _description_
    """
    qubit_idx_dict = gen_qubit_idx_dict(G)
    regular_coupling = list([qubit_idx_dict[n1], qubit_idx_dict[n2]] for n1,n2 in G.edges)
    regular_coupling += list([qubit_idx_dict[n2], qubit_idx_dict[n1]] for n1,n2 in G.edges)
    
    return CouplingMap(regular_coupling)


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
        matrix[j][i] = 1 # Assuming undirected graph

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
            result.append([row - min(row_groups.keys()), seq_index])

    return result


def generate_qecc_synth_backend_from_mech(G: nx.graph) -> tuple[np.array, list]:
    """Generate backend for QECC-Synth given MECH backend

    Args:
        G (nx.graph): Backend

    Returns:
        tuple[np.array, list]: Adjacency graph and qubit index list
    """
    qubit_idx_dict = gen_qubit_idx_dict(G)
    regular_coupling = list([qubit_idx_dict[n1], qubit_idx_dict[n2]] for n1,n2 in G.edges)
    regular_coupling += list([qubit_idx_dict[n2], qubit_idx_dict[n1]] for n1,n2 in G.edges)

    CG = coupling_to_adjacency(regular_coupling)
    qubit_idx_dict = idx_dict_to_list(qubit_idx_dict)

    return (CG, qubit_idx_dict)


def display_simple_backend(backend: nx.Graph, filename: str) -> None:
    """Generate figure of coupling graph for specified backend

    Args:
        backend (nx.Graph): _description_
        filename (str): _description_
    """
    qubit_idx_dict = gen_qubit_idx_dict(backend)
    regular_coupling = list([qubit_idx_dict[n1], qubit_idx_dict[n2]] for n1,n2 in backend.edges)
    regular_coupling += list([qubit_idx_dict[n2], qubit_idx_dict[n1]] for n1,n2 in backend.edges)

    cm = CouplingMap(couplinglist=regular_coupling)
    plot_coupling_map(cm.size(), None, cm.get_edges(), filename=filename)


def write_to_file(object, filename: str) -> None:
    """Write a generate object to file

    Args:
        object (_type_): Object to write
        filename (str): Destination
    """
    filehandler = open(f"{filename}.pkl", "wb") 
    pickle.dump(object, filehandler)


if __name__ == "__main__":
    # Select surface code as this is supported and tested with both MECH and QECC-Synth
    code_name = "surface"

    # Generate square backend of different size and chiplet connectivity
    #nnx = [4, 6, 8]
    #cl = [4]
    #cl = [4, 2, 1]
    nnx = [8]
    cl = [4]

    for n, c in tqdm(itertools.product(nnx, cl), total=len(nnx) * len(cl)):
        logging.info(f"Running experiment for nx = ny = {n} and c = {c}")

        # Generate backend
        simple_dqc_backend, qubit_num, data_qubit_num = generate_simple_dqc_backend(n, n, c)
        # Print backend to file
        display_simple_backend(simple_dqc_backend, f"data/backends/square_{n}_{n}_{c}.png")
        
        # Calculate size of code based on backend size
        d = get_max_d(code_name, data_qubit_num)
        logging.info(f"Max distance for {code_name} is {d}")
        if d < 3:
            logging.error(
                f"Code distance too small! {code_name} with distance {d} and {data_qubit_num} qubits: Execution not possible")
            exit(1)
        cycles = d
        # Generate code
        code = get_code(code_name, d, cycles)
        logging.info("Generated QEC Code")

        # Mapping and routing of the circuit, utilizing different algorithms

        # MECH
        circuit_mech = transpile_circuit_MECH(code.qc, simple_dqc_backend)
        write_to_file(circuit_mech, f"data/transpiled_circuit/{code_name}_{d}_square_{n}_{n}_{c}_mech")

        # QECC-Synth
        # Generate backend for QECC-Synth
        architecture = generate_qecc_synth_backend_from_mech(simple_dqc_backend)
        # Perform algorithm
        circuit_qecc_synth = transpile_circuit_QECCSynth(d, architecture, f"square_{n}_{n}_{c}")
        write_to_file(circuit_qecc_synth, f"data/transpiled_circuit/{code_name}_d_square_{n}_{n}_{c}_qecc_synth")  
        # Found optimium: 6, 6, 4  

        # Qiskit-SABRE
        # Generate backend for qiskit
        cm = generate_qiskit_backend_from_mech(simple_dqc_backend)
        # Perform algorithm
        circuit_qiskit = transpile_circuit_SABRE(circuit = code.qc, coupling_map = cm)
        write_to_file(circuit_qiskit, f"data/transpiled_circuit/{code_name}_d_square_{n}_{n}_{c}_qiskit")
