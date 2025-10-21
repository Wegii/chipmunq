import qiskit
from qiskit.visualization import plot_gate_map
from qiskit.visualization.exceptions import VisualizationError

from qeccm.backends import BackendChipletV2
import numpy as np


def plot_gate_map(backend: BackendChipletV2, filename: str = ""):
    """ Custom implementation of qiskit.visualization.plot_gate_map
    
    Creates coordinates based on backend type rectangular, etc.

    Creates coloring for chiplet backend remote connections

    Create coloring for non-local connections (gross code style)
    """

    qubit_coordinates = generate_coordinates(backend)
    line_colors, qubit_colors = generate_formatting(backend)

    qiskit.visualization.plot_gate_map(
        backend,
        qubit_coordinates = qubit_coordinates,
        qubit_color = qubit_colors,
        line_color = line_colors,
        filename = filename,
    )


def plot_circuit_layout(circuit: qiskit.QuantumCircuit, backend: BackendChipletV2, filename: str = ""):
    """ Plot mapping of quantum circuit on backend.

    TODO: also show ancilla qubit usage, since this is currently ignored

    Modified version of qiskit.visualization.plot_circuit_layout.
    """
    
    qubit_coordinates = generate_coordinates(backend)

    line_colors, qubit_colors = generate_formatting(backend)
    

    view = ""
    view="virtual"

    num_qubits = backend.num_qubits
    cmap = backend.coupling_map
    cmap_len = cmap.graph.num_edges()

    qubits = []
    qubit_labels = [""] * num_qubits

    bit_locations = {
        bit: {"register": register, "index": index}
        for register in circuit._layout.initial_layout.get_registers()
        for index, bit in enumerate(register)
    }
    for index, qubit in enumerate(circuit._layout.initial_layout.get_virtual_bits()):
        if qubit not in bit_locations:
            bit_locations[qubit] = {"register": None, "index": index}

    if view == "virtual":
        for key, val in circuit._layout.initial_layout.get_virtual_bits().items():
            bit_register = bit_locations[key]["register"]
            if bit_register is None or bit_register.name != "ancilla":
                qubits.append(val)
                qubit_labels[val] = str(bit_locations[key]["index"])

    elif view == "physical":
        for key, val in circuit._layout.initial_layout.get_physical_bits().items():
            bit_register = bit_locations[val]["register"]
            if bit_register is None or bit_register.name != "ancilla":
                qubits.append(key)
                qubit_labels[key] = str(key)

    else:
        raise VisualizationError("Layout view must be 'virtual' or 'physical'.")

    qcolors = ["#648fff"] * num_qubits
    for k in qubits:
        qcolors[k] = "black"

    lcolors = ["#648fff"] * cmap_len

    for idx, edge in enumerate(cmap):
        if edge[0] in qubits and edge[1] in qubits:
            lcolors[idx] = "black"

    qiskit.visualization.plot_gate_map(
        backend,
        qubit_color=qcolors,
        qubit_labels=qubit_labels,
        line_color=lcolors,
        qubit_coordinates=qubit_coordinates,
        filename = filename,
    )

def plot_circuit_layout_utilization(circuit: qiskit.QuantumCircuit, backend: BackendChipletV2, filename: str = ""):
    """Plot utilization of qubit

    :param circuit: _description_
    :type circuit: qiskit.QuantumCircuit
    :param backend: _description_
    :type backend: BackendChipletV2
    :param filename: _description_, defaults to ""
    :type filename: str, optional
    """
    from collections import Counter

    counts = Counter()
    layout = circuit.layout
    num_qubits = backend.num_qubits

    # Iterate over all instructions
    for inst, qargs, _ in circuit.data:
        if len(qargs) == 2:   # two-qubit gate
            for q in qargs:
                # Get the *index* of the qubit in the circuit
                qindex = circuit.find_bit(q).index
                counts[qindex] += 1

    # Fill missing qubits
    full_counts = np.array([counts.get(i, 0) for i in range(num_qubits)])
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    # Generate color based on maximum and minimum of qubit usage
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=full_counts.min(), vmax=full_counts.max() or 1)
    qubit_colors = ["#BFBFBF" if c == full_counts.min() else to_hex(cmap(norm(c)), keep_alpha=False) for c in full_counts]

    # Generate coordinates and line color (chiplet connections)
    qubit_coordinates = generate_coordinates(backend)
    line_colors, _ = generate_formatting(backend)
    
    qiskit.visualization.plot_gate_map(
        backend,
        qubit_coordinates = qubit_coordinates,
        qubit_color = qubit_colors,
        line_color = line_colors,
        filename = filename,
    )

def generate_coordinates(backend):
    # Generate coordinates for nodes

    if backend.typology == "grid":
        x_range = range(-backend.n//2, backend.n//2)
        y_range = range(-backend.m//2, backend.m//2)
        #print(len(x_range))
        #print(len(y_range))
        
        coordinates = [(x, y) for x in x_range for y in y_range]
    else:
        return None
    
    #print(coordinates)
    #print(len(coordinates))
    total_qubit_coordinates = []
    if backend.c1 > 1 or backend.c2 > 1:
        if backend.chiplet_topology == 'line':
            for coordinate in coordinates:
                total_qubit_coordinates.append(coordinate)
        
            for i in range(1, backend.c1):
                # Concatenate chiplets
                
                for coordinate in coordinates:
                    total_qubit_coordinates.append(
                        (coordinate[0], coordinate[1] + i*backend.m)
                    )
        elif backend.chiplet_topology == 'grid':
            #for coordinate in coordinates:
            #    total_qubit_coordinates.append(coordinate)
        
            x_c = backend.c1
            y_c = backend.c2
            
            # Iterate over each row
            for y in range(x_c):
                # Iterate over each column
                for x in range(y_c):
                    #idx = (y*y_c + x) * self.n * self.m

                    for coordinate in coordinates:
                        total_qubit_coordinates.append(
                            (coordinate[0]-y*backend.n, coordinate[1]+x*backend.m)
                            
                        )
                        #pass
            #total_qubit_coordinates = []
        

    else:
        total_qubit_coordinates.append(coordinate)

    #print(total_qubit_coordinates)
    #print(len(total_qubit_coordinates))
    return total_qubit_coordinates


def generate_formatting(backend: BackendChipletV2):
    target = backend.target
    coupling_map_backend = target.build_coupling_map()

    line_colors = ["#6D8196" for edge in coupling_map_backend.get_edges()]
    ecr_edges = []
    
    # Get tuples for the edges which have an ecr instruction attached
    for instruction in target.instructions:
        if instruction[0].name == backend.remote_gate_type:
            ecr_edges.append(instruction[1])
    
    for i, edge in enumerate(coupling_map_backend.get_edges()):
        if edge in ecr_edges:
            line_colors[i] = "#FF746C"


    qubit_colors = ["#007878" for qubit in coupling_map_backend.physical_qubits]


    return line_colors, qubit_colors
