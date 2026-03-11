import math
import matplotlib.pyplot as plt
import numpy as np
from qeccm.backends import BackendChipletV2
from pathlib import Path

import qiskit
from qiskit.visualization import plot_gate_map
from qiskit.visualization.exceptions import VisualizationError


def plot_gate_map(backend: BackendChipletV2, filename: str = "", show_bb_node_color: bool = False):
    """Custom implementation of qiskit.visualization.plot_gate_map

    Creates coordinates based on backend type rectangular, etc.

    Creates coloring for chiplet backend remote connections

    Create coloring for non-local connections (gross code style)
    """
    qubit_coordinates = generate_coordinates(backend)
    line_colors, qubit_colors = generate_formatting(backend, qubit_coordinates)

    if backend.connectivity == "nn":
        qiskit.visualization.plot_gate_map(
            backend,
            qubit_coordinates=qubit_coordinates,
            qubit_color=qubit_colors,
            line_color=line_colors,
            filename=filename,
        )
    else:
        # Add similar coloring to nodes as in: https://arxiv.org/pdf/2506.03094
        qubit_shapes = []

        if show_bb_node_color:
            # Prepare color and shape lists
            qubit_colors = []

            for i, (x, y) in enumerate(qubit_coordinates):
                row = abs(int(x))

                if row % 2 == 0:
                    if i % 2 == 0:
                        qubit_colors.append("#2A9374")
                        qubit_shapes.append("s")
                    else:
                        qubit_colors.append("#046494")
                        qubit_shapes.append("o")
                elif i % 2 == 0:
                    qubit_colors.append("#FDD689")
                    qubit_shapes.append("o")
                else:
                    qubit_colors.append("#E18AAA")
                    qubit_shapes.append("s")
        else:
            for i, (x, y) in enumerate(qubit_coordinates):
                qubit_shapes.append("s")

        plt.figure()  # figsize=(8, 8))

        # Draw nodes
        for i, (x, y) in enumerate(qubit_coordinates):
            plt.scatter(x, y, s=50, color=qubit_colors[i], marker=qubit_shapes[i], zorder=3)
            plt.text(x, y, str(i), color="white", ha="center", va="center", fontsize=10, weight="bold", zorder=4)

        # Draw connections
        G = qiskit.transpiler.coupling.CouplingMap(backend.coupling_map).graph
        for edge, color in zip(G.edge_list(), line_colors):
            a, b = edge
            x1, y1 = qubit_coordinates[a]
            x2, y2 = qubit_coordinates[b]

            # Optionally alternate curvature direction to reduce overlap
            import math

            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            curvature = 0 if distance == 1 else -0.15

            plt.annotate(
                "",
                xy=(x2, y2),
                xycoords="data",
                xytext=(x1, y1),
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="-",
                    color=color,
                    lw=1.5,
                    alpha=0.9,
                    connectionstyle=f"arc3,rad={curvature}",
                ),
                zorder=1,
            )

        # Final formatting
        plt.axis("off")
        plt.gca().set_aspect("equal")
        plt.tight_layout()
        
        # Generate file and directory
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename)


def plot_circuit_layout(circuit: qiskit.QuantumCircuit, backend: BackendChipletV2, filename: str = ""):
    """Plot mapping of quantum circuit on backend.

    TODO: also show ancilla qubit usage, since this is currently ignored

    Modified version of qiskit.visualization.plot_circuit_layout.
    """
    qubit_coordinates = generate_coordinates(backend)
    line_colors, qubit_colors = generate_formatting(backend, qubit_coordinates)

    view = ""
    view = "virtual"

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
    
    # Generate file and directory
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    qiskit.visualization.plot_gate_map(
        backend,
        qubit_color=qcolors,
        qubit_labels=qubit_labels,
        # line_color=lcolors,
        line_color=line_colors,
        qubit_coordinates=qubit_coordinates,
        filename=filename,
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
        if len(qargs) == 2:  # two-qubit gate
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
    qubit_colors = [
        "#BFBFBF" if c == full_counts.min() else to_hex(cmap(norm(c)), keep_alpha=False) for c in full_counts
    ]

    # Generate coordinates and line color (chiplet connections)
    qubit_coordinates = generate_coordinates(backend)
    line_colors, _ = generate_formatting(backend, qubit_coordinates)

    # Generate file and directory
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    qiskit.visualization.plot_gate_map(
        backend,
        qubit_coordinates=qubit_coordinates,
        qubit_color=qubit_colors,
        line_color=line_colors,
        filename=filename,
    )


def generate_coordinates(backend):
    # Generate coordinates for nodes

    if backend.topology == "grid":
        x_range = range(-backend.n // 2, backend.n // 2)
        y_range = range(-backend.m // 2, backend.m // 2)
        # print(len(x_range))
        # print(len(y_range))

        coordinates = [(x, y) for x in x_range for y in y_range]
    elif backend.topology == "rotated_grid":
        x_range = range(-backend.n // 2, backend.n // 2)
        y_range = range(-backend.m // 2, backend.m // 2)
        x_range = [x / 2 for x in x_range]
        # print(len(x_range))
        # print(len(y_range))

        coordinates = []
        for c_id, x in enumerate(x_range):
            for row_idx, y in enumerate(y_range):
                y_shifted = y + 0.5 if (c_id % 2 == 1) else y
                coordinates.append((x, y_shifted))

        # coordinates = [(x, y) for x in x_range for y in y_range]
        # (ax, ay) = coordinates[1]
        # coordinates[1] = (ax, ay+0.1)
    else:
        return None

    # print(coordinates)
    # print(len(coordinates))
    total_qubit_coordinates = []
    if backend.c1 > 1 or backend.c2 > 1:
        if backend.chiplet_topology == "line":
            for coordinate in coordinates:
                total_qubit_coordinates.append(coordinate)

            for i in range(1, backend.c1):
                # Concatenate chiplets

                for coordinate in coordinates:
                    total_qubit_coordinates.append((coordinate[0], coordinate[1] + i * backend.m))
        elif backend.chiplet_topology == "grid":
            # for coordinate in coordinates:
            #    total_qubit_coordinates.append(coordinate)

            x_c = backend.c1
            y_c = backend.c2

            # Iterate over each row
            for y in range(x_c):
                # Iterate over each column
                for x in range(y_c):
                    # idx = (y*y_c + x) * self.n * self.m

                    for coordinate in coordinates:
                        total_qubit_coordinates.append(
                            (coordinate[0] - y * (backend.n + 1) / 2, coordinate[1] + x * (backend.m + 1))
                        )
                        # pass
            # total_qubit_coordinates = []

    else:
        total_qubit_coordinates.extend(coordinates)

    # print(total_qubit_coordinates)
    # print(len(total_qubit_coordinates))
    return total_qubit_coordinates


def generate_formatting(backend: BackendChipletV2, qubit_coordinates: list):
    target = backend.target
    coupling_map_backend = target.build_coupling_map()

    # Access the defective map (representing the functional hardware)
    defective_qubit_coupling_map = backend.defective_coupling_map

    # Create sets for faster lookup of functional components
    # We assume the defective map contains ONLY the working edges/nodes
    active_edges = set(defective_qubit_coupling_map.get_edges())
    # active_nodes = set(defective_qubit_coupling_map.physical_qubits)
    active_nodes = set({qubit for edge in active_edges for qubit in edge})

    # 1. Base Line Coloring: Select color depending on distance of connections.
    # Direct connections = Blue (#6D8196), Remote (>1) = Violet (#9400D3)
    line_colors = [
        "#6D8196"
        if math.isclose(
            math.sqrt(
                (qubit_coordinates[a][0] - qubit_coordinates[b][0]) ** 2
                + (qubit_coordinates[a][1] - qubit_coordinates[b][1]) ** 2
            ),
            0,
        )
        else "#6D8196"
        for a, b in coupling_map_backend.get_edges()
    ]

    ecr_edges = []

    # Get tuples for the edges which have an ecr instruction attached
    for instruction in target.instructions:
        if instruction[0].name == backend.remote_gate_type:
            ecr_edges.append(instruction[1])

    # 2. Iterate to apply specific overrides (ECR and Defective)
    for i, edge in enumerate(coupling_map_backend.get_edges()):
        # Apply ECR color (Salmon/Orange)
        if edge in ecr_edges:
            line_colors[i] = "#9400D3"

        # OVERRIDE: If edge is missing from defective map, color it Red
        if edge not in active_edges:
            line_colors[i] = "#FF0000"

    # Node coloring
    qubit_colors = ["#007878" if qubit in active_nodes else "#FF746C" for qubit in coupling_map_backend.physical_qubits]

    return line_colors, qubit_colors
