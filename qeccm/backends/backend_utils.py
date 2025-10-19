import qiskit
from qiskit.visualization import plot_gate_map
from qeccm.backends import BackendChipletV2


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


def plot_circuit_layout(transpiled_circuit: qiskit.QuantumCircuit, backend: BackendChipletV2, filename: str = ""):
    
    qubit_coordinates = generate_coordinates(backend)

    # TODO: color of nodes should now depend on the degree of connectivity of each node (from the circuit)
    line_colors, qubit_colors = generate_formatting(backend)
    
    qiskit.visualization.plot_circuit_layout(
        transpiled_circuit,
        backend,
        qubit_coordinates=qubit_coordinates,
        qubit_color = qubit_colors,
        line_color = line_colors,
        filename = filename
    )


def generate_coordinates(backend):
    # Generate coordinates for nodes

    if backend.typology == "grid":
        x_range = range(-backend.n//2, backend.n//2)
        y_range = range(-backend.m//2, backend.m//2)
        print(len(x_range))
        print(len(y_range))
        
        coordinates = [(x, y) for x in x_range for y in y_range]
    else:
        return None
    
    print(coordinates)
    print(len(coordinates))
    total_qubit_coordinates = []
    if backend.c > 1:
        if backend.chiplet_typology == 'line':
            for coordinate in coordinates:
                total_qubit_coordinates.append(coordinate)
        
            for i in range(1, backend.c):
                # Concatenate chiplets
                
                for coordinate in coordinates:
                    total_qubit_coordinates.append(
                        (coordinate[0], coordinate[1] + i*backend.m)
                    )
        

    else:
        total_qubit_coordinates.append(coordinate)

    print(total_qubit_coordinates)
    print(len(total_qubit_coordinates))
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
