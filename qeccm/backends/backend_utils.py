from matplotlib.ticker import NullFormatter
from qiskit.visualization import plot_gate_map

from qecc_mapping.qeccm.backends import BackendChipletV2


def plot_gate_map(backend: BackendChipletV2):
    """ Custom implementation of qiskit.visualization.plot_gate_map
    
    Creates coordinates based on backend type rectangular, etc.

    Creates coloring for chiplet backend remote connections

    Create coloring for non-local connections (gross code style)
    """

    qubit_coordinates = generate_coordinates(backend)

    line_colors = generate_formatting(backend)

    plot_gate_map(
        backend,
        plot_directed=False,
        qubit_coordinates=qubit_coordinates,
        line_color=line_colors,
        filename = "data/backends/new_chiplet.png",
    )


def generate_coordinates(backend):
    # Generate coordinates for nodes

    if backend.type == "rectangle":
        return None
    else:
        return None


def generate_formatting(backend):
    # Perform formatting: line colors, node colors, etc.
    pass