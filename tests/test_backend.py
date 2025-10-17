import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))

from qeccm.backends.BackendChipletV2 import BackendChipletV2

#
from qiskit.visualization import plot_gate_map


def test_chiplet_backend():

    chiplet_backend = BackendChipletV2((2, 5, 5), 1)

    plot_gate_map(
        chiplet_backend,
        plot_directed=False,
        filename = "data/backends/new_chiplet.png"
    )


if __name__ == "__main__":

    test_chiplet_backend()

    # TODO: Add multiple possible configurations