import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))

from qeccm.backends.BackendChipletV2 import BackendChipletV2
from qeccm.backends.backend_utils import plot_gate_map


def test_chiplet_backend():

    chiplet_backend = BackendChipletV2((4, 6, 6), 1)
    plot_gate_map(backend = chiplet_backend, filename = "data/backends/chiplet_testing.png")


if __name__ == "__main__":

    test_chiplet_backend()

    # TODO: Add multiple possible configurations