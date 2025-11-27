import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))

from qeccm.backends.BackendChipletV2 import BackendChipletV2
from qeccm.backends.backend_utils import plot_gate_map


def test_chiplet_backend():

    chiplet_backend = BackendChipletV2((1, 1, 4, 4), 3)
    plot_gate_map(backend = chiplet_backend, filename = "data/backends/grid/r_single_chiplet_basic.png")

    chiplet_backend = BackendChipletV2((2, 2, 4, 4), 2)
    plot_gate_map(backend = chiplet_backend, filename = "data/backends/grid/r_multi_chiplet_basic.png")

    chiplet_backend = BackendChipletV2((1, 1, 4, 4), 3, "torus")
    plot_gate_map(backend = chiplet_backend, filename = "data/backends/torus/t_single_chiplet_basic.png")
    plot_gate_map(backend = chiplet_backend,
                  filename = "data/backends/torus/t_single_chiplet_colored.png",
                  show_bb_node_color = True)

    chiplet_backend = BackendChipletV2((2, 2, 4, 4), 2, "torus")
    plot_gate_map(backend = chiplet_backend,
                  filename = "data/backends/torus/t_multi_chiplet_colored.png",
                  show_bb_node_color = True)

    chiplet_backend = BackendChipletV2((2, 2, 5, 5), 2, "nn", "rotated_grid")
    plot_gate_map(backend = chiplet_backend,
                  filename = "data/backends/grid/t_multi_chiplet_rotated.png",
                  show_bb_node_color = True)
    
    chiplet_backend = BackendChipletV2((2, 2, 5, 5), 2, "nn", "rotated_grid", num_defective_qubits=3)
    plot_gate_map(backend = chiplet_backend,
                  filename = "data/backends/grid/t_multi_chiplet_rotated_defective.png",
                  show_bb_node_color = True)


if __name__ == "__main__":

    test_chiplet_backend()

    # TODO: Add multiple possible configurations