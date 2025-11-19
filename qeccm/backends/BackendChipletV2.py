# Chiplet Backend based upon BackendV2

# TODO: replace logging with qiskit logger
import logging

# Numerics
import numpy as np

# Graph
import rustworkx as rx
import rustworkx.generators

# Visualizations
from rustworkx.visualization import graphviz_draw
import matplotlib.pyplot as plt

# Qiskit
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.library import XGate, SXGate, RZGate, CZGate, ECRGate
from qiskit.circuit import Measure, Delay, Parameter, Reset

LABEL_ON_CHIP = 'on_chip_connection'
LABEL_INTER_CHIP = 'inter_chip_connection'


class BackendChipletV2(BackendV2):
    """ Simple chiplet backend

    Things to add:
     - Check out how to modify the backend with target that the qiskit compiler knows all potential constraints
     - More transpiler info: https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.Target

    Additions
        - TODO: Add remote gates to give those higher error


    Args:
        BackendV2 (_type_): _description_
    """

    def __init__(self, size, n_inter, connectivity: str = "nn", topology: str = "grid") -> None:
        """Instantiate new multi-chip backend.

        :param size: _description_
        :type size: _type_
        :param n_inter: _description_
        :type n_inter: _type_
        """
        
        super().__init__(name="GenericChiplet")
    
        # Number of chiplets, row, column
        self.c1, self.c2, self.n, self.m  = size
        self.G = None
        self.n_inter = n_inter

        try:
            assert n_inter <= self.m
        except AssertionError:
            logging.warning(f"Number of interconnections must be smaller than width. Setting to {self.n}")
            self.n_intern = self.n

        # TODO: Add linear,
        # TODO: Add heavy hex
        # self.topology = "rotated_grid"
        self.topology = topology

        # Different variants for connecting and placing chiplets:
        # - line: simple line (Peano Curve for placement)
        # - grid: simple grid structure
        # - idea: https://patentimages.storage.googleapis.com/d7/d8/83/51fb5619877a47/US20250181953A1-20250605-D00004.png
        #self.chiplet_topology = "line"
        self.chiplet_topology = "grid"

        # Type of remote gate connecting chiplets
        self.remote_gate_type = "ecr"

        # Type of connectivity (nn: neares-neighbour, torus: connectivity of 6)
        # Note: This can't necessarily be applied on all type of topologies.
        self.connectivity = connectivity

        # TODO: not usable qubits
        # TODO: Create a either a list of qubits that can not be used, or potentially remove the completely from the 
        #       coupling map. For now, this case is not considered

        # Dictionary mapping chiplet index to list of nodes on chiplet 
        self.chiplet_to_nodes = {}

        # Construct target
        # TODO: Better comment why!
        if self.chiplet_topology == "line":
            self.num_qubits_total = self.c1* self.n * self.m
        else:
            self.num_qubits_total = self.c1*self.c2 * self.n * self.m
        self._target = Target("Fake chiplet backend", num_qubits=self.num_qubits_total)

        # Construct local chip and gates (single- and two-qubit gates)
        self.G, self._target = self._generate_chiplet()
        # Connect chiplets together using two-qubit gates
        self._target = self._generate_connected_chiplet()

    def _generate_chiplet(self) -> rx.PyGraph:
        """_summary_

        Heavy-hex:
            - Distance: This number **must** be odd

        :raises NotImplemented: _description_
        :raises NotImplemented: _description_
        :return: _description_
        :rtype: rx.PyGraph
        """

        # For a nice layout have a look at:
        #   https://github.com/munich-quantum-toolkit/qecc/blob/ls-compilation/scripts/co3/layouts.py
        # There, the layout has fixed coordinates.

        if self.topology == "grid":
            if self.connectivity == "nn":
                # Generate simple grid graph with edge to nearest neighbour
                G = rustworkx.generators.grid_graph(self.n, self.m, multigraph=False)
                
                # Add edge payload
                for edge_index in range(G.num_edges()):
                    G.update_edge_by_index(edge_index, LABEL_ON_CHIP)

            if self.connectivity == "torus":
                # Generate torus layout
                G = rustworkx.generators.grid_graph(self.n, self.m, multigraph=False)

                def idx(r, c):
                    return (r % self.n) * self.m + (c % self.m)

                # Vertical modulo wrap
                for c in range(self.m):
                    top_node = idx(0, c)
                    bottom_node = idx(self.n - 1, c)
                    if not G.has_edge(bottom_node, top_node):
                        G.add_edge(bottom_node, top_node, None)

                # Horizontal modulo wrap
                for r in range(self.n):
                    left_node = idx(r, 0)
                    right_node = idx(r, self.m - 1)
                    if not G.has_edge(right_node, left_node):
                        G.add_edge(right_node, left_node, None)

                # Long-range connections "inside" the chip
                for r in range(self.n):
                    for c in range(self.m):
                        node = idx(r, c)

                        # Two remote neighbors (with wrap-around)
                        remote_neighbors = [
                            idx(r + 1, c + 2),  # offset: +1 row, +2 columns
                            idx(r - 1, c - 2),  # offset: -1 row, -2 columns
                        ]

                        for nb in remote_neighbors:
                            if nb != node and not G.has_edge(node, nb):
                                G.add_edge(node, nb, None)
                
        elif self.topology == "rotated_grid":
            if self.connectivity == "nn":
                # Reference: https://blog.google/technology/research/google-willow-quantum-chip/
                # Generate simple grid graph with edge to nearest neighbour

                G = rx.PyGraph(multigraph=False)
                rows = self.n
                cols = self.m

                # Add nodes
                for r in range(rows):
                    for c in range(cols):
                        idx = r * cols + c
                        G.add_node(r*cols + c)

                # Create vertices
                for r in range(rows-1):
                    for c in range(cols):
                        a = r*cols + c
                        b = a+cols + 1

                        if r%2 != 1:
                            b -= 1
                            
                        # Connection to top left node. Not set for left most node in this row
                        if r%2 == 1 or (c >= 1):
                            G.add_edge(a, b-1, None)

                        # Connection to top right node
                        if r%2 != 1 or c < cols-1:
                            G.add_edge(a, b, None)

        elif self.topology == "heavy-hex":
            distance = 3
            G = rx.generators.directed_heavy_hex_graph(distance, bidirectional=False)
        elif self.topology == "line":
            raise NotImplemented

        # Single-qubit gates
        # Generate instruction properties for single qubit gates and a measurement, delay,
        #  and reset operation to every qubit in the backend.
        rng = np.random.default_rng(seed=12345678942)
        rz_props = {}
        x_props = {}
        sx_props = {}
        measure_props = {}
        delay_props = {}
 
        # Add single-qubit gates. Globally use virtual rz, x, sx, and measure
        for i in range(self.num_qubits_total):
            qarg = (i,)
            rz_props[qarg] = InstructionProperties(error=0.0, duration=0.0)
            x_props[qarg] = InstructionProperties(
                error=rng.uniform(1e-6, 1e-4),
                duration=rng.uniform(1e-8, 9e-7),
            )
            sx_props[qarg] = InstructionProperties(
                error=rng.uniform(1e-6, 1e-4),
                duration=rng.uniform(1e-8, 9e-7),
            )
            measure_props[qarg] = InstructionProperties(
                error=rng.uniform(1e-3, 1e-1),
                duration=rng.uniform(1e-8, 9e-7),
            )
            delay_props[qarg] = None
        self._target.add_instruction(XGate(), x_props)
        self._target.add_instruction(SXGate(), sx_props)
        self._target.add_instruction(RZGate(Parameter("theta")), rz_props)
        self._target.add_instruction(Measure(), measure_props)
        self._target.add_instruction(Reset(), measure_props)
 
        self._target.add_instruction(Delay(Parameter("t")), delay_props)

        # Add local two-qubit gates
        cz_props = {}
        for i, c in enumerate(range(self.c1*self.c2)):
            self.chiplet_to_nodes[c] = list(range(i * self.n * self.m, (i+1) * self.n * self.m ))
            #print(c)

            # Construct gate constraints. Add local two-qubit gates (CZ)
            for root_edge in G.edge_list():
                offset = i * len(G)
                edge = (root_edge[0] + offset, root_edge[1] + offset)
                cz_props[edge] = InstructionProperties(
                    error=rng.uniform(7e-4, 5e-3),
                    duration=rng.uniform(1e-8, 9e-7),
                )
            #for edge in G.edge_list():
            #    cz_props[edge] = InstructionProperties(
            #        error=rng.uniform(7e-4, 5e-3),
            #        duration=rng.uniform(1e-8, 9e-7),
            #    )

        self._target.add_instruction(CZGate(), cz_props)

        return G, self._target
    
    def _generate_connected_chiplet(self):
        """_summary_

        :param g: _description_
        :type g: _type_
        """
        rng = np.random.default_rng(seed=12345678942)

        # Add inter-chip two-qubit gates (CX)
        cx_props = {}
        if self.chiplet_topology == "line":
            for i in range(1, self.c1):
                cb_idx, ct_idx, cr_idx, cl_idx = self.get_edge_coordinates(self.n, self.m, (i-1)*self.n*self.m)
                cb_idx1, ct_idx1, cr_idx1, cl_idx1 = self.get_edge_coordinates(self.n, self.m, i*self.n*self.m)

                edge = (
                    cr_idx,
                    cl_idx1,
                )
                cx_props[edge] = InstructionProperties(
                    error=rng.uniform(7e-4, 5e-3),
                    duration=rng.uniform(1e-8, 9e-7),
                )
        elif self.chiplet_topology == "grid":
            #x_c = np.sqrt(int(self.c))
            #y_c = np.sqrt(int(self.c))
            x_c = self.c1
            y_c = self.c2
            
            # Iterate over each row
            for y in range(x_c):
                # Iterate over each column
                for x in range(y_c):
                    idx = (y*y_c + x) * self.n * self.m
                    
                    # Get the edges for the current node
                    cb_idx, ct_idx, cr_idx, cl_idx = self.get_edge_coordinates(self.n, self.m, idx)

                    # Calculate offset_indices for multiple connections between chiplets
                    nu = int(np.ceil((self.n_inter)/2))
                    nl = int(np.floor((self.n_inter)/2))
                    offset_indices = list(range(-nl, nu))

                    # Connect to right
                    if x < y_c - 1:
                        right_idx = idx + self.n*self.m 
                        cb_r, ct_r, cr_r, cl_r = self.get_edge_coordinates(self.n, self.m, right_idx)

                        for oi in offset_indices:
                            edge = (cr_idx + (oi*self.m), cl_r + (oi*self.m))
                            #print(edge)
                            #print("right")
                            cx_props[edge] = InstructionProperties(
                                error=rng.uniform(7e-4, 5e-3),
                                duration=rng.uniform(1e-8, 9e-7),
                            )


                    # Connect to bottom
                    if y < x_c - 1:
                        bottom_idx = idx + y_c*self.n*self.m
                        cb_b, ct_b, cr_b, cl_b = self.get_edge_coordinates(self.n, self.m, bottom_idx)

                        for oi in offset_indices:
                            edge = (cb_idx + oi, ct_b + oi)
                            cx_props[edge] = InstructionProperties(
                                error=rng.uniform(7e-4, 5e-3),
                                duration=rng.uniform(1e-8, 9e-7),
                            )


        if self.remote_gate_type == "ecr":
            self._target.add_instruction(ECRGate(), cx_props)
        else:
            # TODO: add option to have other remote gates
            self._target.add_instruction(ECRGate(), cx_props)

        return self._target

    def get_chiplet_at(self, index: int):
        # Return nodes associated with specified chiplet
        return self.chiplet_to_nodes[index]

        
    def get_edge_coordinates(self, n, m, offset=0) -> tuple:
        cb_idx = (np.floor(m/2)).astype(int)
        ct_idx = ((n - 1) * m + np.floor(m/2)).astype(int)

        # Right edge
        cr_idx = (np.floor(n/2) * m + m-1).astype(int)
        # Left edge
        cl_idx = (np.floor(n/2) * m).astype(int)

        return cb_idx + offset, ct_idx + offset, cr_idx + offset, cl_idx + offset

    def get_num_chips(self) -> int:
        """Return number of chiplets depending on chiplet_topology

        :return: Number of chiplets
        :rtype: int
        """

        if self.chiplet_topology == "line":
            return self.c1
        elif self.chiplet_topology == "grid":
            return self.c1 * self.c2
        else:
            return -1

    def get_chip_size(self) -> int:
        """Return size of the biggest chip.

        Note: At the moment, all chips have the same size.

        :return: Size of a single chip
        :rtype: int
        """
        return self.m * self.n

    @property
    def target(self):
        return self._target
    
    @property
    def max_circuits(self):
        return None
    
    @classmethod
    def _default_options(cls):
        return Options(shots=1024)
    
    def run(self, circuit, **kwargs):
        raise NotImplementedError("This backend does not contain a run method")