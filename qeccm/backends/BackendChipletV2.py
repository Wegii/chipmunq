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
     - TODO: Add inter-qpu connections and different constraints to backend
     - TODO: Add list or something to get the qubits that connect to other qpus
     - TODO: Check this link out https://quantum.cloud.ibm.com/docs/en/guides/represent-quantum-computers
     - TODO: Move to GenericBackendV2: https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.providers.fake_provider.GenericBackendV2
     - TODO: Check https://quantum.cloud.ibm.com/docs/en/migration-guides/qiskit-2.0

    TODO: check out implementation for chiplets
    Adapted from https://quantum.cloud.ibm.com/docs/en/guides/custom-backend
    TODO: they also have a fake Kookaburra backend



    Args:
        BackendV2 (_type_): _description_
    """

    def __init__(self, size, n_inter) -> None:
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
        self.typology = "grid"

        # Different variants for connecting and placing chiplets:
        # - line: simple line (Peano Curve for placement)
        # - grid: simple grid structure
        #self.chiplet_topology = "line"
        self.chiplet_topology = "grid"

        # Type of remote gate connecting chiplets
        self.remote_gate_type = "ecr"

        # Type of connectivity (nn: neares-neighbour, torus: connectivity of 6)
        # Note: This can't necessarily be applied on all type of topologies.
        self.connectivity = "nn"
        #self.connectivity = 'torus'

        # TODO: not usable qubits
        # TODO: Create a either a list of qubits that can not be used, or potentially remove the completely from the 
        #       coupling map. For now, this case is not considered

        # Dictionary mapping chiplet index to list of nodes on chiplet 
        self.chiplet_to_nodes = {}

        # Construct target
        # TODO: Better comment why!
        self._target = Target("Fake chiplet backend", num_qubits=self.c1*self.c2 * self.n * self.m)

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

        if self.typology == "grid":
            if self.connectivity == "nn":
                # Generate simple grid graph with edge to nearest neighbour
                G = rustworkx.generators.grid_graph(self.n, self.m, multigraph=False)

                # TODO: Add higher-order connectivity
                
                # Add edge payload
                for edge_index in range(G.num_edges()):
                    G.update_edge_by_index(edge_index, LABEL_ON_CHIP)

            if self.connectivity == "torus":
                # TODO: Check if utilization of self.n and self.m is correct
                G = rx.generators.directed_grid_graph(self.n, self.m)
                for column in range(self.n):
                    G.add_edge(column, (self.m-1) * self.n + column, None)
                for row in range(self.m):
                    G.add_edge(row * self.n, row * self.m + (self.n-1), None)

        elif self.typology == "heavy-hex":
            distance = 3
            G = rx.generators.directed_heavy_hex_graph(distance, bidirectional=False)
        elif self.typology == "line":
            raise NotImplemented

        num_qubits = self.c1*self.c2 * self.n*self.m
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
        for i in range(num_qubits):
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

            # Construct gate constraints. Add local two-qubit gates (CZ)
            for root_edge in G.edge_list():
                offset = i * len(G)
                edge = (root_edge[0] + offset, root_edge[1] + offset)
                cz_props[edge] = InstructionProperties(
                    error=rng.uniform(7e-4, 5e-3),
                    duration=rng.uniform(1e-8, 9e-7),
                )

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

                    # Connect to right
                    if x < y_c - 1:
                        right_idx = idx + self.n*self.m 
                        cb_r, ct_r, cr_r, cl_r = self.get_edge_coordinates(self.n, self.m, right_idx)

                        edge = (cr_idx, cl_r)
                        print(edge)
                        print("right")
                        cx_props[edge] = InstructionProperties(
                            error=rng.uniform(7e-4, 5e-3),
                            duration=rng.uniform(1e-8, 9e-7),
                        )

                    # Connect to bottom
                    if y < x_c - 1:
                        bottom_idx = idx + y_c*self.n*self.m
                        cb_b, ct_b, cr_b, cl_b = self.get_edge_coordinates(self.n, self.m, bottom_idx)

                        edge = (cb_idx, ct_b)
                        print(edge)
                        print("bottom")
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

    def edge_attr_fn(self, edge):
        attr_dict = {
            #"label": edge,
            "color": "black",
            "penwidth": str(3),
        }

        if edge == LABEL_INTER_CHIP:
            attr_dict['color'] = 'red'
        
        return attr_dict
    
    def node_attr_fn(self, node):
        attr_dict = {
            "fontcolor": "white",
            "color": "darkcyan", 
            "fill_color": "darkcyan",
            "style": "filled",
            "shape": "circle",
            #"label": str(node),
            "width": ".5",
            "height": ".5",
            "rank": "same"
        }
        
        return attr_dict

    def visualize_coupling_map(self):
        # TODO: nice visualization slide 11 at: https://docs.google.com/presentation/d/1jfFkAl5iXKAwr9SH0FCukEQ7G2c2gMhMVwMitzbRDyw/edit?resourcekey=0-BUIhaZ_kk5O8OYA7fJUN8A&slide=id.g2b15005381b_0_120#slide=id.g2b15005381b_0_120
        graphviz_draw(self.G, method="neato", edge_attr_fn=self.edge_attr_fn, node_attr_fn=self.node_attr_fn,
                      filename=f"data/backends/chiplet_{self.c1}_{self.c2}_{self.n}_{self.m}.png")

    def mapped_node_attr_fn(self, node):
        attr_dict = {
            "style": "filled",
            "shape": "circle",
            #"label": str(node),
            "width": ".5",
            "height": ".5",
            "rank": "same"
        }

        # Change color of node if mapped
        if str(node) == "u":
            attr_dict["fontcolor"] = "white"
            attr_dict["fill_color"] = "darkcyan"
            attr_dict["color"] = "darkcyan"
        else:
            attr_dict["fontcolor"] = "black"
            attr_dict["fill_color"] = "white"
        
        # TODO: add label showing which qubit is placed on which node on the backend

        return attr_dict

    def visualize_mapping(self, mapping, filename):
        
        # TODO: Improve the visualization
        # - A nice way of visualizing the mapping is shown here: https://quantum.cloud.ibm.com/docs/en/guides/represent-quantum-computers
        # - Potentially use plot_circuit_layout for the mapping visualization

        # Iterate over chiplets
        for chiplet_key in mapping:
            # Iterate over each node in mapping and mark as utilized
            chiplet_mapping = mapping[chiplet_key]
            for node_mapping_kay in chiplet_mapping:
                self.G[chiplet_mapping[node_mapping_kay]] = "u"

        graphviz_draw(self.G, method="neato", edge_attr_fn=self.edge_attr_fn, node_attr_fn=self.mapped_node_attr_fn,
                filename=filename)
        
    def get_edge_coordinates(self, n, m, offset=0) -> tuple:
        cb_idx = (np.floor(m/2)).astype(int)
        ct_idx = ((n - 1) * m + np.floor(m/2)).astype(int)

        # Right edge
        cr_idx = (np.floor(n/2) * m + m-1).astype(int)
        # Left edge
        cl_idx = (np.floor(n/2) * m).astype(int)

        return cb_idx + offset, ct_idx + offset, cr_idx + offset, cl_idx + offset

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