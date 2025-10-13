import logging

# Graph
import rustworkx as rx
import rustworkx.generators

# Visualizations
from rustworkx.visualization import mpl_draw, graphviz_draw
import matplotlib.pyplot as plt

# Qiskit
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import CouplingMap
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.library import XGate, SXGate, RZGate, CZGate, ECRGate
from qiskit.circuit import Measure, Delay, Parameter, Reset
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_gate_map

# Numerics
import numpy as np

# Coupling maps:
# Create different coupling maps
#   - Monolithical
#   - Chiplet
#       - heterogeneous with different chips of different connectivities
#
# G(V, E)
# Vertices: V
#   - as property also error; readout error; one and two-qubit error
#   - if the qubit is measured, weight the readout error more
# Edges: E (what gates can be performed on them?)
#   - Remote edges normally only SWAP
#       - IBM Flamingo technical report [21], reports CNOT gates on coupler links with a 3.5% error rate
#       - Also possible to perform cnot gate -> test mapping with and without CNOT (and then only SWAP)
#   - All other normally all (CNOT + single qubit rotation)
#   
# Two types of connections
# Intermodule connections


# Visualization part:
# Normal visualization for nearest-neighbour connectivity
# For connectivity of 6 (e.g. toric), show these in another color overlayed
# (this way the graph can still be displayed on a plane)

LABEL_ON_CHIP = 'on_chip_connection'
LABEL_INTER_CHIP = 'inter_chip_connection'


def get_edge_coordinates(n, m, offset=0) -> tuple:
    cb_idx = (np.floor(m/2)).astype(int)
    ct_idx = ((n - 1) * m + np.floor(m/2)).astype(int)

    # Right edge
    cr_idx = (np.floor(n/2) * m + m-1).astype(int)
    # Left edge
    cl_idx = (np.floor(n/2) * m).astype(int)

    return cb_idx + offset, ct_idx + offset, cr_idx + offset, cl_idx + offset

# TODO: add class or function for constructing different connectivity graphs (grid, ring, heavyhex) 

class GenericMonolythicalBackend(BackendV2):
    """ Simple monolythical backend

    Args:
        BackendV2 (_type_): _description_
    """

    def __init__(self, size):
        
        super().__init__(name="GenericMonolythical", backend_version=2)
        self.n, self.m  = size

        self.G = None
        # TODO:  Add a)Linear, (b) Ring, (c) Grid, and (d) Star.
        self.typology = 'grid'


    def build_backend(self):
        self.G = rustworkx.generators.grid_graph(self.n, self.m, multigraph=False)

        for node in self.G.node_indices():
            self.G[node] = node
        
    def edge_attr_fn(self, node):
        attr_dict = {
            "color": "black",
            "penwidth": str(3),
        }
        
        return attr_dict
    
    def node_attr_fn(self, node):
        attr_dict = {
            "fontcolor": "white",
            "color": "darkcyan", 
            "fill_color": "darkcyan",
            "style": "filled",
            "shape": "circle",
            "label": str(node),
            "width": "1",
            "height": "1",
            "rank": "same"
        }
        
        return attr_dict
    
    def visualize_coupling_map(self):

        graphviz_draw(self.G, node_attr_fn=self.node_attr_fn, edge_attr_fn=self.edge_attr_fn, method="neato", 
                      filename=f"data/backends/monolythical_{self.n}_{self.m}_test.png")
        

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


class GenericChipletBackend(BackendV2):
    """ Simple chiplet backend

    Things to add:
     - Check out how to modify the backend with target that the qiskit compiler knows all potential constraints
     - More transpiler info: https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.Target

    Additions
     - TODO: Add inter-qpu connections and different constraints to backend
     - TODO: Add list or something to get the qubits that connect to other qpus
     - TODO: Check this link out https://quantum.cloud.ibm.com/docs/en/guides/represent-quantum-computers

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
        self.c, self.n, self.m  = size
        self.G = None
        self.n_inter = n_inter

        try:
            assert n_inter <= self.m
        except AssertionError:
            logging.warning(f"Number of interconnections must be smaller than width. Setting to {self.n}")
            self.n_intern = self.n

        # TODO:  Add a) Linear, (b) Ring, (c) Grid, and (d) Star.
        self.typology = 'grid'

        # Dictionary mapping chiplet index to list of nodes on chiplet 
        self.chiplet_to_nodes = {}

        num_qubits = self.c * self.n*self.m
        self._target = Target(
            "Fake multi-chip backend", num_qubits=num_qubits
        )

        # RNG for gate errors
        rng = np.random.default_rng(seed=42)



        # Construct local chip and gates
        g = self._single_graph()
        G_partitioned = None
        cz_props = {}
        for i, c in enumerate(range(self.c)):
            # Construct backend as graph
            if G_partitioned == None:
                G_partitioned = g
            else:
                G_partitioned = rx.union(G_partitioned, g, merge_nodes=False, merge_edges=False)

            self.chiplet_to_nodes[c] = list(range(i * self.n * self.m, (i+1) * self.n * self.m ))

            # Construct gate constraints. Add local two-qubit gates (CZ)
            for root_edge in g.edge_list():
                offset = i * len(g)
                edge = (root_edge[0] + offset, root_edge[1] + offset)
                cz_props[edge] = InstructionProperties(
                    error=rng.uniform(7e-4, 5e-3),
                    duration=rng.uniform(1e-8, 9e-7),
                )

        self.G = G_partitioned
        self._target.add_instruction(CZGate(), cz_props)




        # TODO: Calculate number of rows and columns for structuring the subgraphs


        # Construct inter-chip gates


        if self.c == 4:
            cb_idx, ct_idx, cr_idx, cl_idx = get_edge_coordinates(self.n, self.m, 0)
            cb_idx1, ct_idx1, cr_idx1, cl_idx1 = get_edge_coordinates(self.n, self.m, self.n*self.m)
            cb_idx2, ct_idx2, cr_idx2, cl_idx2 = get_edge_coordinates(self.n, self.m, (self.n*self.m)*2)
            cb_idx3, ct_idx3, cr_idx3, cl_idx3 = get_edge_coordinates(self.n, self.m, (self.n*self.m)*3)

            # Connect graphs together
            self.G.add_edges_from([(ct_idx, cb_idx1, LABEL_INTER_CHIP)])
            self.G.add_edges_from([(cr_idx1, cl_idx2, LABEL_INTER_CHIP)])
            self.G.add_edges_from([(cb_idx2, ct_idx3, LABEL_INTER_CHIP)])     
            self.G.add_edges_from([(cl_idx3, cr_idx, LABEL_INTER_CHIP)])     
        else:
            # Construct backend as graph
            cb_idx, ct_idx, cr_idx, cl_idx = get_edge_coordinates(self.n, self.m, 0)
            cb_idx1, ct_idx1, cr_idx1, cl_idx1 = get_edge_coordinates(self.n, self.m, self.n*self.m)

            # Connect graphs together
            self.G.add_edges_from([(ct_idx, cb_idx1, LABEL_INTER_CHIP)])

            # Construct gate constraints. Add inter-chip two-qubit gates (CX)
            cx_props = {}
            edge = (
                ct_idx,
                cb_idx1,
            )
            cx_props[edge] = InstructionProperties(
                error=rng.uniform(7e-4, 5e-3),
                duration=rng.uniform(1e-8, 9e-7),
            )
            self._target.add_instruction(ECRGate(), cx_props)


    def _single_graph(self) -> rx.PyGraph:

        # For a nice layout have a look at:
        #   https://github.com/munich-quantum-toolkit/qecc/blob/ls-compilation/scripts/co3/layouts.py
        # There, the layout has fixed coordinates.

        # Generate simple grid graph with edge to nearest neighbour
        G = rustworkx.generators.grid_graph(self.n, self.m, multigraph=False)

        # TODO: Add higher-order connectivity
        
        # Add edge payload
        for edge_index in range(G.num_edges()):
            G.update_edge_by_index(edge_index, LABEL_ON_CHIP)

        return G

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
        graphviz_draw(self.G, method="neato", edge_attr_fn=self.edge_attr_fn, node_attr_fn=self.node_attr_fn,
                      filename=f"data/backends/chiplet_{self.c}_{self.n}_{self.m}.png")

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


if __name__== "__main__":
    # a = GenericMonolythicalBackend((10, 10))
    # a.build_backend()
    # a.visualize_coupling_map()

    a = GenericChipletBackend((4, 10, 10), 1)
    a.build_backend()
    a.visualize_coupling_map()