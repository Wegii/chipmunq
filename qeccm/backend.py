import logging

# Graph
import rustworkx as rx
import rustworkx.generators

# Visualizations
from rustworkx.visualization import mpl_draw, graphviz_draw
import matplotlib.pyplot as plt

# Qiskit
from qiskit.providers import BackendV2, Options

# Numerics
import numpy as np

# Coupling maps:
# Create different coupling maps
#   - Monolithical
#   - Chiplet
#
# G(V, E)
# Vertices: V
# Edges: E (what gates can be performed on them?)
#   - Remote edges normally only SWAP
#   - All other normally all (CNOT + single qubit rotation)
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


class GenericMonolythicalBackend(BackendV2):
    """ Simple monolythical backend

    Args:
        BackendV2 (_type_): _description_
    """

    def __init__(self, size):
        super().__init__(name="GenericMonolythical", backend_version=2)
        self.n, self.m  = size

        self.G = None


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

    Args:
        BackendV2 (_type_): _description_
    """

    def __init__(self, size, n_inter) -> None:
        super().__init__(name="GenericChiplet", backend_version=2)
    
        # Number of chiplets, row, column
        self.c, self.n, self.m  = size
        self.G = None
        self.n_inter = n_inter

        try:
            assert n_inter <= self.m
        except AssertionError:
            logging.warning(f"Number of interconnections must be smaller than width. Setting to {self.n}")
            self.n_intern = self.n

    def _single_graph(self) -> rx.PyGraph:

        # For a nice layout have a look at: https://github.com/munich-quantum-toolkit/qecc/blob/ls-compilation/scripts/co3/layouts.py
        # There, the layout has fixed coordinates.

        # Generate simple grid graph with edge to nearest neighbour
        G = rustworkx.generators.grid_graph(self.n, self.m, multigraph=False)

        # TODO: Add higher-order connectivity
        
        # Add edge payload
        for edge_index in range(G.num_edges()):
            G.update_edge_by_index(edge_index, LABEL_ON_CHIP)

        return G

    def build_backend(self) -> None:
        # Construct sub-graphs based on number of chiplets specified
        G_partitioned = None
        for c in range(self.c):
            g = self._single_graph()

            if G_partitioned == None:
                G_partitioned = g
            else:
                G_partitioned = rx.union(G_partitioned, g, merge_nodes=False, merge_edges=False)

        self.G = G_partitioned

        # TODO: Calculate number of rows and columns for structuring the subgraphs


        # Connect edges from center
        cb_idx, ct_idx, cr_idx, cl_idx = get_edge_coordinates(self.n, self.m, 0)
        cb_idx1, ct_idx1, cr_idx1, cl_idx1 = get_edge_coordinates(self.n, self.m, self.n*self.m)
        cb_idx2, ct_idx2, cr_idx2, cl_idx2 = get_edge_coordinates(self.n, self.m, (self.n*self.m)*2)
        cb_idx3, ct_idx3, cr_idx3, cl_idx3 = get_edge_coordinates(self.n, self.m, (self.n*self.m)*3)

        # Connect graphs together
        self.G.add_edges_from([(ct_idx, cb_idx1, LABEL_INTER_CHIP)])
        self.G.add_edges_from([(cr_idx1, cl_idx2, LABEL_INTER_CHIP)])
        self.G.add_edges_from([(cb_idx2, ct_idx3, LABEL_INTER_CHIP)])     
        self.G.add_edges_from([(cl_idx3, cr_idx, LABEL_INTER_CHIP)])        


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

    a = GenericChipletBackend((4, 6, 5), 1)
    a.build_backend()
    a.visualize_coupling_map()