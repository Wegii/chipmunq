# Wrapper around BackendV2


# Graph
import rustworkx.generators

# Qiskit
from qiskit.providers import BackendV2, Options

# Visualizations
from rustworkx.visualization import graphviz_draw

# Numerics

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


# TODO: add class or function for constructing different connectivity graphs (grid, ring, heavyhex)


class BackendMonolythicalV2(BackendV2):
    """Simple monolythical backend

    Args:
        BackendV2 (_type_): _description_
    """

    def __init__(self, size):

        super().__init__(name="GenericMonolythical", backend_version=2)
        self.n, self.m = size

        self.G = None
        # TODO:  Add a)Linear, (b) Ring, (c) Grid, and (d) Star.
        self.typology = "grid"

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
            "rank": "same",
        }

        return attr_dict

    def visualize_coupling_map(self):

        graphviz_draw(
            self.G,
            node_attr_fn=self.node_attr_fn,
            edge_attr_fn=self.edge_attr_fn,
            method="neato",
            filename=f"data/backends/monolythical_{self.n}_{self.m}_test.png",
        )

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
