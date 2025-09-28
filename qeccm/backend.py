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