import networkx as nx
import matplotlib.pyplot as plt

# Example graph
G = nx.karate_club_graph()

# Example partitions (as lists of nodes)
partition1 = [0, 1, 2, 3, 4, 5]
partition2 = [6, 7, 8, 9, 10, 11, 12]
partition3 = [13, 14, 15, 16, 17, 18, 19, 20]

#partitions = [partition1, partition2, partition3]
g1_nodes, g2_nodes = nx.algorithms.community.kernighan_lin_bisection(G, max_iter=10)
partitions = [g1_nodes, g2_nodes]
colors = ['red', 'green']  # color per partition

# Create a mapping: node -> color
node_colors = {}
for part, color in zip(partitions, colors):
    for node in part:
        node_colors[node] = color

# Get color list for drawing
color_list = [node_colors.get(node, 'gray') for node in G.nodes()]

# Layout for visualization
pos = nx.spring_layout(G, seed=42)  # positions for all nodes

# Draw nodes with partition colors
nx.draw_networkx_nodes(G, pos, node_color=color_list, node_size=500)
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=10)

plt.axis('off')
#plt.show()
plt.savefig("tests/test.png")