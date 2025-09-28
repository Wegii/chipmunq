import copy
import networkx as nx
import matplotlib.pyplot as plt


class CouplingGraph:
    def __init__(self, coupling_graph=None):
        self.graph = coupling_graph
        self._dist_matrix = None
        self._size = None

    def __copy__(self):
        new_instance = copy.deepcopy(self)
        return new_instance
    
    @property
    def distance_matrix(self):
        if self._dist_matrix is None:
            self.compute_distance_matrix()
        return self._dist_matrix
    
    def compute_distance_matrix(self):
        self._dist_matrix = nx.floyd_warshall_numpy(self.graph, nodelist=[i for i in range(len(self.graph.nodes))])
        
    def draw(self, size=5, with_labels=False):
        f = plt.figure(figsize=(size,size))
        nx.draw(self.graph, pos={node: node for node in self.graph.nodes}, with_labels=with_labels, node_size=100, font_size=6, node_color=['pink' for node in self.graph.nodes])
    
