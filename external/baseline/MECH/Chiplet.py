import networkx as nx
import matplotlib.pyplot as plt
from numpy import *
from CouplingGraph import *

def pick_n_elements(lst, n):
    m = len(lst) // n
    r = len(lst) % n
    idx = [floor(r/2) + ceil((m-1)/2) + i for i in range(len(lst)) if i % m == 0][:n]
    return [lst[int(i)] for i in idx]

def pick_n_links(size, lst, n):
    if n > len(lst):
        print('AT MOST {} LINKS'.format(len(lst)))
        return lst
    links_1 = pick_n_elements(lst, n)
    mid_link_1 = links_1[int(floor(len(links_1) / 2))]
    new_lst = list(lst[::-1])
    links_2 = pick_n_elements(new_lst, n)
    mid_link_2 = links_2[int(floor(len(links_2) / 2))]
    mid_link = (size+1) / 2 - 1
    if abs(mid_link_2 - mid_link) < abs(mid_link_1 - mid_link):
        return links_2
    else:
        return links_1

def gen_square_lattice(x_dim, y_dim):
    return nx.grid_2d_graph(x_dim, y_dim)

def gen_hexagon_lattice(x_dim, y_dim):
    G = nx.grid_2d_graph(x_dim, y_dim)
    for node in G.nodes:
        if node[1] < y_dim-1 and (node[0] + node[1])%2 == 1:
            G.remove_edge(node, (node[0], node[1]+1))
    return G

def gen_heavy_square_lattice(x_dim, y_dim):
    G = nx.grid_2d_graph(x_dim, y_dim)
    nodes_to_remove = []
    for node in G.nodes:
        if node[0]%2==1 and node[1]%2==1:
           nodes_to_remove.append(node)
    G.remove_nodes_from(nodes_to_remove)
    return G 

def gen_heavy_hexagon_lattice(x_dim, y_dim):
    G=gen_heavy_square_lattice(x_dim, y_dim)
    nodes_to_remove = []
    for x,y in G.nodes:
        if y<y_dim-1 and x%2==0 and y%2==0 and (x+y)%4==2:
            nodes_to_remove.append((x,y+1))
    G.remove_nodes_from(nodes_to_remove)
    return G 

def gen_node_ids(lattice, chiplet_x_size, chiplet_y_size):
    for x, y in lattice.nodes:
        lattice.nodes[(x,y)]['id'] = (x//chiplet_x_size, y//chiplet_y_size, x%chiplet_x_size, y%chiplet_y_size)
        
def gen_link_types(lattice):
    for u,v in lattice.edges:
        if lattice.nodes[u]['id'][:2] == lattice.nodes[v]['id'][:2]:
            lattice.edges[(u,v)]['type'] = 'on_chip'
        else:
            lattice.edges[(u,v)]['type'] = 'cross_chip'

def gen_sparse_chip_links(lattice, rows, cols):
    edges_to_remove = []
    for u,v in lattice.edges:
        u_id, v_id = lattice.nodes[u]['id'], lattice.nodes[v]['id']
        if lattice.edges[(u,v)]['type'] == 'cross_chip':
            if u_id[2] == v_id[2] and u_id[2] not in cols:
                edges_to_remove.append((u,v))
            elif u_id[3] == v_id[3] and u_id[3] not in rows:
                edges_to_remove.append((u,v))
    lattice.remove_edges_from(edges_to_remove)

def gen_chiplet_array(geometry, array_x_dim, array_y_dim, chiplet_x_size, chiplet_y_size, cross_link_sparsity=None, cross_link_row_sparsity=None, cross_link_col_sparsity=None, auto=True):

    if geometry == 'square':
        G=gen_square_lattice(array_x_dim * chiplet_x_size, array_y_dim * chiplet_y_size)
        G.cross_link_rows = list(range(chiplet_y_size))
        G.cross_link_cols = list(range(chiplet_x_size))
    if geometry == 'hexagon':
        if auto:
            if chiplet_x_size % 2 == 1:
                print('WARNING: hexagon chiplets should have even columns')
                chiplet_x_size -= 1
        G=gen_hexagon_lattice(array_x_dim * chiplet_x_size, array_y_dim * chiplet_y_size)
        G.cross_link_rows = list(range(chiplet_y_size))
        G.cross_link_cols = list(range((chiplet_y_size+1)%2,chiplet_x_size,2))
    if geometry == 'heavy_square':
        if auto:
            if chiplet_x_size % 2 == 1:
                print('WARNING: heavy square chiplets should have even columns')
                chiplet_x_size -= 1
            if chiplet_y_size % 2 == 1:
                print('WARNING: heavy square chiplets should have even rows')
                chiplet_y_size -= 1
        G=gen_heavy_square_lattice(array_x_dim * chiplet_x_size, array_y_dim * chiplet_y_size)
        G.cross_link_rows = list(range(0,chiplet_y_size,2))
        G.cross_link_cols = list(range(0,chiplet_x_size,2))
    if geometry == 'heavy_hexagon':
        if auto:
            if chiplet_x_size % 4 != 0:
                print('WARNING: heavy hexagon chiplets should have 4n columns')
                chiplet_x_size = chiplet_x_size // 4 * 4
            if chiplet_y_size % 4 != 0:
                print('WARNING: heavy hexagon chiplets should have 4n rows')
                chiplet_y_size = chiplet_y_size // 4 * 4
        G=gen_heavy_hexagon_lattice(array_x_dim * chiplet_x_size, array_y_dim * chiplet_y_size)
        G.cross_link_rows = list(range(0,chiplet_y_size,2))
        G.cross_link_cols = list(range(2,chiplet_x_size,4))
    G.structure = geometry
    G.array_x_dim = array_x_dim
    G.array_y_dim = array_y_dim
    G.chiplet_x_size = chiplet_x_size
    G.chiplet_y_size = chiplet_y_size
    G.highway_qubits = None
    G.highway_distance_dict = None
    G.local_coupling_graph = None
    G.highway_coupling_graph = None
    for node in G.nodes:
        G.nodes[node]['type'] = 'data'
        
    gen_node_ids(G, chiplet_x_size, chiplet_y_size)
    gen_link_types(G)
    if cross_link_row_sparsity is None and cross_link_col_sparsity is None and cross_link_sparsity is not None:
        cross_link_row_sparsity = cross_link_col_sparsity = cross_link_sparsity
    if cross_link_row_sparsity:
        G.cross_link_rows = pick_n_links(G.chiplet_y_size, sorted(G.cross_link_rows), cross_link_row_sparsity)
    if cross_link_col_sparsity:
        G.cross_link_cols = pick_n_links(G.chiplet_x_size, sorted(G.cross_link_cols), cross_link_col_sparsity)
    gen_sparse_chip_links(G, G.cross_link_rows, G.cross_link_cols)
    return G

def get_node_type(G, node):
    return G.nodes[node].get('type')

def get_highway_qubits(G):
    if G.highway_qubits is None:
        G.highway_qubits = set([node for node in G.nodes if get_node_type(G, node) == 'highway'])
    return G.highway_qubits

def potential_highway_nodes_within_radius(G, node, radius):
    neighborhood = nx.ego_graph(G, node, radius=radius, center=False)
    filtered_nodes = [node for node in neighborhood.nodes if G.nodes[node].get('type') in {'highway', 'undetermined'}]
    return filtered_nodes

def get_local_coupling_graph(G):
    if G.local_coupling_graph is None:
        coupling_graph = G.copy()
        nodes_to_remove = []
        for node in G.nodes:
            if G.nodes[node]['type'] == 'highway':
                nodes_to_remove.append(node)
        coupling_graph.remove_nodes_from(nodes_to_remove)
        for node in coupling_graph.nodes:
            coupling_graph.nodes[node]['pos'] = node
        G.local_coupling_graph = CouplingGraph(coupling_graph)
    return G.local_coupling_graph

def get_highway_coupling_graph(G, refresh=False):
    if refresh or G.highway_coupling_graph is None:
        coupling_graph = nx.Graph()
        highway_qubits = set()
        for node in G.nodes:
            if get_node_type(G, node) in {'highway', 'undetermined'}:
                coupling_graph.add_node(node, pos=node)
                highway_qubits.add(node)
        
        for node in highway_qubits:
            neighbors_of_higway_neighbors = set()
            highway_neighbors_in_1_step = potential_highway_nodes_within_radius(G, node, radius=1)
            for nei in highway_neighbors_in_1_step:
                coupling_graph.add_edge(node, nei, weight=1)
                neighbors_of_higway_neighbors = neighbors_of_higway_neighbors.union(set(G.neighbors(nei)))
            highway_neighbors_in_2_step = potential_highway_nodes_within_radius(G, node, radius=2)
            filtered_highway_neighbors = set(highway_neighbors_in_2_step) - set(highway_neighbors_in_1_step) - neighbors_of_higway_neighbors
            for nei in filtered_highway_neighbors:
                coupling_graph.add_edge(node, nei, weight=2)
        G.highway_coupling_graph = CouplingGraph(coupling_graph)
    return G.highway_coupling_graph

def get_unsatisfying_highway_coupling_nodes(graph):
        unsatisfying_nodes = set()
        for node in graph.nodes:
            if graph.degree(node) >= 3:
                interleaved_neighbors = [nei for nei in graph.neighbors(node) if graph.edges[(node, nei)]['weight'] > 1]
                if len(interleaved_neighbors) > 2:
                    unsatisfying_nodes.add(node)
        return unsatisfying_nodes

def gen_efficient_highway_coupling_graph(G):
    graph = get_highway_coupling_graph(G, refresh=True).graph
    multi_degree_nodes = set([node for node in graph.nodes if graph.degree(node) >= 3])
    residual_subgraph = nx.subgraph(graph, set(graph.nodes) - multi_degree_nodes).copy()
    unsatisfying_subgraph = nx.Graph()
    for node in multi_degree_nodes:
        for nei in graph.neighbors(node):
            unsatisfying_subgraph.add_edge(node, nei)

    for component in nx.connected_components(unsatisfying_subgraph):
        left_most_node, right_most_node = min(component, key=lambda pos: pos[0]), max(component, key=lambda pos: pos[0])
        bottom_most_node, up_most_node = min(component, key=lambda pos: pos[1]), max(component, key=lambda pos: pos[1])
        horizontal_paths = nx.all_shortest_paths(graph, left_most_node, right_most_node, weight='weight')
        vertical_paths = nx.all_shortest_paths(graph, bottom_most_node, up_most_node, weight='weight')

        success = False
        for h_path in horizontal_paths:
            for v_path in vertical_paths:
                crossroad_subgraph = nx.Graph()
                for idx in range(1, len(h_path)):
                    crossroad_subgraph.add_edge(h_path[idx-1], h_path[idx], weight=graph.edges[(h_path[idx-1], h_path[idx])]['weight'])
                for idx in range(1, len(v_path)):
                    crossroad_subgraph.add_edge(v_path[idx-1], v_path[idx], weight=graph.edges[(v_path[idx-1], v_path[idx])]['weight'])
                for n1 in [left_most_node, right_most_node]:
                    for n2 in [bottom_most_node, up_most_node]:
                        if not nx.has_path(crossroad_subgraph, n1, n2):
                            subgraph = nx.subgraph(graph, crossroad_subgraph.nodes)
                            path = nx.shortest_path(subgraph, n1, n2)
                            for idx in range(1, len(path)):
                                crossroad_subgraph.add_edge(path[idx-1], path[idx], weight=graph.edges[(path[idx-1], path[idx])]['weight'])
                if not get_unsatisfying_highway_coupling_nodes(crossroad_subgraph):
                    residual_subgraph.add_edges_from(crossroad_subgraph.edges)
                    success = True
                    break
            if success:
                break
        if not success:
            return None
    for edge in residual_subgraph.edges:
        residual_subgraph.edges[edge]['weight'] = graph.edges[edge]['weight']
    return CouplingGraph(residual_subgraph)

def gen_interleaving_path_between(G, source, target, adhoc_dense=True, offset=None):
    if offset == 'left' or offset == 'down':
        coor_offset = -1
    elif offset == 'right' or offset == 'up':
        coor_offset = 1
    else:
        coor_offset = 0
    all_shortest_paths = list(nx.all_shortest_paths(G, source, target))
    shortest_path = min(all_shortest_paths, key=lambda path: sum(abs(x-(source[0] + 0.5*coor_offset)) for x,_ in path))
    left_pointer, right_pointer = 0, len(shortest_path) - 1

    while right_pointer - left_pointer >= 2:
        left_node, right_node = shortest_path[left_pointer], shortest_path[right_pointer]
        if get_node_type(G, left_node) == 'data':
            G.nodes[left_node]['type'] = 'highway'
        if get_node_type(G, right_node) == 'data':
            G.nodes[right_node]['type'] = 'highway'
        
        if right_pointer - 2 < left_pointer + 2:
            break
        else:
            left_pointer += 2
            right_pointer -= 2

    left_node, right_node = shortest_path[left_pointer], shortest_path[right_pointer]
    if get_node_type(G, left_node) == 'data':
        G.nodes[left_node]['type'] = 'highway'
    if get_node_type(G, right_node) == 'data':
        G.nodes[right_node]['type'] = 'highway'

    if right_pointer - left_pointer != 1:
        G.nodes[shortest_path[left_pointer+1]]['type'] = 'undetermined'
        G.nodes[shortest_path[right_pointer-1]]['type'] = 'undetermined'

def gen_road_along(G, row=None, col=None, offset=None, adhoc_dense=True):
    max_x, max_y = G.array_x_dim * G.chiplet_x_size, G.array_y_dim * G.chiplet_y_size
    if not adhoc_dense:
        if row:
            gen_interleaving_path_between(G, (0, row), (max_x-1, row))
        if col:
            gen_interleaving_path_between(G, (col, 0), (col, max_y-1), offset=offset)
    else:
        if row:
            left_most_id, right_most_id = G.nodes[(0, row)]['id'], G.nodes[(max_x-1, row)]['id']
            for array_x in range(left_most_id[0], right_most_id[0]+1):
                
                gen_interleaving_path_between(G, (array_x * G.chiplet_x_size, row), ((array_x+1) * G.chiplet_x_size -1, row))
        if col:
            bottom_most_id, up_most_id = G.nodes[(col, 0)]['id'], G.nodes[(col, max_y-1)]['id']
            for array_y in range(bottom_most_id[1], up_most_id[1]+1):
                gen_interleaving_path_between(G, (col, array_y * G.chiplet_y_size), (col, (array_y+1) * G.chiplet_y_size -1), offset=offset)

def deal_with_undetermined_nodes(G, adhoc_dense=True):
    for node in G.nodes:
        if G.nodes[node]['type'] == 'undetermined':
            neighborhood = potential_highway_nodes_within_radius(G, node, 2)
            for nei in neighborhood:
                if not adhoc_dense:
                    x, y = nei
                    max_x, max_y = G.array_x_dim * G.chiplet_x_size, G.array_y_dim * G.chiplet_y_size
                else:
                    x, y = G.nodes[nei]['id'][-2:]
                    max_x, max_y = G.chiplet_x_size, G.chiplet_y_size
                is_on_edge = x==0 or x==max_x-1 or y==0 or y==max_y-1

                if G.nodes[nei]['type'] == 'highway':
                    least_neighbors = 2 - is_on_edge + 1
                if G.nodes[nei]['type'] == 'undetermined':
                    least_neighbors = 4 - is_on_edge
                if len(potential_highway_nodes_within_radius(G, nei, 2)) < least_neighbors:
                        G.nodes[node]['type'] = 'highway'
                        break
                else:
                    neighborhood = potential_highway_nodes_within_radius(G, nei, 2)
                    neighborhood.remove(node)
                    this_row = [n for n in neighborhood if n[1] == nei[1]]
                    if len(this_row) == 1:
                        G.nodes[node]['type'] = 'highway'
                        break
            if G.nodes[node]['type'] == 'undetermined': 
                G.nodes[node]['type'] = 'data' 
                if gen_efficient_highway_coupling_graph(G)is None:
                    G.nodes[node]['type'] = 'highway'

def gen_highway_layout(G):
    highway_row = min(G.cross_link_rows, key=lambda x: abs(x - G.chiplet_y_size/2))
    highway_col = min(G.cross_link_cols, key=lambda x: abs(x - G.chiplet_x_size/2))
    if highway_col <  G.chiplet_x_size/2:
        offset = 'right'
    else:
        offset = 'left'
    for y in range(G.array_y_dim):
        row = y * G.chiplet_y_size + highway_row
        gen_road_along(G, row=row)
        
    for x in range(G.array_x_dim):
        col = x * G.chiplet_x_size + highway_col
        gen_road_along(G, col=col, offset=offset)
        
    deal_with_undetermined_nodes(G)
    efficient_highway_coupling_graph = gen_efficient_highway_coupling_graph(G)
    if efficient_highway_coupling_graph is not None:
        G.highway_coupling_graph = efficient_highway_coupling_graph

def custom_highway_layout(G, local_positions_on_single_chiplet, local_highway_coupling_edges=[]):
    for i in range(G.array_x_dim):
        for j in range(G.array_y_dim):
            for x, y in local_positions_on_single_chiplet:
                node = (G.chiplet_x_size * i + x, G.chiplet_y_size * j + y)
                G.nodes[node]['type'] = 'highway'
    
    if local_highway_coupling_edges:                
        highway_coupling_G = nx.Graph()
        for i in range(G.array_x_dim):
            for j in range(G.array_y_dim):
                for n1, n2 in local_highway_coupling_edges:
                    node1 = (G.chiplet_x_size * i + n1[0], G.chiplet_y_size * j + n1[1])
                    node2 = (G.chiplet_x_size * i + n2[0], G.chiplet_y_size * j + n2[1])
                    highway_coupling_G.add_edge(node1, node2)

        for node in get_highway_qubits(G):
            for nei in list(G.neighbors(node)):
                if nei in G.highway_qubits:
                    highway_coupling_G.add_edge(node, nei)
        G.highway_coupling_graph = CouplingGraph(highway_coupling_G)


def gen_idx_qubit_dict(chiplet):
    idx_qubit_dict = dict()
    data_idx, highway_idx = 0, len(chiplet.nodes) - len(get_highway_qubits(chiplet))
    sorted_nodes = sorted(chiplet.nodes, key=lambda node: 1.1 * chiplet.nodes[node]['id'][0] + chiplet.nodes[node]['id'][1])
    for node in sorted_nodes:
        if get_node_type(chiplet, node)  == 'data':
            idx_qubit_dict[data_idx] = node
            data_idx += 1
        else:
            idx_qubit_dict[highway_idx] = node
            highway_idx += 1
    return idx_qubit_dict

def gen_qubit_idx_dict(chiplet):
    qubit_idx_dict = dict()
    data_idx, highway_idx = 0, len(chiplet.nodes) - len(get_highway_qubits(chiplet))
    sorted_nodes = sorted(chiplet.nodes, key=lambda node: 1.1 * chiplet.nodes[node]['id'][0] + chiplet.nodes[node]['id'][1])
    for node in sorted_nodes:
        if get_node_type(chiplet, node)  == 'data':
            qubit_idx_dict[node] = data_idx
            data_idx += 1
        else:
            qubit_idx_dict[node] = highway_idx
            highway_idx += 1
    return qubit_idx_dict


def draw_lattice(graph, size=8, node_size=300, with_labels=True, border=True, data_color='#99CCFF', highway_color='#004C99', on_chip_width=1, cross_link_width=8, fig_name=None, save=False):
        node_lable_dict = gen_qubit_idx_dict(graph)
        
        node_colors = []
        for node in graph.nodes:
            node_type = graph.nodes[node].get('type', '')
            if node_type == 'highway':
                node_colors.append(highway_color)
            elif node_type == 'undetermined':
                node_colors.append('yellowgreen')
            else:
                node_colors.append(data_color)

        edge_color = ['red' if graph.edges[edge].get('type', None) == 'cross_chip' else 'black' for edge in graph.edges]
        edge_width = [cross_link_width if graph.edges[edge].get('type', None) == 'cross_chip' else on_chip_width for edge in graph.edges]

        if border:
            edgecolors = ['black' for node in graph.nodes]
        else:
            edgecolors = data_color
        f = plt.figure(figsize=(size,size))
        ids = {node: graph.nodes[node].get('id', node)[-2:] for node in graph.nodes}
        nx.draw(graph, pos={(i,j): (i,j) for i, j in graph.nodes()}, with_labels=with_labels, labels=node_lable_dict, font_size=8, node_size=node_size, node_color=node_colors, edgecolors=edgecolors, linewidths=0.5, edge_color=edge_color, width=edge_width)
        if save:
            if fig_name is None:
                fig_name = '{}x{}_{}x{}_'.format(graph.array_x_dim, graph.array_y_dim, graph.chiplet_x_size, graph.chiplet_y_size) + graph.structure + '_' + data_color
            print('./figures/'+ fig_name + '.pdf')
            plt.savefig('./figures/'+ fig_name + '.pdf')


def are_regularly_connected(G, node_1, node_2):
    if get_node_type(G, node_1) == 'highway' or get_node_type(G, node_2) == 'highway':
        return False
    else:
        return G.has_edge(node_1, node_2)

def is_qubit_next_to_highway(G, node):
    if get_node_type(G, node) == 'highway':
        return False
    for nei in list(G.neighbors(node)):
        if G.nodes[nei].get('type') == 'highway':
            return True
    return False

def qubits_next_to_highway(G):
    result = set()
    highway_qubtis = get_highway_qubits(G)
    for highway_qubit in highway_qubtis:
        for nei in list(G.neighbors(highway_qubit)):
            if G.nodes[nei].get('type') == 'data':
                result.add(nei)
    return result


def get_distance_to_highway(G, node):
    if G.highway_distance_dict is None: 
        G.highway_distance_dict = {}
        for source_node in G.nodes():
            shortest_paths = nx.single_source_dijkstra_path_length(G, source_node)
            G.highway_distance_dict[source_node] = min(shortest_paths[highway_qubit] for highway_qubit in get_highway_qubits(G))

    return G.highway_distance_dict[node]


def find_possible_entrances_with_paths(G, node, range_beyond_closest=2):
        max_steps=int(get_distance_to_highway(G, node)) + range_beyond_closest
        entrance_with_paths = []
        nearby_entrance_cadidates = set(potential_highway_nodes_within_radius(G, node, radius=max_steps))
        for entrance in nearby_entrance_cadidates:
            all_shortest_paths = nx.all_shortest_paths(G, node, entrance)
            for path in all_shortest_paths:
                if len(set(path).intersection(get_highway_qubits(G))) == 1:
                    entrance_with_paths.append((entrance, tuple(path[:-1])))
        return entrance_with_paths
