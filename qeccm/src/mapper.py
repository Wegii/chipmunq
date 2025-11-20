# Typing
import kahypar

# Numerics
import random
import math
import numpy as np

# Qiskit transpiler
from qiskit.providers import BackendV2
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.layout import Layout
from qiskit.circuit import Qubit, QuantumRegister

# Hypergraph stuff
import networkx as nx
from qeccm.circuit.hypergraph_circuit import PartitionedHyperGraph
from collections import deque
from itertools import product

# Visualization
import matplotlib.pyplot as plt


class GenericMapper(AnalysisPass):

    def __init__(self):
        """ GenericMapper initializer """ 
        super().__init__()

        # Backend for mapping
        #self.backend = None
        # Dictionary with mapping of nodes
        #self.mapping = {}

    def perform_mapping(self):
        raise NotImplementedError
    
    def visualize_mapping(self, filename) -> None:
        """Visualize mapping on backend"""

        #self.backend.visualize_mapping(self.mapping, filename)   
        # TODO: this needs to be replaced by plot_circuit_layout from backend_utils     
        #self.backend.visualize_mapping(self.property_set["block_node_map"], filename)
    

class RandomMapper(GenericMapper):
    """Random mapping to the backend.

    Idea:
    - Randomly map necessary qubits to the backend (currently one-to-one mapping)
    - Since most algorithms are heuristics, this can be used as some sort of baseline for comparison
    """

    def __init__(self, backend):
        """ RandomMapper initializer """        

        super().__init__()

        # Coupling map to map the dag to
        self.coupling_map = backend.coupling_map
        self.backend = backend

    def run(self, dag: DAGCircuit) -> None:

        # Construct dictionary with block as key and value as (random) mapping from node to backend node 
        full_node_map = {}

        # KaHyPar partitioning as HyperNetX
        partitioned_hgc = self.property_set["partitioned_hyper_dag"]
        # Mapping from partition to QPU
        partition_to_qpu = self.property_set["partition_to_qpu"]

        # Get blocks from partitioned hypergraph
        for block in partitioned_hgc._btn.items():
            # Extract index from key
            block_idx = int(block[0][2:])

            # Get qubit nodes from chiplet to which this partition/block is mapped to
            nodes_in_backend = self.backend.get_chiplet_at(partition_to_qpu[block_idx])
            #print(nodes_in_backend)

            # Map all nodes randomly to chiplet
            for n, node in enumerate(block[1]):
                # Pick a random element
                picked = random.choice(nodes_in_backend)
                # Remove the picked element from the list
                nodes_in_backend.remove(picked)

                full_node_map[node] = picked

        layout = Layout()
        regs = dag.qubits + list(dag.qregs.values())

        hgc = self.property_set['hyper_dag']
        for reg in regs:
            if isinstance(reg, QuantumRegister):
                layout.add_register(reg)
            else:
                # Map qubit id to graph id (since the partitioning works on the graph ids)
                qubit_to_node = reg._index#hgc.node_idx.get(reg._index)

                # Add qubit mapping from partitioning
                if qubit_to_node is not None:
                    # Virtual qubit is used and mapped. Get the mapping from the mapping list 
                    p_b = full_node_map[qubit_to_node]
                    
                    #print(f"mapping qubit {reg._index} as {qubit_to_node} to {p_b}")
                    layout.add(reg, p_b)

        # Virtual to physical qubit mapping
        self.property_set["layout"] = layout


class TrivialMapper(GenericMapper):
    """Map to chiplet backend

    TODO: Description
    """

    def __init__(self, backend):
        """ TrivialMapper initializer """        

        super().__init__()

        # Coupling map to map the dag to
        self.coupling_map = backend.coupling_map
        self.backend = backend

    def run(self,
            dag: DAGCircuit
            ) -> None:
        """_summary_

        :param dag: _description_
        :type dag: DAGCircuit
        """

        # 0. Get partitioning from partition pass
        partitioned_hgc = self.property_set["partitioned_hyper_dag"]

        # 1. Map partitions to QPUs
        partition_to_qpu, utilized_qpus = self.assign_partition_to_qpu(partitioned_hgc)

        # 2. Map each partition onto the assigned QPU
        vq_to_pq_mapping = self.map_partition_on_qpu(partitioned_hgc, partition_to_qpu, utilized_qpus)

        # 3. Create layout from mapping
        layout = self.generate_layout(dag, vq_to_pq_mapping)
        self.property_set["layout"] = layout

    def generate_layout(self,
                        dag: DAGCircuit,
                        vq_to_pq: dict
                        ) -> Layout:
        """_summary_

        :param dag: _description_
        :type dag: DAGCircuit
        :param vq_to_pq: _description_
        :type vq_to_pq: list
        :return: _description_
        :rtype: Layout
        """

        print("Generating Layout")


        """
        print("Manualmapping")

        vq_to_pq = {
            0: 0,
            1: 1,
            2: 2,
            3: 25,
            4: 26,

            5: 5,
            6: 6,
            7: 7,
            8: 30,
            9: 31,

            10: 10,
            11: 11,
            12: 12,
            13: 35,
            14: 36,

            15: 15,
            16: 16,
            17: 17,
            18: 40,
            19: 41,

            20: 20,
            21: 21,
            22: 22,
            23: 45,
            24: 46,
        }
        vq_to_pq = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,

            5: 5,
            6: 6,
            7: 7,
            8: 8,
            9: 9,

            10: 10,
            11: 11,
            12: 12,
            13: 13,
            14: 14,

            15: 15,
            16: 16,
            17: 17,
            18: 18,
            19: 19,

            20: 20,
            21: 21,
            22: 22,
            23: 23,
            24: 25,
        }
        layout = Layout()

        # Add all qubit registers first (optional but consistent with Qiskit behavior)
        for qreg in dag.qregs.values():
            layout.add_register(qreg)

        # Now add virtual → physical mapping using global qubit indices
        for i, vq in enumerate(dag.qubits):
            if i in vq_to_pq:
                layout.add(vq, vq_to_pq[i])
        """

        
        layout = Layout()
        regs = dag.qubits + list(dag.qregs.values())

        # Iterate over all available qubits add mapping for utilized virtual qubits only
        for reg in regs:
            if isinstance(reg, QuantumRegister):
                layout.add_register(reg)
            else:
                if reg._index in vq_to_pq:
                    # Get physical qubit
                    physical_qubit = vq_to_pq[reg._index]
                    # Map mapping from virtual qubit to physical qubit
                    layout.add(reg, physical_qubit)
        

        # Virtual to physical qubit mapping
        self.property_set["layout"] = layout

        return layout

    def map_partition_on_qpu(self,
                             partitioned_hg: PartitionedHyperGraph,
                             partition_to_qpu: dict,
                             utilized_qpus: dict
                             ) -> dict:
        # Each partition is now assigned an QPU. Now map all partitions to their assigned qpu

        print("Partition to QPU")
        
        print("Node placement:")
        for n, b in partition_to_qpu.items():
            print(f"{n} -> {b}")
        
        print("\nBlock contents:")
        for b, nodes in utilized_qpus.items():
            if nodes.placed_partitions:
                print(f"{b}: {nodes.placed_partitions}")
        
        partition_size = {}
        partitions = {}
        for partition_key, nodes in partitioned_hg._btn.items():
            partition_size[partition_key] = int(math.ceil(math.sqrt(len(nodes))))
            partitions[int(partition_key[2:])] = nodes
            
        def dimension_to_linear_index(x, w, h):
            x1, x2 = x  # unpack coordinates
            if not (0 <= x1 < w) or not (0 <= x2 < h):
                raise ValueError(f"Coordinates {x} are out of bounds for grid {w}x{h}")
            
            # Row-major order: row * width + column
            idx = (x1+1) * w + x2
            return idx
            
        # Dictionary with virtual_qubit to physical_qubit mapping
        placement = {}

        # Iterate over all QPUs
        for qpu, qpu_partitions in utilized_qpus.items():

            if qpu_partitions.placed_partitions:
                # Physical qubits for this QPU
                nodes_on_qpu = self.backend.get_chiplet_at(dimension_to_linear_index(qpu, self.backend.c1,
                                                                                     self.backend.c2))

                # Iterate over all partitions that are placed on this QPU
                for p in qpu_partitions.placed_partitions:
                    partition_id, local_x, local_y, patch_width, patch_height = p
                    # Virtual qubits for this partition
                    nodes_of_partition = partitions[partition_id]
                    print(f"Trying to place nodes {nodes_of_partition}")
                    #print(nodes_of_partition)
                    # Calculate offset given the local coordinates
                    offset = local_x * patch_width + local_y

                    rotated = True
                    if not rotated:                    
                        # Iterate over all virtual qubits of this partition and assign physical qubits to it in a linear
                        # fashion
                        # Iterator index for the virtual qubits
                        index = 0
                        # Iterator index for the physical qubits
                        placement_idx = offset

                        for node in nodes_of_partition:
                            #print(placement_idx)
                            #print(nodes_on_qpu[placement_idx])
                            placement[node] = nodes_on_qpu[placement_idx]
                            #print(placement_idx)
                            index += 1
                            placement_idx += 1
                            if index % patch_width == 0:
                                # Jump to the next row. We have to jump chiplet_width - patch_width
                                placement_idx += (self.backend.m - patch_width)

                    else:
                        # TODO: specify distance of code
                        distance = 5#3
                        
                        placement_idx = offset + self.backend.n

                        # Placement patterns of the qubits of a rotated surface code patch. The patch is mapped to
                        # the physical qubits from the left top to the bottom right
                        print("ATTENTION: UTILIZING PLACEMENT FOR DISTANCE 3 PATCHES")
                        d3_column_pattern = [
                            [3, 1, -1],
                            [4, 2, 0],
                            [3, 1, -1],
                            [2, 0, -2],
                            [3, 1, -1]
                        ]
                        d5_column_pattern = [
                            [7, 5, 3, 1, -1],
                            [8, 6, 4, 2, 0],
                            [7, 5, 3, 1, -1],
                            [6, 4, 2, 0, -2],
                            [7, 5, 3, 1, -1],
                            [8, 6, 4, 2, 0],
                            [7, 5, 3, 1, -1],
                            [6, 4, 2, 0, -2],
                            [7, 5, 3, 1, -1],
                        ]
                        
                        if distance == 3:
                            start_row = local_x + 2
                            col = local_y
                            # TODO: Adjust for higher code distance: d=5 -> 2 nodes to place, d=7 -> 3 nodes to place
                            # Placement of first few nodes
                            placement[nodes_of_partition[0]] = nodes_on_qpu[start_row * self.backend.m + col]
                            node_index = 1
                        elif distance == 5:
                            start_row = local_x + 2
                            col = local_y
                            # TODO: Adjust for higher code distance: d=5 -> 2 nodes to place, d=7 -> 3 nodes to place
                            # Placement of first few nodes
                            placement[nodes_of_partition[0]] = nodes_on_qpu[(start_row+4) * self.backend.m + col]
                            placement[nodes_of_partition[1]] = nodes_on_qpu[start_row * self.backend.m + col]
                            node_index = 2

                        # place the rest in groups of 3
                        pattern_index = 0

                        if distance == 3: column_patterns = d3_column_pattern
                        elif distance == 5: column_patterns = d5_column_pattern

                        while node_index < len(nodes_of_partition):
                            offsets = column_patterns[pattern_index]

                            for off in offsets:
                                if node_index >= len(nodes_of_partition):
                                    break

                                row = start_row + off

                                placement[nodes_of_partition[node_index]] = nodes_on_qpu[row * self.backend.n + col]
                                node_index += 1

                            if pattern_index%2 == 0:
                                col += 1
                            print(col)
                            pattern_index = (pattern_index + 1) % len(column_patterns)

                        # Placement of last few nodes
                        # TODO: Adjust for higher code distance: d=5 -> 2 nodes to place, d=7 -> 3 nodes to place
                        if distance == 3:
                            placement[nodes_of_partition[-1]] = nodes_on_qpu[(start_row + 2) * self.backend.m + 3]
                        elif distance == 5:
                            placement[nodes_of_partition[-2]] = nodes_on_qpu[(start_row + 6) * self.backend.m + 5]
                            placement[nodes_of_partition[-1]] = nodes_on_qpu[(start_row + 2) * self.backend.m + 5]

                        

                        
        return placement

    def assign_partition_to_qpu(self,
                                partitioned_hg: PartitionedHyperGraph
                                ) -> dict:
        """Assign each partition to a QPU, based on the interactions with the other partitions.
        
        Big Issue: However, partitioning assumes full qubit connectivity inside and across the quantum processors
                    to reduce the problem to a graph partitioning problem. on a higher level, this constrained is
                    already known to a high level compiler (e. g. for lattice surgery). Thus, we should generally
                    not get a circuit that has to communicate with another node, to which no direct connection is.
                    Note: This is not entirely true, since the ancilla qubits used in lattice surgery could become
                          an issue, if it is not possible to map these also to the same node!
        
        Calculate mapping of partition to QPU. This is necessary, since partitioners assume an all-to-all chiplet 
        topology. Depending on the backend chiplet_topology, we do not have and all-to-all connection.

        :param kahypar_hg: _description_
        :type kahypar_hg: kahypar.Hypergraph
        :return: _description_
        :rtype: List[int]
        """

        # Calculate size of patches (simply number of nodes in this partition)
        partition_size = {}
        for partition_key, nodes in partitioned_hg._btn.items():
            partition_size[partition_key] = len(nodes) #math.sqrt(len(nodes))

        # Assign every node, depending on it's dependency, to a 2D grid of QPUs
        # TODO: This should also work for more complicated QPU layouts (other than 2D)
        placement, blocks = self.bfs_capacitated_grid_placement(
            G = partitioned_hg._collapsed_phg,
            width = self.backend.c1,
            height = self.backend.c2,
            partition_size = partition_size
        )

        # Plot the assignment of partitions to QPU
        self.plot_block_counts(width = self.backend.c1,
                               height = self.backend.c2,
                               block_assignments = blocks,
                               filename="tests/data/figures/partition_to_qpu.png")

        return placement, blocks

    def bfs_capacitated_grid_placement(self,
                                       G: nx.Graph,
                                       width: int,
                                       height: int,
                                       partition_size: dict
                                       ) -> tuple[dict, dict]: 
        """Assignment of partition to qpu

        Every partition receives 2d coordinates of the qpu to map to.

        Steps:
            - 1. Construct order using BFS
            - 2. Place partitions on 2D grid of QPUs given the order from the first step 

        :param G: _description_
        :type G: nx.Graph
        :param width: _description_
        :type width: int
        :param height: _description_
        :type height: int
        :param partition_size: _description_
        :type partition_size: dict
        :return: _description_
        :rtype: dict
        """
        
        # Construct BFS order of partitions. This order is then used in the assignment of partition to qpu 
        nodes_iter = list(G.nodes())
        visited = set()
        bfs_order = []

        start_list = nodes_iter

        # Iterate over all 
        for seed in start_list:
            if seed in visited:
                continue
            
            # Construct queue containing the first partition. If the partition has neighbours (edge to other nodes), 
            # add these to the queue.
            queue = deque([seed])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue

                visited.add(node)
                bfs_order.append(node)

                # Enqueue neighbours 
                for nbr in G.neighbors(node):
                    if nbr not in visited:
                        queue.append(nbr)

        # Initialize all QPUs with their widht and height, as well as coordinates. The widht and height are used for
        # calculating which partitions (given their width and height) can be placed on this QPU.
        qpu_blocks = {(x, y): QPUBlock(self.backend.n, self.backend.m, (x, y))
              for x, y in product(range(width), range(height))}
        
        # Start with the first block
        block_coords_list = list(qpu_blocks.keys())
        current_block_idx = 0
        placement = {}
    
        # Iterate over all partitions, given the order, and place them greedily on the current block. If the QPU is
        # full, find the next QPU to fill.
        for node in bfs_order:
            # TODO: calculate width and height of this node.
            print("ATTENTION: PATCH WIDTH AND HEIGHT ARE CURRENTLY SET MANUALLY!!!")
            # For unrotated surface code d=3
            pw = 5
            ph = 5
            # For rotate surface code d=3
            pw = 3
            ph = 7
            start_idx = current_block_idx

            # Iteratively try to find a QPU to place this partition
            while True:
                block_coord = block_coords_list[current_block_idx]
                block = qpu_blocks[block_coord]
                # Try to place the partition
                pos = block.place_partition(node, pw, ph)
                if pos is not None:
                    # Partition fits
                    placement[node] = pos
                    break

                # Partition does not fit! Try with the next QPU. Currently this simply taks the next QPU in the Qpu list
                current_block_idx = (current_block_idx + 1) % len(block_coords_list)

                # All QPUs are full!
                if current_block_idx == start_idx: raise RuntimeError(f"No space to place partition for node {node}")
        
        return placement, qpu_blocks

    def plot_block_counts(self,
                          width: int,
                          height: int,
                          block_assignments: dict,
                          filename: str
                          ) -> None:
        """Plot 2D grid of QPUs with the number of assigned partitions shown

        :param width: _description_
        :type width: int
        :param height: _description_
        :type height: int
        :param block_assignments: _description_
        :type block_assignments: dict
        :param filename: _description_
        :type filename: str
        """

        fig, ax = plt.subplots(figsize=(width, height))

        # Loop through each grid block and get number of partitions per QPU
        for y in range(height):
            for x in range(width):
                count = len(block_assignments.get((x, y), []).placed_partitions)
                ax.text(x + 0.5, height - y - 0.5, str(count),
                        ha='center', va='center', fontsize=12)

        # Draw grid lines
        ax.set_xticks(np.arange(0, width + 1, 1))
        ax.set_yticks(np.arange(0, height + 1, 1))
        ax.grid(True)
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Save figure
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
    

class QPUBlock:
    """Class representing a QPU chiplet  

    Has list of all partitions on this QPU. The list additionally also has the location (x, y) where this partition can
    be placed on this QPU
    """

    def __init__(self, width: int, height: int, block_coord: tuple):
        """_summary_

        :param width: _description_
        :type width: int
        :param height: _description_
        :type height: int
        :param block_coord: (x, y) position of the block in the grid
        :type block_coord: tuple
        """
        self.width = width
        self.height = height
        self.coord = block_coord  

        # free rectangles inside block
        self.free_rects = [(0, 0, width, height)]  
        # list of (partition_id, x, y, w, h)
        self.placed_partitions = []  

    def can_place_partition(self, pw, ph):
        for fx, fy, fw, fh in self.free_rects:
            if pw <= fw and ph <= fh:
                return True
        return False

    def place_partition(self, partition_id, pw, ph):
        for i, (fx, fy, fw, fh) in enumerate(self.free_rects):
            if pw <= fw and ph <= fh:
                # Place partition at top-left of free rectangle
                x, y = fx, fy
                self.placed_partitions.append((partition_id, x, y, pw, ph))
                
                # Remove used rectangle
                self.free_rects.pop(i)
                
                # Add remaining free rectangles (guillotine split)
                if fw - pw > 0:
                    self.free_rects.append((fx + pw, fy, fw - pw, ph))
                if fh - ph > 0:
                    self.free_rects.append((fx, fy + ph, fw, fh - ph))
                if fw - pw > 0 and fh - ph > 0:
                    self.free_rects.append((fx + pw, fy + ph, fw - pw, fh - ph))
                
                return (self.coord[0] + x, self.coord[1] + y)  # global coordinates
        return None  # cannot fit