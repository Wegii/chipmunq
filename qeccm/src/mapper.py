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
        partition_to_qpu, utilized_qpus = self.assign_partition_to_qpu(partitioned_hgc, dag)

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

        pre_defined_partitions = self.property_set["pre_defined_partitions"]
            
        def dimension_to_linear_index(x, w, h):
            x1, x2 = x  # unpack coordinates

            idx = x2 * h + x1
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
                    print(pre_defined_partitions[partition_id]['indices'] == nodes_of_partition)
                    print(f"Trying to place nodes {nodes_of_partition}")

                    # TODO: Extract the type of patch from either the partition or somewhere
                    rotated_full = False
                    code_distance = -1
                    if pre_defined_partitions != None:
                        match pre_defined_partitions[partition_id]['type']:
                            case "rectangle":
                                rectangle = True
                            case "rotated_surface_code":
                                rotated_full = True
                                code_distance = pre_defined_partitions[partition_id]['distance']
                            case "rotated_surface_code_ancilla":
                                rotated_full = True
                                code_distance = pre_defined_partitions[partition_id]['distance']
                    else:
                        # TODO: Calculate partition type from number of nodes
                        # TODO: Calculate code distance from number of nodes
                        print("Trying to place partition without pre-defined partitions")
                
   
                    if rotated_full:
                        # Place a rotated_surface_code patch (either memory or ancilla region) to the QPU

                        # Calculate starting row and column given the code distance
                        if code_distance == 3:
                            pass
                        elif code_distance == 5:
                            #start_row = local_y + 8
                            start_row = local_y + 10
                            column_length = 12
                        elif code_distance == 7:
                            # TODO: Implement distance 7
                            print("Distance 7 not implemented!")
                            pass
                        elif code_distance == 9:
                            # TODO: Implement distance 9
                            print("Distance 9 not implemented!")
                            pass
                        
                        
                        # Define starting row and column
                        # TODO: Fix this mess
                        row = start_row
                        col = local_x

                        if patch_width == 1:
                            # Place vertical patch

                            index = 0
                            i = 0
                            while index < len(nodes_of_partition):
                                #print((row-i)*self.backend.m)
                                placement[nodes_of_partition[index]] = nodes_on_qpu[(row-i)*self.backend.m + col]
                                i += 2
                                index += 1
                        elif patch_height == 1 and patch_width > 1:
                            # Place horizontal patch

                            # TODO: Fix row and column
                            row = local_y
                            col = local_x

                            index = 0
                            while index < len(nodes_of_partition):
                                #print((row-i)*self.backend.m)
                                # TODO: Fix this
                                placement[nodes_of_partition[index]] = nodes_on_qpu[(row)*self.backend.m + col]
                                index += 1
                                col += 1
                        
                        else:
                            # Place full block
                            col_iter = 0
                            i = 0
                            index = 0
                            
                            while index < len(nodes_of_partition):
                                placement[nodes_of_partition[index]] = nodes_on_qpu[(row-i)*self.backend.m + col]
                                i += 2
                                index += 1

                                if i == column_length or (i == (column_length-2) and col_iter % 2 != 0):

                                    if col_iter % 2 != 0:
                                        row = start_row
                                        col += 1
                                    else:
                                        row = start_row - 1

                                    col_iter += 1
                                    i = 0

        return placement

    def assign_partition_to_qpu(self,
                                partitioned_hg: PartitionedHyperGraph,
                                dag: DAGCircuit
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

        placement, blocks = self.placement_aware_assignment(partitioned_hg,
                                                            dag,
                                                            width = self.backend.c2,
                                                            height = self.backend.c1
                                                            )

        # Plot the assignment of partitions to QPU
        self.plot_block_counts(width = self.backend.c2,
                               height = self.backend.c1,
                               block_assignments = blocks,
                               filename="tests/data/figures/partition_to_qpu_2.png")

        return placement, blocks
    
    def placement_aware_assignment(self,
                                   partitioned_hg: PartitionedHyperGraph,
                                   dag: DAGCircuit,
                                   width: int,
                                   height: int) -> tuple[dict, dict]: 
        """Assignment of partition to qpu

        Every partition receives 2d coordinates of the qpu to map to.

        Idea:
            - BFS throught the contracted nodes
            - Place first node
            - Place the next node. If it is connected to the first node, calculate where to place. Options of placement
              are: left, right, bottom, top
            - If placed at bottom or top, place from left to right
            - If placed at left or right, place from right to left

        Placement:
            - Place blocks in the middle if possible. If a node needs to be place below it, do so. If not possible on
              chip, place at the top of the chiplet below. Same idea for placing to the left/right

        :param partitioned_hg: _description_
        :type partitioned_hg: PartitionedHyperGraph
        :param dag: _description_
        :type dag: DAGCircuit
        :param width: _description_
        :type width: int
        :param height: _description_
        :type height: int
        :return: _description_
        :rtype: tuple[dict, dict]
        """
        
        partition_size = {}
        for partition_key, nodes in partitioned_hg._btn.items():
            partition_size[partition_key] = len(nodes) #math.sqrt(len(nodes))

        partitions = {}
        for partition_key, nodes in partitioned_hg._btn.items():
            partition_size[partition_key] = int(math.ceil(math.sqrt(len(nodes))))
            partitions[int(partition_key[2:])] = nodes

        # Construct BFS ordering of partition dependencies
        collapsed_hg = partitioned_hg._collapsed_phg
        nodes_iter = list(collapsed_hg.nodes())
        visited = set()
        # List that contains all lists of partitions that share a connection
        connected_partitions = []

        start_list = nodes_iter

        # Iterate over all nodes to ensure every component is found. This is necessary, since it is possible to have
        # multiple connected_partitions that do not share any connections with each other
        for seed in start_list:
            if seed in visited:
                continue
            
            # Found a new, unvisited node: start a new connected component search
            current_connected_partition = []
            # Construct queue containing the seed node
            queue = deque([seed])
            
            # Perform BFS to find all nodes connected to the seed
            while queue:
                node = queue.popleft()
                
                # Skip node, if we already check this node and all of its neighbours
                if node in visited:
                    continue

                visited.add(node)
                current_connected_partition.append(node)

                # Add all non visited neighbors
                for nbr in collapsed_hg.neighbors(node):
                    if nbr not in visited:
                        queue.append(nbr)
                        
            # Add all partitions that are connected with each other
            connected_partitions.append(current_connected_partition)

        print(connected_partitions)

        # Initialize all QPUs with their widht and height, as well as coordinates. The widht and height are used for
        # calculating which partitions (given their width and height) can be placed on this QPU.
        qpu_blocks = {(x, y): QPUBlock(self.backend.m, self.backend.n, (x, y))
              for x, y in product(range(width), range(height))}
        
        print(qpu_blocks)
        placement = {}
        pre_defined_partitions = self.property_set["pre_defined_partitions"]

        # Iterate over all connected partitions
        for partition_bfs in connected_partitions:
            # Iterate over all partitions that are connected with each other
            for li, partition_id in enumerate(partition_bfs):
                #print(partition_id)

                # Extract width and height of the patch from the pre-defined patches. Otherwise calculate from the
                # number of nodes
                if pre_defined_partitions != None:
                    pw = pre_defined_partitions[partition_id]['width']
                    ph = pre_defined_partitions[partition_id]['height']
                else:
                    # TODO: Implement this
                    print("Calculating width and height of partition given the number of nodes!")

                if li == 0:
                    # Place partition
                    # There are different options to place new partitions to the chiplet. This also heavily depends on
                    # the chiplet layout. For now we assume the grid layout.
                    # For the grid layout is is either possible to fill the grid from the top to the right, or from the
                    # top to the bottom. The option implemented iterates from the top left to the bottom left
                    
                    current_x = 0
                    current_y = 0
                    for y in range(height):
                        for x in range(width):
                            pos = qpu_blocks[(x, y)].place_partition(partition_id, pw, ph)

                            # QPU found
                            if pos != None:
                                print(f"Placed partition on QPU {current_x}{current_y}")
                                placement[partition_id] = pos
                                current_x = x
                                current_y = y
                                break

                        # QPU found
                        if pos != None:
                            break
                else:
                    # Partition which the current partition should be placed relative to
                    partition_anchor = partitions[partition_bfs[li-1]]
                    # Partition that we want to place
                    partition_current = partitions[partition_id]
                    print("anchor")
                    print(min(partition_anchor))
                    print(max(partition_anchor))
                    print("current")
                    print(min(partition_current))
                    print(max(partition_current))
                    ##min(partition_current) > min(partition_anchor) or max(partition_current) > max(partition_anchor):
                    #6 > 17
                    if min(partition_current) < max(partition_anchor):#min(partition_current) > min(partition_anchor) and min(partition_current) < max(partition_anchor):
                        # Place at the bottom
                        print(f"Place {partition_id} below {partition_bfs[li-1]}")

                        # Note that above and below is flipped in the QPUBlock, as the QPUBlock starts from top left,
                        # while the backend starts from the bottom left.
                        pos = qpu_blocks[(current_x, current_y)].place_relative(partition_id,
                                                                                pw,
                                                                                ph,
                                                                                partition_bfs[li-1],
                                                                                "above"
                                                                                )

                        if pos == None:
                            # If it is not possible to place the partition to the bottom on this QPU, select the QPU
                            # below the current one. This should always be possible
                            pos = qpu_blocks[(current_x, current_y+1)].place_partition(partition_id, pw, ph)
                            
                            if pos == None:
                                ValueError(f"Something wrong for placement below!")

                            # Update index of utilized QPU
                            current_y += 1

                        print(f"Placed partition {partition_id} on QPU {current_x}{current_y}")
                        placement[partition_id] = pos
                    else:
                        # place to the right
                        print(f"Place {partition_id} to the right of {partition_bfs[li-1]}")

                        # If it is not possible to place the partition to the right on this QPU, select the next QPU
                        # to the right of the current selected one
                        pos = qpu_blocks[(current_x, current_y)].place_relative(partition_id,
                                                                                pw,
                                                                                ph,
                                                                                partition_bfs[li-1],
                                                                                "right"
                                                                                )

                        if pos == None:
                            # If it is not possible to place the partition to the bottom on this QPU, select the QPU
                            # below the current one. This should always be possible
                            pos = qpu_blocks[(current_x+1, current_y)].place_partition(partition_id, pw, ph)
                            
                            if pos == None:
                                ValueError(f"Something wrong for placement below!")

                            # Update index of utilized QPU
                            current_x += 1

                        print(f"Placed partition {partition_id} on QPU {current_x}{current_y}")
                        placement[partition_id] = pos
        
        
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
    """Class representing a QPU chiplet with no-placement zones"""

    def __init__(self, width: int, height: int, block_coord: tuple, no_placement_zones: list[tuple] = None):
        self.width = width
        self.height = height
        self.coord = block_coord

        print(f"block has width {width} and height {height}")

        # List of (x, y, w, h) for forbidden areas
        self.no_placement_zones = no_placement_zones if no_placement_zones is not None else []
        
        # free rectangles inside block
        self.free_rects = [(0, 0, width, height)]
        # list of (partition_id, x, y, w, h)
        self.placed_partitions = []

    # --- Helper methods (_overlaps, _find_covering_free_rect, _split_free_rect, place_partition) remain as before ---
    
    def _overlaps_partitions(self, x, y, w, h):
        for pid, px, py, pw, ph in self.placed_partitions:
            if not (x + w <= px or px + pw <= x or y + h <= py or py + ph <= y):
                return True
        return False

    def _overlaps_forbidden(self, x, y, w, h):
        for fx, fy, fw, fh in self.no_placement_zones:
            if not (x + w <= fx or fx + fw <= x or y + h <= fy or fy + fh <= y):
                return True
        return False

    def _overlaps(self, x, y, w, h):
        return self._overlaps_partitions(x, y, w, h) or self._overlaps_forbidden(x, y, w, h)

    def _find_covering_free_rect(self, x, y, w, h):
        for i, (fx, fy, fw, fh) in enumerate(self.free_rects):
            if (x >= fx and y >= fy and 
                x + w <= fx + fw and 
                y + h <= fy + fh):
                return i, (fx, fy, fw, fh)
        return None, None

    def _split_free_rect(self, index, fx, fy, fw, fh, x, y, w, h):
        del self.free_rects[index]

        # left side
        if x > fx:
            self.free_rects.append((fx, fy, x - fx, fh))

        # right side
        if x + w < fx + fw:
            self.free_rects.append((x + w, fy, (fx + fw) - (x + w), fh))

        # top side (above the partition) - constrained to partition's width
        if y > fy:
            self.free_rects.append((x, fy, w, y - fy))

        # bottom side (below the partition) - constrained to partition's width
        if y + h < fy + fh:
            self.free_rects.append((x, y + h, w, (fy + fh) - (y + h)))
    
    def place_partition(self, partition_id, pw, ph):
        # Preferred y: center vertically in the WHOLE block
        preferred_y = (self.height - ph) // 2

        for i, (fx, fy, fw, fh) in enumerate(self.free_rects):
            if pw <= fw and ph <= fh:
                
                if fy <= preferred_y and preferred_y + ph <= fy + fh:
                    y_to_check = preferred_y
                else:
                    y_to_check = fy
                
                x_to_check = fx

                if not self._overlaps(x_to_check, y_to_check, pw, ph):
                    x, y = x_to_check, y_to_check
                    
                    self.placed_partitions.append((partition_id, x, y, pw, ph))
                    self._split_free_rect(i, fx, fy, fw, fh, x, y, pw, ph)

                    print(f"Placed {partition_id} at ({x}, {y})")

                    return (self.coord[0] + x, self.coord[1] + y)

        return None

    # ----------------------------------------------------------
    # MODIFIED: Relative placement with iterative search (shift)
    # ----------------------------------------------------------
    def place_relative(self, partition_id, pw, ph, anchor_id, direction, max_shift=5):
        anchor = next((p for p in self.placed_partitions if p[0] == anchor_id), None)
        if anchor is None:
            raise ValueError(f"Anchor partition {anchor_id} not found.")

        _, ax, ay, aw, ah = anchor
        
        # Initial target position (shift=0)
        if direction == "right":
            base_x, base_y = ax + aw, ay
        elif direction == "left":
            base_x, base_y = ax - pw, ay
        elif direction == "below":
            base_x, base_y = ax, ay + ah
        elif direction == "above":
            base_x, base_y = ax, ay - ph
        else:
            raise ValueError("Direction must be one of: right/left/above/below")

        # --- Search Loop ---
        # Search starting from 0 shift up to max_shift
        for shift in range(max_shift + 1):
            
            # Calculate current placement attempt (x, y) based on shift
            x, y = base_x, base_y
            
            if shift > 0:
                if direction in ("right", "left"):
                    # Shift vertically
                    y += shift # Try shifting 'up' first (lower y value is higher on screen/chip)
                    
                    # NOTE: You could also implement a strategy to try shifting in the
                    # opposite direction (e.g., y -= shift) or both, but a single
                    # incremental shift is a common simple heuristic.
                
                elif direction in ("below", "above"):
                    # Shift horizontally
                    x += shift # Try shifting 'right' first
                    
                    # NOTE: As above, a strategy could include x -= shift

            # 1. Bounds check
            if x < 0 or y < 0 or x + pw > self.width or y + ph > self.height:
                # If even the base position (shift=0) is out of bounds, we fail immediately.
                # For shift > 0, we simply stop this iteration.
                continue

            # 2. Overlap check (with partitions AND forbidden zones)
            if self._overlaps(x, y, pw, ph):
                continue # Try next shift

            # 3. Find free rect that fully contains this placement
            idx, rect = self._find_covering_free_rect(x, y, pw, ph)
            if idx is None:
                continue # Try next shift

            # If all checks pass, we have found a valid placement!
            fx, fy, fw, fh = rect

            # Place partition
            self.placed_partitions.append((partition_id, x, y, pw, ph))

            # Split the free rectangle
            self._split_free_rect(idx, fx, fy, fw, fh, x, y, pw, ph)
            
            print(f"Placed {partition_id} at ({x}, {y}) with shift {shift} (Direction: {direction})")

            return (self.coord[0] + x, self.coord[1] + y)

        # If the loop finishes without finding a valid position
        print(f"Placement for {partition_id} failed: No valid position found within {max_shift} units of shift.")
        return None