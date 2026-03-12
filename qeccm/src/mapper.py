# Typing

# Numerics
import math
import random
from collections import deque
from itertools import product

from qiskit.circuit import QuantumRegister

# Qiskit transpiler
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.layout import Layout

# Hypergraph stuff
from qeccm.circuit.hypergraph_circuit import PartitionedHyperGraph
from qeccm.src.qpublock import QPUBlock, dimension_to_linear_index, linear_index_to_dimension, plot_block_counts


class GenericMapper(AnalysisPass):
    def __init__(self):
        """GenericMapper initializer"""
        super().__init__()

        # Backend for mapping
        # self.backend = None
        # Dictionary with mapping of nodes
        # self.mapping = {}

    def perform_mapping(self):
        raise NotImplementedError

    def visualize_mapping(self, filename) -> None:
        """Visualize mapping on backend"""

        # self.backend.visualize_mapping(self.mapping, filename)
        # TODO: this needs to be replaced by plot_circuit_layout from backend_utils
        # self.backend.visualize_mapping(self.property_set["block_node_map"], filename)


class RandomMapper(GenericMapper):
    """Random mapping to the backend.

    Idea:
    - Randomly map necessary qubits to the backend (currently one-to-one mapping)
    - Since most algorithms are heuristics, this can be used as some sort of baseline for comparison
    """

    def __init__(self, backend):
        """RandomMapper initializer"""
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
            # print(nodes_in_backend)

            # Map all nodes randomly to chiplet
            for n, node in enumerate(block[1]):
                # Pick a random element
                picked = random.choice(nodes_in_backend)
                # Remove the picked element from the list
                nodes_in_backend.remove(picked)

                full_node_map[node] = picked

        layout = Layout()
        regs = dag.qubits + list(dag.qregs.values())

        hgc = self.property_set["hyper_dag"]
        for reg in regs:
            if isinstance(reg, QuantumRegister):
                layout.add_register(reg)
            else:
                # Map qubit id to graph id (since the partitioning works on the graph ids)
                qubit_to_node = reg._index  # hgc.node_idx.get(reg._index)

                # Add qubit mapping from partitioning
                if qubit_to_node is not None:
                    # Virtual qubit is used and mapped. Get the mapping from the mapping list
                    p_b = full_node_map[qubit_to_node]

                    # print(f"mapping qubit {reg._index} as {qubit_to_node} to {p_b}")
                    layout.add(reg, p_b)

        # Virtual to physical qubit mapping
        self.property_set["layout"] = layout


class TrivialMapper(GenericMapper):
    """Map to chiplet backend"""

    def __init__(self, backend, patch_initialization: str = ""):
        """TrivialMapper initializer

        :param backend: _description_
        :type backend: _type_
        :param patch_initialization: _description_, defaults to "center"
        :type patch_initialization: str, optional
        """
        super().__init__()

        # Coupling map to map the dag to
        self.coupling_map = backend.coupling_map
        self.backend = backend

        # Initialization location of patches on chiplets. More information in the documentation of QPUBlock
        self.patch_initialization = patch_initialization

    def run(self, dag: DAGCircuit) -> None:
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

    def generate_layout(self, dag: DAGCircuit, vq_to_pq: dict) -> Layout:
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
            elif reg._index in vq_to_pq:
                # Get physical qubit
                physical_qubit = vq_to_pq[reg._index]
                # Map mapping from virtual qubit to physical qubit
                layout.add(reg, physical_qubit)

        # Virtual to physical qubit mapping
        self.property_set["layout"] = layout

        return layout

    def map_partition_on_qpu(
        self, partitioned_hg: PartitionedHyperGraph, partition_to_qpu: dict, utilized_qpus: dict
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

        # Dictionary with virtual_qubit to physical_qubit mapping
        placement = {}
        # Map from physical qubits to partition
        pq_to_partition = {}

        # Iterate over all QPUs
        for qpu, qpu_partitions in utilized_qpus.items():
            if qpu_partitions.placed_partitions:
                # Physical qubits for this QPU
                nodes_on_qpu = self.backend.get_chiplet_at(
                    dimension_to_linear_index(qpu, self.backend.c1, self.backend.c2)
                )

                # Iterate over all partitions that are placed on this QPU
                for p in qpu_partitions.placed_partitions:
                    partition_id, local_x, local_y, patch_width, patch_height = p
                    # Virtual qubits for this partition
                    nodes_of_partition = partitions[partition_id]
                    print(pre_defined_partitions[partition_id]["indices"] == nodes_of_partition)
                    print(f"Trying to place nodes {nodes_of_partition}")

                    # TODO: Extract the type of patch from either the partition or somewhere
                    rotated_full = False
                    code_distance = -1
                    if pre_defined_partitions != None:
                        match pre_defined_partitions[partition_id]["type"]:
                            case "rectangle":
                                rectangle = True
                            case "rotated_surface_code":
                                rotated_full = True
                                code_distance = pre_defined_partitions[partition_id]["distance"]
                            case "rotated_surface_code_ancilla":
                                rotated_full = True
                                code_distance = pre_defined_partitions[partition_id]["distance"]
                    else:
                        # TODO: Calculate partition type from number of nodes
                        # TODO: Calculate code distance from number of nodes
                        print("Trying to place partition without pre-defined partitions")

                    if rotated_full:
                        # Place a rotated_surface_code patch (either memory or ancilla region) to the QPU

                        # Calculate starting row and column given the code distance
                        if code_distance == 3:
                            start_row = local_y + 6
                            column_length = 8
                        elif code_distance == 5:
                            # start_row = local_y + 8
                            start_row = local_y + 10
                            column_length = 12
                        elif code_distance == 7:
                            start_row = local_y + 14
                            column_length = 16
                        elif code_distance == 9:
                            start_row = local_y + 18
                            column_length = 20
                        elif code_distance == 9:
                            start_row = local_y + 22
                            column_length = 24
                        elif code_distance == 9:
                            start_row = local_y + 26
                            column_length = 28
                        elif code_distance == 15:
                            start_row = local_y + 30
                            column_length = 32

                        # Define starting row and column
                        # TODO: Fix this mess
                        row = start_row
                        col = local_x

                        if patch_width == 1:
                            # Place vertical patch

                            index = 0
                            i = 0
                            while index < len(nodes_of_partition):
                                # print((row-i)*self.backend.m)
                                p_index = nodes_on_qpu[(row - i) * self.backend.m + col]
                                placement[nodes_of_partition[index]] = p_index
                                pq_to_partition[p_index] = pre_defined_partitions[partition_id]

                                i += 2
                                index += 1
                        elif patch_height == 1 and patch_width > 1:
                            # Place horizontal patch

                            # TODO: Fix row and column
                            row = local_y
                            col = local_x

                            index = 0
                            while index < len(nodes_of_partition):
                                # print((row-i)*self.backend.m)
                                # TODO: Fix this
                                p_index = nodes_on_qpu[(row) * self.backend.m + col]
                                placement[nodes_of_partition[index]] = p_index
                                pq_to_partition[p_index] = pre_defined_partitions[partition_id]

                                index += 1
                                col += 1

                        else:
                            # Place full block
                            col_iter = 0
                            i = 0
                            index = 0

                            while index < len(nodes_of_partition):
                                p_index = nodes_on_qpu[(row - i) * self.backend.m + col]
                                # Assign virtual to physical qubit
                                placement[nodes_of_partition[index]] = p_index
                                # Assign physical qubit to partition
                                pq_to_partition[p_index] = pre_defined_partitions[partition_id]

                                i += 2
                                index += 1

                                if i == column_length or (i == (column_length - 2) and col_iter % 2 != 0):
                                    if col_iter % 2 != 0:
                                        row = start_row
                                        col += 1
                                    else:
                                        row = start_row - 1

                                    col_iter += 1
                                    i = 0

        # Save the physical qubit to partition mapping
        self.property_set["pq_to_partition"] = pq_to_partition

        return placement

    def assign_partition_to_qpu(self, partitioned_hg: PartitionedHyperGraph, dag: DAGCircuit) -> dict:
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
        placement, blocks = self.placement_aware_assignment(
            partitioned_hg, dag, width=self.backend.c2, height=self.backend.c1
        )

        # Plot the assignment of partitions to QPU
        plot_block_counts(
            width=self.backend.c2,
            height=self.backend.c1,
            block_assignments=blocks,
            filename="tests/data/figures/partition_to_qpu_2.png",
        )

        return placement, blocks

    def placement_aware_assignment(
        self, partitioned_hg: PartitionedHyperGraph, dag: DAGCircuit, width: int, height: int
    ) -> tuple[dict, dict]:
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
            partition_size[partition_key] = len(nodes)  # math.sqrt(len(nodes))

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

        # Calculate defective qubit positions for the QPUs
        defective_qubits_coordinates = {}
        for x, y in product(range(width), range(height)):
            # Get linear index of defective qubits
            qpu_pos = dimension_to_linear_index((x, y), self.backend.c1, self.backend.c2)
            qpu_defective_qubits = self.backend.chiplet_to_defective_qubits[qpu_pos]

            # Calculate (x, y) index given linear index of defective qubits
            qpu_defective_qubits_coords = []
            for qi in qpu_defective_qubits:
                x_i, y_i = linear_index_to_dimension(qi, self.backend.m)
                qpu_defective_qubits_coords.append((x_i, y_i))

            # Assign coordinates of defective qubit to every qpu
            defective_qubits_coordinates[(x, y)] = qpu_defective_qubits_coords

            # print(f"{x}, {y} has defective qubits at {qpu_defective_qubits_coords}")

        # Initialize all QPUs with their widht and height, as well as coordinates. The widht and height are used for
        # calculating which partitions (given their width and height) can be placed on this QPU.
        qpu_blocks = {
            (x, y): QPUBlock(
                self.backend.m,
                self.backend.n,
                (x, y),
                no_placement_zones=defective_qubits_coordinates[(x, y)],
                patch_initialization=self.patch_initialization,
            )
            for x, y in product(range(width), range(height))
        }

        placement = {}
        pre_defined_partitions = self.property_set["pre_defined_partitions"]

        # Place the first partition at the "top left" (depending on the chiplet layout) chiplet
        current_x = 0
        current_y = 0

        # Iterate over all connected partitions
        for partition_bfs in connected_partitions:
            print(partition_bfs)

            # Iterate over all partitions that are connected with each other
            for li, partition_id in enumerate(partition_bfs):
                # print(partition_id)

                # Extract width and height of the patch from the pre-defined patches. Otherwise calculate from the
                # number of nodes
                if pre_defined_partitions != None:
                    pw = pre_defined_partitions[partition_id]["width"]
                    ph = pre_defined_partitions[partition_id]["height"]
                else:
                    # TODO: Implement this
                    print("Calculating width and height of partition given the number of nodes!")

                if li == 0:
                    # Place partition
                    # There are different options to place new partitions to the chiplet. This also heavily depends on
                    # the chiplet layout. For now we assume the grid layout.
                    # For the grid layout is is either possible to fill the grid from the top to the right, or from the
                    # top to the bottom. The option implemented iterates from the top left to the bottom left

                    # Find placement for the first partition in this bfs
                    while True:
                        pos = qpu_blocks[(current_x, current_y)].place_partition(partition_id, pw, ph)

                        if pos != None:
                            print(f"Placed partition on QPU {current_x}{current_y}")
                            placement[partition_id] = pos
                            break

                        current_y += 1
                        if current_y == height:
                            current_x += 1
                            current_y = 0
                        if current_x == width:
                            break
                else:
                    # Partition which the current partition should be placed relative to
                    partition_anchor = partitions[partition_bfs[li - 1]]
                    # Partition that we want to place
                    partition_current = partitions[partition_id]

                    if min(partition_current) < max(partition_anchor):
                        # Place at the bottom
                        print(f"Place {partition_id} below {partition_bfs[li - 1]}")

                        # Note that above and below is flipped in the QPUBlock, as the QPUBlock starts from top left,
                        # while the backend starts from the bottom left.
                        pos = qpu_blocks[(current_x, current_y)].place_relative(
                            partition_id, pw, ph, partition_bfs[li - 1], "above"
                        )
                        print(pos)
                        if pos == None:
                            # If it is not possible to place the partition to the bottom on this QPU, try to place it
                            # on the QPU below
                            for new_x in range(current_x, width):
                                for new_y in range(current_y + 1, height):
                                    pos = qpu_blocks[(new_x, new_y)].place_partition(partition_id, pw, ph)

                                    if pos != None:
                                        current_y = new_y
                                        break
                                if pos != None:
                                    current_x = new_x
                                    break

                        print(f"Placed partition {partition_id} on QPU {current_x}{current_y}")
                        placement[partition_id] = pos
                    else:
                        # place to the right
                        print(f"Place {partition_id} to the right of {partition_bfs[li - 1]}")

                        # If it is not possible to place the partition to the right on this QPU, select the next QPU
                        # to the right of the current selected one
                        pos = qpu_blocks[(current_x, current_y)].place_relative(
                            partition_id, pw, ph, partition_bfs[li - 1], "right"
                        )

                        if pos == None:
                            # If it is not possible to place the partition to the right on this QPU, select the QPU
                            # to the right of the current one.
                            for new_y in range(current_y, height):
                                for new_x in range(current_x + 1, width):
                                    pos = qpu_blocks[(new_x, new_y)].place_partition(partition_id, pw, ph)

                                    if pos != None:
                                        current_x = new_x
                                        break
                                if pos != None:
                                    current_y = new_y
                                    break

                            ## Update index of utilized QPU
                            # current_x += 1

                        print(f"Placed partition {partition_id} on QPU {current_x}{current_y}")
                        placement[partition_id] = pos

            print("\n\n")
            print("placing new bfs")
            # Try to place the partition below the last bfs_partition
            current_y += 1

            if current_y == height:
                # If this is not possible, place at the top of the chiplet layout
                current_y = 0
                current_x += 1
            else:
                # If it is possible to place the partition below, place it to the left
                current_x -= 1

        return placement, qpu_blocks
