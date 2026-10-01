from __future__ import annotations

import copy
import logging

# QECCircuit
import random
from itertools import product
import numpy as np
import qiskit
from functools import reduce
from scipy.sparse import identity, hstack, kron, csr_matrix
from collections import deque

from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit


# Plotting
import stim
from qiskit import QuantumCircuit
from tqec import compile_block_graph
from tqec.computation.block_graph import BlockGraph
from tqec.computation.cube import ZXCube
from tqec.computation.pipe import PipeKind
from tqec.gallery import cnot, memory, three_cnots
from tqec.gallery.steane_encoding import steane_encoding
from tqec.utils.enums import Basis
from tqec.utils.position import Position3D

from experiments.exp_utils.circuit_utils import stim_to_qiskit, row_echelon, inverse, compute_code_distance, kernel


def get_tqec_cnot_rotated(distance_scale: int = 2, n1: int = 1, n2: int = 0) -> tuple[StimCodeCircuit, list]:
    circuit_generator = QECCircuit()
    stim_circuit, partitions = circuit_generator.single_cnot_full_memory(distance_scale=distance_scale, n1=n1, n2=n2)

    with open(f"stim_cnot_d_{distance_scale}.stim", "w") as f:
        print(stim_circuit, file=f)

    return stim_circuit, partitions


class GenericCircuit:
    def __init__(self, nq: int):
        self.num_qubits = nq

    def generate_circuit(self, num_patches):

        # Generate GHZ circuit
        ghz = QuantumCircuit(self.num_qubits)
        # Apply H on qubit 0
        # ghz.h(0)
        # Apply CNOT chain
        for i in range(self.num_qubits - 1):
            ghz.cx(i, i + 1)

        """
        # Apply H on qubit 0
        ghz.h(0)
        # Apply CNOT chain
        for i in range(self.num_qubits - 1):
            ghz.cx(i, i + 1)

        # Apply H on qubit 0
        ghz.h(0)
        # Apply CNOT chain
        for i in range(self.num_qubits - 1):
            ghz.cx(i, i + 1)

        # Apply H on qubit 0
        ghz.h(0)
        # Apply CNOT chain
        for i in range(self.num_qubits - 1):
            ghz.cx(i, i + 1)
        """

        # Create num_patches of the GHZ circuit
        total_qubits = self.num_qubits * num_patches
        patched_circuit = QuantumCircuit(total_qubits, name="Big_GHZ")

        # Stitch GHZ patches together
        for patch_index in range(num_patches):
            offset = patch_index * self.num_qubits

            # Append with correct qubit mapping
            patched_circuit.append(ghz.to_instruction(), qargs=list(range(offset, offset + self.num_qubits)))

        return patched_circuit


class QECMemory:
    """QECC memory circuits

    Supported codes:
        - Color Code
        - Surface Code
        - Steane Code
    """

    # QECC Memory circuits
    def __init__(self, nq: int, gate_set: list = []):
        self.num_qubits = nq

        self.gate_set = gate_set

    def draw_circuit(self, circuit: qiskit.QuantumCircuit, filename: str = "") -> None:
        renderer = "mpl"

        if filename == "":
            circuit.draw(output=renderer)
        else:
            circuit.draw(output=renderer, filename=filename)

    def _check_circuit_gateset(self, circuit) -> qiskit.QuantumCircuit:

        # qiskit transpilation step with the gateset
        self.gate_set
        return circuit

    def _generate_code_from_stim(self, codename: str, distance_scale: int = -1) -> StimCodeCircuit:
        """Generate QECC memory circuit using stim

        :param codename: Name of QEC code to generate
        :type codename: str
        :return: QECC memory circuit
        :rtype: qiskit.QuantumCircuit
        """

        d = 2 * distance_scale + 1

        if d < 3:
            logging.error(
                f"Code distance too small! {codename} with distance {d} and {self.num_qubits} qubits: Execution not possible"
            )
            exit(1)

        # Generate code
        cycles = d

        if codename == "surface":
            stim_circuit = stim.Circuit.generated(
                "surface_code:unrotated_memory_z",
                # "surface_code:rotated_memory_z",
                rounds=cycles,
                distance=d,
            )
        if codename == "rotated_surface":
            stim_circuit = stim.Circuit.generated("surface_code:rotated_memory_x", rounds=cycles, distance=d)
        else:
            stim_circuit = None
        return stim_circuit

    def generate_code_memory(self, codename: str, patches: int = 0, distance_scale: int = 1) -> StimCodeCircuit:
        """Generate QECC memory circuit

        Currently used as wrapper around the circuit generation function from stim. Extend this function if
        some unsupported QEC code should be generated.

        :param codename: Name of QEC code to generate
        :type codename: str
        :return: QECC memory circuit
        :rtype: qiskit.QuantumCircuit
        """
        qecc_mem = self._generate_code_from_stim(codename, distance_scale=distance_scale)

        # TODO: Check if circuit only consists of allowed gates, i. e. it is important that the circuit only consists of
        # e. g. two qubit gates as this will influence the hypergraph construction
        # Note: The allowed gateset also needs to be taken into account in the local routing algorithm.
        qecc_mem_transpiled = qecc_mem  # self._check_circuit_gateset(qecc_mem)

        return qecc_mem_transpiled


def _cnot_partitions(k: int) -> tuple[list[dict], int]:
    """Qubit partitions of the tqec single-CNOT layout (control, ancilla, target, CA, AT).

    The tqec qubit ordering is regular in k:
      - top region: d+1 "even" rows of 2*(d+1) qubits (control | ancilla),
        interleaved with d "odd" rows of 2*d+1 qubits (control | CA | ancilla)
      - then the AT ancilla column (d qubits), then the target patch (d^2 + (d+1)^2 qubits).
    Reproduces the previously hardcoded k = 1, 2, 3, 7 partitions exactly.

    :param k: tqec distance scale (code distance d = 2k + 1)
    :return: (list of partition dicts, max qubit index of one CNOT)
    """
    d = 2 * k + 1
    w = d + 1
    even, odd = 2 * w, 2 * d + 1
    period = even + odd
    top = period * d + even

    control, ancilla, ca = [], [], []
    for r in range(d + 1):
        b = r * period
        control += range(b, b + w)
        ancilla += range(b + w, b + even)
        if r < d:
            b += even
            control += range(b, b + d)
            ca.append(b + d)
            ancilla += range(b + d + 1, b + odd)

    at = list(range(top, top + d))
    target = list(range(top + d, top + d + d * d + w * w))
    max_qubit = target[-1]

    patch = dict(width=w, height=2 * d + 1, distance=d, type="rotated_surface_code")
    partitions = [
        # Control patch
        {"indices": control, **patch},
        # Ancilla patch
        {"indices": ancilla, **patch},
        # Target patch
        {"indices": target, **patch},
        # CA patch (control-ancilla merge)
        {"indices": ca, "width": w, "height": 1, "distance": d, "type": "rotated_surface_code_ancilla"},
        # AT patch (ancilla-target merge)
        {"indices": at, "width": 1, "height": w, "distance": d, "type": "rotated_surface_code_ancilla"},
    ]
    return partitions, max_qubit



class QECCircuit:
    """Logical circuits using lattice surgery

    Tasks
        - TODO: Needs quite a lot of improvement
        - TODO: Perform circuit verification + circuit conversion (in topologiq and pyzx) verification
        - TODO: Pass custom generic circuit
        - TODO:


    References:
    - Workflow taken from tQEC library
    """

    def __init__(self):
        pass

    def single_memory_patch(self, distance_scale: int = 1):
        """Generate single logical memory

        Code adapted from: https://tqec.github.io/tqec/gallery/memory.html

        :param distance_scale: Scale of surface code patch, defaults to 1
        :type distance_scale: int, optional
        """
        # TODO: add option for manhattan radius

        graph = memory(Basis.Z)
        compiled_graph = compile_block_graph(graph)
        stim_circuit = compiled_graph.generate_stim_circuit(k=distance_scale, manhattan_radius=3)

        return stim_to_qiskit(stim_circuit), stim_circuit

    def multiple_memory_patch(self, num_x1, num_x2: int = 0, distance_scale: int = 1):

        g = BlockGraph("Move Rotation")

        if num_x2 == 0:
            # One line of memory patches
            for x1 in range(num_x1):
                nodes = [
                    (Position3D(x1, 0, 0), "P", "In"),
                    (Position3D(x1, 0, 1), "ZXZ", ""),
                    (Position3D(x1, 0, 3), "P", "Out"),
                ]
                for pos, kind, label in nodes:
                    g.add_cube(pos, kind, label)

                pipes = [(0, 1), (1, 2)]  # , (2, 3), (3, 4)]
                for p0, p1 in pipes:
                    g.add_pipe(nodes[p0][0], nodes[p1][0])

                g.fill_ports({"In": ZXCube.from_str("ZXZ"), "Out": ZXCube.from_str("ZXZ")})
        else:
            # 2d grid of memory patches
            for x1, x2 in product(range(num_x1), range(num_x2)):
                nodes = [
                    (Position3D(x1, x2, 0), "P", "In"),
                    (Position3D(x1, x2, 1), "ZXZ", ""),
                    (Position3D(x1, x2, 2), "P", "Out"),
                ]
                for pos, kind, label in nodes:
                    g.add_cube(pos, kind, label)

                pipes = [(0, 1), (1, 2)]  # , (2, 3), (3, 4)]
                for p0, p1 in pipes:
                    g.add_pipe(nodes[p0][0], nodes[p1][0])

                g.fill_ports({"In": ZXCube.from_str("ZXZ"), "Out": ZXCube.from_str("ZXZ")})

        compiled_graph = compile_block_graph(g)
        stim_circuit = compiled_graph.generate_stim_circuit(k=distance_scale, manhattan_radius=3)

        return stim_circuit

    def full_memory_patch(self, distance_scale: int = 1):
        """This implements a memory patch on which a hadamard is applied. This allows tqec to assign and utilize all
        qubits in the patch

        :param distance_scale: _description_, defaults to 1
        :type distance_scale: int, optional
        :return: _description_
        :rtype: _type_
        """

        g = BlockGraph("HadamardExample")

        # Compatible cubes for Hadamard
        in_cube = Position3D(0, 0, 0)
        g.add_cube(in_cube, "P", "In")

        # Hadamard cube (middle)
        h_cube = Position3D(0, 0, 1)
        g.add_cube(h_cube, "XZZ", "H")

        # Output cube
        out_cube = Position3D(0, 0, 2)
        g.add_cube(out_cube, "P", "Out")

        # Use the built-in Hadamard pipe
        g.add_pipe(Position3D(0, 0, 0), Position3D(0, 0, 1))
        g.add_pipe(
            h_cube,
            out_cube,
            PipeKind(
                Basis.X,
                Basis.Z,
                None,
                has_hadamard=True,
            ),
        )  # TQEC infers default pipe for identity

        # Fill ports
        g.fill_ports(
            {
                "In": ZXCube.from_str("XZZ"),
                "Out": ZXCube.from_str("ZXX"),
            }
        )

        compiled_graph = compile_block_graph(g)
        stim_circuit = compiled_graph.generate_stim_circuit(k=distance_scale, manhattan_radius=3)

        # TODO: This should also return the qubits of all patches
        if distance_scale == 2:
            memory_d5 = [
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    38,
                    39,
                    40,
                    41,
                    42,
                    43,
                    44,
                    45,
                    46,
                    47,
                    48,
                    49,
                    50,
                    51,
                    52,
                    53,
                    54,
                    55,
                    56,
                    57,
                    58,
                    59,
                    60,
                ]
            ]
        else:
            pass

        return stim_circuit

    def single_cnot(self, distance_scale: int = 1):
        """Generate single logical CNOT with lattice surgery.

        Code adapted from: https://tqec.github.io/tqec/gallery/cnot.html

        :param distance_scale: Scale of surface code patch, defaults to 1
        :type distance_scale: int, optional
        """
        # TODO: add option for manhattan radius

        # graph = cz(["XI -> XZ", "IZ -> IZ"])#.rotate(Direction3D.Z, )
        graph = cnot(Basis.X)
        compiled_graph = compile_block_graph(graph)
        stim_circuit = compiled_graph.generate_stim_circuit(k=distance_scale, manhattan_radius=2)

        return stim_to_qiskit(stim_circuit), stim_circuit

    def single_cnot_full_memory_extended(
        self,
        distance_scale: int = 1,
    ):
        g = BlockGraph("Logical CNOT with extended syndrom measurement rounds")

        cnot_counter = 0

        nodes = [
            (Position3D(0, 0, 0), "P", f"In_Control_{cnot_counter}"),
            (Position3D(0, 0, 1), "ZXX", ""),
            (Position3D(0, 0, 2), "ZXZ", ""),
            (Position3D(0, 0, 3), "ZXZ", ""),  # Additional rounds
            # ADD (Position3D(0, 0, Correct Z), "ZXZ", "") for even more rounds on control
            (Position3D(0, 0, 4), "P", f"Out_Control_{cnot_counter}"),  # Adjust Z here if you add another pipe
            (Position3D(0, 1, 1), "ZXX", ""),
            (Position3D(0, 1, 2), "ZXZ", ""),
            (Position3D(1, 1, 0), "P", f"In_Target_{cnot_counter}"),
            (Position3D(1, 1, 1), "ZXZ", ""),
            (Position3D(1, 1, 2), "ZXZ", ""),
            (Position3D(1, 1, 3), "ZXZ", ""),  # Additional cycle
            # ADD (Position3D(1, 1, Correct Z), "ZXZ", "") for even more rounds on target
            (Position3D(1, 1, 4), "P", f"Out_Target_{cnot_counter}"),  # Adjust Z here if you add another pipe
        ]
        for pos, kind, label in nodes:
            g.add_cube(pos, kind, label)

        # add pipes as tuples (source, target) and adjust indices!
        pipes = [
            (0, 1),
            (1, 2),
            (2, 3),  # Control
            (3, 4),
            (1, 5),
            (5, 6),  # Ancilla
            (6, 9),  # Merge
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),  # Target
        ]

        for p0, p1 in pipes:
            g.add_pipe(nodes[p0][0], nodes[p1][0])

        g.fill_ports(ZXCube.from_str("ZXZ"))

        # Compile the block graph and construct stim circuit
        compiled_graph = compile_block_graph(g)
        stim_circuit = compiled_graph.generate_stim_circuit(k=distance_scale, manhattan_radius=2)

        return stim_circuit

    def single_cnot_full_memory(self, distance_scale: int = 1, n1: int = 1, n2: int = 0):

        if n1 >= 1 and n2 == 0:
            # go from left to right and place cnots

            # Contains all patches and operations
            g = BlockGraph("Logical CNOT")

            placement_x = 0
            placement_y = 0
            cnot_counter = 0
            for _ in range(n1):
                nodes = [
                    (Position3D(placement_x, placement_y, 0), "P", f"In_Control_{cnot_counter}"),
                    (Position3D(placement_x, placement_y, 1), "ZXX", ""),
                    (Position3D(placement_x, placement_y, 2), "ZXZ", ""),
                    (Position3D(placement_x, placement_y, 3), "P", f"Out_Control_{cnot_counter}"),
                    (Position3D(placement_x, placement_y + 1, 1), "ZXX", ""),
                    (Position3D(placement_x, placement_y + 1, 2), "ZXZ", ""),
                    (Position3D(placement_x + 1, placement_y + 1, 0), "P", f"In_Target_{cnot_counter}"),
                    (Position3D(placement_x + 1, placement_y + 1, 1), "ZXZ", ""),
                    (Position3D(placement_x + 1, placement_y + 1, 2), "ZXZ", ""),
                    (Position3D(placement_x + 1, placement_y + 1, 3), "P", f"Out_Target_{cnot_counter}"),
                ]
                for pos, kind, label in nodes:
                    g.add_cube(pos, kind, label)

                pipes = [
                    (0, 1),
                    (1, 2),
                    (2, 3),  # Control
                    (1, 4),
                    (4, 5),  # Ancilla
                    (5, 8),  # Merge
                    (6, 7),
                    (7, 8),
                    (8, 9),  # Target
                ]

                for p0, p1 in pipes:
                    g.add_pipe(nodes[p0][0], nodes[p1][0])

                g.fill_ports(ZXCube.from_str("ZXZ"))

                # Every CNOTS needs a width of 2
                placement_x += 2

            # Compile the block graph and construct stim circuit
            compiled_graph = compile_block_graph(g)
            stim_circuit = compiled_graph.generate_stim_circuit(k=distance_scale, manhattan_radius=2)

            # Utilize the patches from the first and add the qubit shift the every additional patch
            # Partition layout of the tqec single-CNOT is regular in k; see _cnot_partitions
            single_partitions, max_qubit = _cnot_partitions(distance_scale)
            qubit_shift = max_qubit + 1
            assert stim_circuit.num_qubits == n1 * qubit_shift, (
                f"Partition layout mismatch for k={distance_scale}: "
                f"stim has {stim_circuit.num_qubits} qubits, expected {n1 * qubit_shift}"
            )

            partitions = []
            for nc in range(n1):
                print(nc)

                # Iterate over all single partitions
                for p in single_partitions:
                    modified_p = copy.deepcopy(p)
                    modified_p["indices"] = [i + nc * qubit_shift for i in modified_p["indices"]]
                    partitions.append(modified_p)

            print(partitions)
        else:
            print("Generating multiple single_cnot")

            # Contains all patches and operations
            g = BlockGraph("Logical CNOT")

            n1 = 2
            n2 = 2

            placement_x = 0
            placement_y = 0
            cnot_counter = 0
            # Used to determine if the cnot is placed downwards+right, or right+downwards, in order to completely
            # fill the grid optimally
            rotation_counter = 0
            for _ in range(n1):
                placement_y = 0

                for _ in range(n2):
                    if rotation_counter % 2 == 0:
                        # Place downwards+right
                        nodes = [
                            (Position3D(placement_x, placement_y, 0), "P", f"In_Control_{cnot_counter}"),
                            (Position3D(placement_x, placement_y, 1), "ZXX", ""),
                            (Position3D(placement_x, placement_y, 2), "ZXZ", ""),
                            (Position3D(placement_x, placement_y, 3), "P", f"Out_Control_{cnot_counter}"),
                            (Position3D(placement_x, placement_y + 1, 1), "ZXX", ""),
                            (Position3D(placement_x, placement_y + 1, 2), "ZXZ", ""),
                            (Position3D(placement_x + 1, placement_y + 1, 0), "P", f"In_Target_{cnot_counter}"),
                            (Position3D(placement_x + 1, placement_y + 1, 1), "ZXZ", ""),
                            (Position3D(placement_x + 1, placement_y + 1, 2), "ZXZ", ""),
                            (Position3D(placement_x + 1, placement_y + 1, 3), "P", f"Out_Target_{cnot_counter}"),
                        ]
                        for pos, kind, label in nodes:
                            g.add_cube(pos, kind, label)

                        pipes = [
                            (0, 1),
                            (1, 2),
                            (2, 3),  # Control
                            (1, 4),
                            (4, 5),  # Ancilla
                            (5, 8),  # Merge
                            (6, 7),
                            (7, 8),
                            (8, 9),  # Target
                        ]

                        for p0, p1 in pipes:
                            g.add_pipe(nodes[p0][0], nodes[p1][0])

                        g.fill_ports(ZXCube.from_str("ZXZ"))

                    else:
                        # Place right+downwards
                        nodes_2 = [
                            (Position3D(placement_x, placement_y, 0), "P", f"In_Control_{cnot_counter}"),
                            (Position3D(placement_x, placement_y, 1), "ZXZ", ""),
                            (Position3D(placement_x, placement_y, 2), "ZXZ", ""),
                            (Position3D(placement_x, placement_y, 3), "P", f"Out_Control_{cnot_counter}"),
                            (Position3D(placement_x + 1, placement_y, 1), "ZXZ", ""),
                            (Position3D(placement_x + 1, placement_y, 2), "ZXX", ""),
                            (Position3D(placement_x + 1, placement_y + 1, 0), "P", f"In_Target_{cnot_counter}"),
                            (Position3D(placement_x + 1, placement_y + 1, 1), "ZXZ", ""),
                            (Position3D(placement_x + 1, placement_y + 1, 2), "ZXX", ""),
                            (Position3D(placement_x + 1, placement_y + 1, 3), "P", f"Out_Target_{cnot_counter}"),
                        ]

                        for pos, kind, label in nodes_2:
                            g.add_cube(pos, kind, label)

                        pipes_2 = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (5, 8), (6, 7), (7, 8), (8, 9)]

                        # Get the absolute positions for the new pipes.
                        new_pipe_positions = [(nodes_2[p0][0], nodes_2[p1][0]) for p0, p1 in pipes_2]

                        # Add the new pipes to the graph.
                        for p0_pos, p1_pos in new_pipe_positions:
                            g.add_pipe(p0_pos, p1_pos)

                        # Fill ports for the new cubes (optional, but good for completeness)
                        g.fill_ports(ZXCube.from_str("ZXZ"))

                    # Counter for labeling the input and output ports of every pipe
                    cnot_counter += 1
                    # Every CNOTS needs a height 2, so increment the placement_y by 2
                    placement_y += 2

                # Every CNOTS needs a width of 1 or 2, depending on orientation
                if rotation_counter % 2 != 0:
                    placement_x += 2
                else:
                    placement_x += 1
                rotation_counter += 1

            # Compile the block graph and construct stim circuit
            compiled_graph = compile_block_graph(g)
            stim_circuit = compiled_graph.generate_stim_circuit(k=distance_scale, manhattan_radius=2)

            patch_size = (2 * distance_scale + 1) * 2 + 1

            # Calculate number of rows for even and odd
            num_even = n2 * 2 * (2 * distance_scale + 1 + 1)
            num_odd = n2 * 2 * (2 * distance_scale + 1 + 1) - 1
            print(num_even)
            print(num_odd)

            # Iterate over column
            num_col = (2 + (n1 - 1) * 2 - 1) * ((2 * distance_scale + 1) * 2 + 1) + ((2 + (n1 - 1) * 2 - 1) - 1)

            row_counter = 0
            qubit_index = 0
            between_1 = True
            perform_horizontal_skip = True

            for nc in range(num_col):
                # This has to be turned on and off every 7 columns or so
                patch_down = False

                if row_counter % 2 == 0:
                    # Even
                    for r in range(num_even):
                        # print(qubit_index)
                        qubit_index += 1

                else:
                    # Odd
                    for r in range(num_odd):
                        # Skip horizontal
                        if patch_down:
                            if (r + 1) % (2 * distance_scale + 2):
                                continue

                        # Skipping horizontal ancilla patch between non-connected patches. Here we need to skip one row
                        # Step where we move from one patch to another from top to bottom
                        patch_move_down = [(2 * (2 * distance_scale + 1) + 1) * (i + 1) + i for i in range(n1)]
                        if r in patch_move_down:
                            continue

                        # Horizontal skip

                        if perform_horizontal_skip:
                            if r in []:
                                continue
                            # pass

                        # Step where we move from one patch to another from left to right
                        patch_move_right = [patch_size * (i + 1) + i for i in range(n1)]
                        if nc in patch_move_right:
                            perform_horizontal_skip = not perform_horizontal_skip

                            skip = np.array(list(range(2 * distance_scale + 2)))
                            vertical_skip = [skip + (i * (2 * (2 * distance_scale + 1) + 2)) for i in range(n2)]
                            # Convert to simple list
                            vertical_skip = np.concatenate(vertical_skip)
                            vertical_skip = vertical_skip.tolist()

                            if between_1:
                                if r in vertical_skip:
                                    continue
                            elif r not in vertical_skip:
                                continue

                        qubit_index += 1

                row_counter += 1

            # Iteate over y (38)
            print(qubit_index)

            partitions = []
            # Compute partitions
            # TODO: This needs to be fixed for multiple cnots
            if distance_scale == 1:
                # 3
                # width = height = 4
                # ancilla width/height = 3
                patch_width = patch_height = 4
                ancilla_size = 3
            if distance_scale == 2:
                # 5
                # width = height = 6
                # ancilla width / height = 5
                pass
            if distance_scale == 3:
                # 7
                # width = height = 8
                # ancilla width / height = 7
                pass
            if distance_scale == 4:
                # 9
                # width = height = 10
                # ancilla width / height = 9
                pass
            else:
                partitions = None

        return stim_circuit, partitions

    def three_cnot(self, distance_scale: int = 1):
        """Generate three logical CNOTs with lattice surgery.

        Code adapted from: https://tqec.github.io/tqec/gallery/three_cnots.html

        :param distance_scale: _description_, defaults to 1
        :type distance_scale: int, optional
        :return: _description_
        :rtype: _type_
        """
        graph = three_cnots(Basis.X)
        compiled_graph = compile_block_graph(graph)
        stim_circuit = compiled_graph.generate_stim_circuit(k=distance_scale, manhattan_radius=2)

        return stim_to_qiskit(stim_circuit), stim_circuit

    def steane_encoding(self, distance_scale: int = 1):
        """Generate logical steane encoding

        Code adapted from: https://tqec.github.io/tqec/gallery/steane_encoding.html

        :param distance_scale: _description_, defaults to 1
        :type distance_scale: int, optional
        :return: _description_
        :rtype: _type_
        """
        graph = steane_encoding(Basis.X)
        compiled_graph = compile_block_graph(graph)
        stim_circuit = compiled_graph.generate_stim_circuit(k=distance_scale, manhattan_radius=2)

        return stim_to_qiskit(stim_circuit), stim_circuit






#########################################################################
# Functions adapted from https://github.com/gongaa/SlidingWindowDecoder
#########################################################################

class css_code(): # a refactored version of Roffe's package
    # do as less row echelon form calculation as possible.
    def __init__(self, hx=np.array([[]]), hz=np.array([[]]), code_distance=np.nan, name=None, name_prefix="", check_css=False):

        self.hx = hx # hx pcm
        self.hz = hz # hz pcm

        self.lx = np.array([[]]) # x logicals
        self.lz = np.array([[]]) # z logicals

        self.N = np.nan # block length
        self.K = np.nan # code dimension
        self.D = code_distance # do not take this as the real code distance
        # TODO: use QDistRnd to get the distance
        # the quantum code distance is the minimum weight of all the affine codes
        # each of which is a coset code of a non-trivial logical op + stabilizers
        self.L = np.nan # max column weight
        self.Q = np.nan # max row weight

        _, nx = self.hx.shape
        _, nz = self.hz.shape

        assert nx == nz, "hx and hz should have equal number of columns!"
        assert nx != 0,  "number of variable nodes should not be zero!"
        if check_css: # For performance reason, default to False
            assert not np.any(hx @ hz.T % 2), "CSS constraint not satisfied"
        
        self.N = nx
        self.hx_perp, self.rank_hx, self.pivot_hx = kernel(hx) # orthogonal complement
        self.hz_perp, self.rank_hz, self.pivot_hz = kernel(hz)
        self.hx_basis = self.hx[self.pivot_hx] # same as calling row_basis(self.hx)
        self.hz_basis = self.hz[self.pivot_hz] # but saves one row echelon calculation
        self.K = self.N - self.rank_hx - self.rank_hz

        self.compute_ldpc_params()
        self.compute_logicals()
        if code_distance is np.nan:
            dx = compute_code_distance(self.hx_perp, is_pcm=False, is_basis=True)
            dz = compute_code_distance(self.hz_perp, is_pcm=False, is_basis=True)
            self.D = np.min([dx,dz]) # this is the distance of stabilizers, not the distance of the code

        self.name = f"{name_prefix}_n{self.N}_k{self.K}" if name is None else name

    def compute_ldpc_params(self):

        #column weights
        hx_l = np.max(np.sum(self.hx, axis=0))
        hz_l = np.max(np.sum(self.hz, axis=0))
        self.L = np.max([hx_l, hz_l]).astype(int)

        #row weights
        hx_q = np.max(np.sum(self.hx, axis=1))
        hz_q = np.max(np.sum(self.hz, axis=1))
        self.Q = np.max([hx_q, hz_q]).astype(int)

    def compute_logicals(self):

        def compute_lz(ker_hx, im_hzT):
            # lz logical operators
            # lz\in ker{hx} AND \notin Im(hz.T)
            # in the below we row reduce to find vectors in kx that are not in the image of hz.T.
            log_stack = np.vstack([im_hzT, ker_hx])
            pivots = row_echelon(log_stack.T)[3]
            log_op_indices = [i for i in range(im_hzT.shape[0], log_stack.shape[0]) if i in pivots]
            log_ops = log_stack[log_op_indices]
            return log_ops

        self.lx = compute_lz(self.hz_perp, self.hx_basis)
        self.lz = compute_lz(self.hx_perp, self.hz_basis)

        return self.lx, self.lz

    def canonical_logicals(self):
        temp = inverse(self.lx @ self.lz.T % 2)
        self.lx = temp @ self.lx % 2

def create_circulant_matrix(l, pows):
    h = np.zeros((l,l), dtype=int)
    for i in range(l):
        for c in pows:
            h[(i+c)%l, i] = 1
    return h


def create_generalized_bicycle_codes(l, a, b, name=None):
    A = create_circulant_matrix(l, a)
    B = create_circulant_matrix(l, b)
    hx = np.hstack((A, B))
    hz = np.hstack((B.T, A.T))
    return css_code(hx, hz, name=name, name_prefix="GB")

def create_bivariate_bicycle_codes(l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows, name=None):
    S_l=create_circulant_matrix(l, [-1])
    S_m=create_circulant_matrix(m, [-1])
    x = kron(S_l, identity(m, dtype=int))
    y = kron(identity(l, dtype=int), S_m)
    A_list = [x**p for p in A_x_pows] + [y**p for p in A_y_pows]
    B_list = [y**p for p in B_y_pows] + [x**p for p in B_x_pows] 
    A = reduce(lambda x,y: x+y, A_list).toarray()
    B = reduce(lambda x,y: x+y, B_list).toarray()
    hx = np.hstack((A, B))
    hz = np.hstack((B.T, A.T))
    return css_code(hx, hz, name=name, name_prefix="BB", check_css=True), A_list, B_list

def build_circuit(code, A_list, B_list, num_repeat, z_basis=True, use_both=False, HZH=False): # Refactored to exclude errors
    n = code.N
    a1, a2, a3 = A_list
    b1, b2, b3 = B_list

    def nnz(m):
        a, b = m.nonzero()
        return b[np.argsort(a)]

    A1, A2, A3 = nnz(a1), nnz(a2), nnz(a3)
    B1, B2, B3 = nnz(b1), nnz(b2), nnz(b3)

    A1_T, A2_T, A3_T = nnz(a1.T), nnz(a2.T), nnz(a3.T)
    B1_T, B2_T, B3_T = nnz(b1.T), nnz(b2.T), nnz(b3.T)

    # |+> ancilla: 0 ~ n/2-1. Control in CNOTs.
    X_check_offset = 0
    # L data qubits: n/2 ~ n-1. 
    L_data_offset = n//2
    # R data qubits: n ~ 3n/2-1.
    R_data_offset = n
    # |0> ancilla: 3n/2 ~ 2n-1. Target in CNOTs.
    Z_check_offset = 3*n//2

    detector_circuit_str = ""
    for i in range(n//2):
        detector_circuit_str += f"DETECTOR rec[{-n//2+i}]\n"
    detector_circuit = stim.Circuit(detector_circuit_str)

    detector_repeat_circuit_str = ""
    for i in range(n//2):
        detector_repeat_circuit_str += f"DETECTOR rec[{-n//2+i}] rec[{-n-n//2+i}]\n"
    detector_repeat_circuit = stim.Circuit(detector_repeat_circuit_str)

    def append_blocks(circuit, repeat=False):
        # Round 1
        if repeat:        
            for i in range(n//2):
                # measurement preparation errors
                if HZH:
                    circuit.append("H", [X_check_offset + i])
        else:
            for i in range(n//2):
                circuit.append("H", [X_check_offset + i])

        for i in range(n//2):
            # CNOTs from R data to to Z-checks
            circuit.append("CNOT", [R_data_offset + A1_T[i], Z_check_offset + i])

        # tick
        circuit.append("TICK")

        # Round 2
        for i in range(n//2):
            # CNOTs from X-checks to L data
            circuit.append("CNOT", [X_check_offset + i, L_data_offset + A2[i]])
            # CNOTs from R data to Z-checks
            circuit.append("CNOT", [R_data_offset + A3_T[i], Z_check_offset + i])

        # tick
        circuit.append("TICK")

        # Round 3
        for i in range(n//2):
            # CNOTs from X-checks to R data
            circuit.append("CNOT", [X_check_offset + i, R_data_offset + B2[i]])
            # CNOTs from L data to Z-checks
            circuit.append("CNOT", [L_data_offset + B1_T[i], Z_check_offset + i])

        # tick
        circuit.append("TICK")

        # Round 4
        for i in range(n//2):
            # CNOTs from X-checks to R data
            circuit.append("CNOT", [X_check_offset + i, R_data_offset + B1[i]])
            # CNOTs from L data to Z-checks
            circuit.append("CNOT", [L_data_offset + B2_T[i], Z_check_offset + i])

        # tick
        circuit.append("TICK")

        # Round 5
        for i in range(n//2):
            # CNOTs from X-checks to R data
            circuit.append("CNOT", [X_check_offset + i, R_data_offset + B3[i]])
            # CNOTs from L data to Z-checks
            circuit.append("CNOT", [L_data_offset + B3_T[i], Z_check_offset + i])

        # tick
        circuit.append("TICK")

        # Round 6
        for i in range(n//2):
            # CNOTs from X-checks to L data
            circuit.append("CNOT", [X_check_offset + i, L_data_offset + A1[i]])
            # CNOTs from R data to Z-checks
            circuit.append("CNOT", [R_data_offset + A2_T[i], Z_check_offset + i])

        # tick
        circuit.append("TICK")

        # Round 7
        for i in range(n//2):
            # CNOTs from X-checks to L data
            circuit.append("CNOT", [X_check_offset + i, L_data_offset + A3[i]])
            # Measure Z-checks
            circuit.append("MR", [Z_check_offset + i])
            # identity gates on R data, moved to beginning of the round
        
        # Z check detectors
        if z_basis:
            if repeat:
                circuit += detector_repeat_circuit
            else:
                circuit += detector_circuit
        elif use_both and repeat:
            circuit += detector_repeat_circuit

        # tick
        circuit.append("TICK")
        
        # Round 8
        for i in range(n//2):
            if HZH:
                circuit.append("H", [X_check_offset + i])
                circuit.append("MR", [X_check_offset + i])
            else:
                circuit.append("MRX", [X_check_offset + i])
            # identity gates on L data, moved to beginning of the round
            # circuit.append("DEPOLARIZE1", L_data_offset + i, p_before_round_data_depolarization)
            
        # X basis detector
        if not z_basis:
            if repeat:
                circuit += detector_repeat_circuit
            else:
                circuit += detector_circuit
        elif use_both and repeat:
            circuit += detector_repeat_circuit
        
        # tick
        circuit.append("TICK")

   
    circuit = stim.Circuit()
    for i in range(n//2): # ancilla initialization
        circuit.append("R", X_check_offset + i)
        circuit.append("R", Z_check_offset + i)
    for i in range(n):
        circuit.append("R" if z_basis else "RX", L_data_offset + i)

    # begin round tick
    circuit.append("TICK") 
    append_blocks(circuit, repeat=False) # encoding round


    rep_circuit = stim.Circuit()
    append_blocks(rep_circuit, repeat=True)
    circuit += (num_repeat-1) * rep_circuit

    for i in range(0, n):
        # flip before collapsing data qubits
        # circuit.append("X_ERROR" if z_basis else "Z_ERROR", L_data_offset + i, p_before_measure_flip_probability)
        circuit.append("M" if z_basis else "MX", L_data_offset + i)
        
    pcm = code.hz if z_basis else code.hx
    logical_pcm = code.lz if z_basis else code.lx
    stab_detector_circuit_str = "" # stabilizers
    for i, s in enumerate(pcm):
        nnz = np.nonzero(s)[0]
        det_str = "DETECTOR"
        for ind in nnz:
            det_str += f" rec[{-n+ind}]"       
        det_str += f" rec[{-n-n+i}]" if z_basis else f" rec[{-n-n//2+i}]"
        det_str += "\n"
        stab_detector_circuit_str += det_str
    stab_detector_circuit = stim.Circuit(stab_detector_circuit_str)
    circuit += stab_detector_circuit
        
    log_detector_circuit_str = "" # logical operators
    for i, l in enumerate(logical_pcm):
        nnz = np.nonzero(l)[0]
        det_str = f"OBSERVABLE_INCLUDE({i})"
        for ind in nnz:
            det_str += f" rec[{-n+ind}]"        
        det_str += "\n"
        log_detector_circuit_str += det_str
    log_detector_circuit = stim.Circuit(log_detector_circuit_str)
    circuit += log_detector_circuit

    return circuit

    
#########################################################################

BBCODE_DICT = {
    # (n, k, d): [l, m, exp_A, exp_B]
    (54,8,6): [3,9,[0,2,4],[3,1,2]],
    (72,12,6): [6,6,[3,1,2],[3,1,2]],
    (90,8,10): [15,3,[9,1,2],[0,2,7]],
    (98,6,12): [7,7,[3,5,6],[2,3,5]],
    (108,8,10): [9,6,[3,1,2],[3,1,2]],
    (126,8,10): [3,21,[1,2,10],[3,1,2]],
    (144,12,12): [12,6,[3,1,2],[3,1,2]], #GrossCode
    (150,16,8): [5,15,[1,6,8],[5,1,4]],
    (162,8,14): [3,27,[0,10,14],[12,1,2]],
    (180,8,16): [6,15,[3,1,2],[6,4,5]],
    (288,12,18): [12,12,[3,2,7],[3,1,2]],
    (360,12,24): [20,6,[9,1,2],[3,25,26]],
    (756,16,34): [21,18,[3,10,17],[5,3,19]]
}

def get_gross_code(d=12, T=12):
    code, A_list, B_list = create_bivariate_bicycle_codes(12, 6, [3], [1,2], [1,2], [3])
    stim_circuit =  build_circuit(code, A_list, B_list, 
        num_repeat=T, # usually set to code distance
        z_basis=True,   # whether in the z-basis or x-basis
        use_both=False, # whether use measurement results in both basis to decode one basis
    )
    return stim_circuit #StimCodeCircuit(stim_circuit = stim_circuit)



def generate_gross_code(num_qubits: int = 1):
    """ Generate a gross code memory """

    stim_circuit = get_gross_code()


    partitions = [
                {
                    "indices": list(range(0, 288)),
                    "width": 12,
                    "height": 6,
                    "distance": 12,
                    "type": "gross_code",
                },
                
            ]

    with open(f"stim_gross.stim", "w") as f:
        print(stim_circuit, file=f)

    return stim_circuit, partitions