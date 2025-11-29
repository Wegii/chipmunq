from __future__ import annotations

import sys
import os
import logging
#sys.path.append(os.path.join(os.getcwd(), "glue/eccentric_bench/"))
#sys.path.append(os.path.join(os.getcwd(), "../eccentric_bench/external/qiskit_qec/src"))
#sys.path.append(os.path.join(os.getcwd(), "../eccentric_bench/external/qiskit_qec/"))

import qiskit
import numpy as np

from itertools import product

from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit


# QECCircuit
import random
import pyzx as zx
zx.settings.colors = zx.rgb_colors

import qiskit.qasm2 as qasm2
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

import sinter
from tqec import compile_block_graph, NoiseModel

from tqec.utils.enums import Basis
from tqec.computation.block_graph import BlockGraph, BlockKind, block_kind_from_str
from tqec.computation.cube import CubeKind, Port, YHalfCube
from tqec.computation.pipe import PipeKind
from tqec.computation.cube import ZXCube
from tqec.utils.position import FloatPosition3D, Position3D
from tqec.utils.scale import round_or_fail
from tqec.gallery import cnot, three_cnots, memory, stability, cz
from tqec.gallery.steane_encoding import steane_encoding
from tqec.utils.position import Direction3D, Position3D, SignedDirection3D
from experiments.exp_utils.circuit_utils import stim_to_qiskit

import stim

# Plotting
import matplotlib.pyplot as plt


def get_tqec_cnot_rotated(distance_scale: int = 2, n1: int = 1, n2: int = 0) -> tuple[StimCodeCircuit, list]:
    circuit_generator = QECCircuit()
    stim_circuit, partitions = circuit_generator.single_cnot_full_memory(distance_scale = distance_scale,
                                                                         n1 = n1,
                                                                         n2 = n2)

    with open("stim_circuit_cnot_multiple.stim", "w") as f:
        print(stim_circuit, file=f)
  
    return stim_circuit, partitions



class GenericCircuit():
    def __init__(self, nq: int):
        self.num_qubits = nq

    def generate_circuit(self, num_patches):
        
        # Generate GHZ circuit
        ghz = QuantumCircuit(self.num_qubits)
        # Apply H on qubit 0
        #ghz.h(0)
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
            patched_circuit.append(ghz.to_instruction(),
                    qargs=list(range(offset, offset + self.num_qubits)))

        return patched_circuit

class QECMemory():
    """QECC memory circuits
    
    Supported codes:
        - Color Code
        - Surface Code
        - Steane Code
        - BB Codes: have a look at https://github.com/AndersenQubitLab/small_qLDPC_codes
    """

    # QECC Memory circuits
    def __init__(self, nq: int, gate_set: list = []):
        self.num_qubits = nq

        self.gate_set = gate_set 

    def draw_circuit(self, circuit: qiskit.QuantumCircuit, filename: str = "") -> None:
        renderer = "mpl"

        if filename == "":
            circuit.draw(output = renderer)
        else:
            circuit.draw(output = renderer, filename = filename)

    def _check_circuit_gateset(self, circuit) -> qiskit.QuantumCircuit:

        # qiskit transpilation step with the gateset
        self.gate_set
        return circuit

    def _generate_code_from_eccentric_bench(self, codename: str, distance_scale: int = -1) -> StimCodeCircuit:
        """Generate QECC memory circuit using eccentric_bench library

        :param codename: Name of QEC code to generate
        :type codename: str
        :return: QECC memory circuit
        :rtype: qiskit.QuantumCircuit
        """
        if distance_scale == -1:
            d = get_max_d("surface", self.num_qubits)
        else:
            d = 2*distance_scale + 1

        if d < 3:
            logging.error(
                f"Code distance too small! {codename} with distance {d} and {self.num_qubits} qubits: Execution not possible")
            exit(1)

        # Generate code
        cycles = d

        if codename == "surface":
            stim_circuit = stim.Circuit.generated(
                "surface_code:unrotated_memory_z",
                #"surface_code:rotated_memory_z",
                rounds=cycles,
                distance=d
                )
        if codename == "rotated_surface":
            stim_circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_x",
                rounds=cycles,
                distance=d
                )
        else:
            stim_circuit = None
        return stim_circuit
    
    def _generate_distributed_code_from_eccentric_bench(self, codename: str) -> qiskit.QuantumCircuit:
        # Simply generate multiple patches of the same code, and concatenate the generated code circuits

        qecc_mem = self._generate_code_from_eccentric_bench(codename)

        # Generate code

        # Stitch circuits together

        # TODO: This is more complicated, since it is necessary to modify the detectors and logicals.

        return qecc_mem

    def generate_code_memory(self, codename: str, patches: int = 0, distance_scale: int = 1) -> StimCodeCircuit:
        """Generate QECC memory circuit

        Currently used as wrapper around the circuit generation function from eccentric_bench. Extend this function if
        some unsupported QEC code should be generated.

        :param codename: Name of QEC code to generate
        :type codename: str
        :return: QECC memory circuit
        :rtype: qiskit.QuantumCircuit
        """

        qecc_mem = self._generate_code_from_eccentric_bench(codename, distance_scale = distance_scale)
        

        #TODO: Check if circuit only consists of allowed gates, i. e. it is important that the circuit only consists of
        # e. g. two qubit gates as this will influence the hypergraph construction
        # Note: The allowed gateset also needs to be taken into account in the local routing algorithm.
        qecc_mem_transpiled = qecc_mem #self._check_circuit_gateset(qecc_mem)

        return qecc_mem_transpiled


class QECCircuit:
    """ Logical circuits using lattice surgery
    
    
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
        stim_circuit = compiled_graph.generate_stim_circuit(
            k = distance_scale,
            manhattan_radius=3
        )

        return stim_to_qiskit(stim_circuit), stim_circuit

    def complete_memory_patch(self, distance_scale: int = 1):
        # General issue: when instantiating a memory block, not all qubits are initialized.
        # TODO: Find a way how to initialize all qubits, even if not all of them are used
        pass
    
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

                pipes = [(0, 1), (1, 2)]#, (2, 3), (3, 4)]
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

                pipes = [(0, 1), (1, 2)]#, (2, 3), (3, 4)]
                for p0, p1 in pipes:
                    g.add_pipe(nodes[p0][0], nodes[p1][0])

                g.fill_ports({"In": ZXCube.from_str("ZXZ"), "Out": ZXCube.from_str("ZXZ")})


        compiled_graph = compile_block_graph(g)
        stim_circuit = compiled_graph.generate_stim_circuit(
            k = distance_scale,
            manhattan_radius=3
        )

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
        g.add_pipe(h_cube, out_cube, PipeKind(
            Basis.X,
            Basis.Z,
            None,
            has_hadamard=True,
        ))  # TQEC infers default pipe for identity

        # Fill ports
        g.fill_ports({
            "In": ZXCube.from_str("XZZ"),
            "Out": ZXCube.from_str("ZXX"),
        })

        compiled_graph = compile_block_graph(g)
        stim_circuit = compiled_graph.generate_stim_circuit(
            k = distance_scale,
            manhattan_radius=3
        )

        # TODO: This should also return the qubits of all patches
        if distance_scale == 2:
            memory_d5 = [[
                0, 1, 2, 3, 4, 5,
                6, 7, 8, 9, 10,
                11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21,
                22, 23, 24, 25, 26, 27,
                28, 29, 30, 31, 32,
                33, 34, 35, 36, 37, 38,
                39, 40, 41, 42, 43,
                44, 45, 46, 47, 48, 49,
                50, 51, 52, 53, 54,
                55, 56, 57, 58, 59, 60
            ]]
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

        #graph = cz(["XI -> XZ", "IZ -> IZ"])#.rotate(Direction3D.Z, )
        graph = cnot(Basis.X)
        compiled_graph = compile_block_graph(graph)
        stim_circuit = compiled_graph.generate_stim_circuit(
            k = distance_scale,
            manhattan_radius=2
        )

        return stim_to_qiskit(stim_circuit), stim_circuit
    
    def single_cnot_full_memory(self, distance_scale: int = 1, n1: int = 1, n2: int = 0):

        if n1 == 1 and n2 == 0:
            # Construct a single CNOT
            g = cnot(Basis.Z)

            compiled_graph = compile_block_graph(g)
            stim_circuit = compiled_graph.generate_stim_circuit(
                k = distance_scale,
                manhattan_radius=2
            )

            if distance_scale == 2:
                partitions = [
                    # Control patch
                    {
                        "indices": [
                            0, 1, 2, 3, 4, 5,
                            12, 13, 14, 15, 16,
                            23, 24, 25, 26, 27, 28,
                            35, 36, 37, 38, 39,
                            46, 47, 48, 49, 50, 51,
                            58, 59, 60, 61, 62,
                            69, 70, 71, 72, 73, 74,
                            81, 82, 83, 84, 85,
                            92, 93, 94, 95, 96, 97,
                            104, 105, 106, 107, 108,
                            115, 116, 117, 118, 119, 120
                        ],
                        "width": 6,
                        "height": 11,
                        "distance": 5,
                        "type": "rotated_surface_code"
                    },
                    # Ancilla Patch
                    {
                        "indices": [
                            6, 7, 8, 9, 10, 11,
                            18, 19, 20, 21, 22,
                            29, 30, 31, 32, 33, 34,
                            41, 42, 43, 44, 45,
                            52, 53, 54, 55, 56, 57,
                            64, 65, 66, 67, 68,
                            75, 76, 77, 78, 79, 80,
                            87, 88, 89, 90, 91,
                            98, 99, 100, 101, 102, 103,
                            110, 111, 112, 113, 114,
                            121, 122, 123, 124, 125, 126
                        ],
                        "width": 6,
                        "height": 11,
                        "distance": 5,
                        "type": "rotated_surface_code"
                    },
                    # Target Patch
                    {
                        "indices": [
                            132, 133, 134, 135, 136, 137,
                            138, 139, 140, 141, 142,
                            143, 144, 145, 146, 147, 148,
                            149, 150, 151, 152, 153,
                            154, 155, 156, 157, 158, 159,
                            160, 161, 162, 163, 164,
                            165, 166, 167, 168, 169, 170,
                            171, 172, 173, 174, 175,
                            176, 177, 178, 179, 180, 181,
                            182, 183, 184, 185, 186,
                            187, 188, 189, 190, 191, 192
                        ],
                        "width": 6,
                        "height": 11,
                        "distance": 5,
                        "type": "rotated_surface_code"
                    },
                    # CA_Patch
                    {
                        "indices": [17, 40, 63, 86, 109],
                        "width": 6,
                        "height": 1,
                        "distance": 5,
                        "type": "rotated_surface_code_ancilla"
                    },
                    # AT_Patch
                    {
                        "indices": [127, 128, 129, 130, 131],
                        "width": 1,
                        "height": 6,
                        "distance": 5,
                        "type": "rotated_surface_code_ancilla"
                    }
                ]
            elif distance_scale == 3:
                # TODO: implement distance 7
                pass
            elif distance_scale == 4:
                # TODO: implement distance 9
                pass
        else:
            print("Generating multiple single_cnot")

            # Contains all patches and operations
            g = BlockGraph("Logical CNOT")

            n1 = 3
            n2 = 3

            placement_x = 0
            placement_y = 0
            cnot_counter = 0
            # Used to determine if the cnot is placed downwards+right, or right+downwards, in order to completely
            # fill the grid optimally
            rotation_counter = 0
            for x1 in range(n1):
                placement_y = 0

                for y1 in range(n2):
                    if rotation_counter%2 == 0:
                        # Place downwards+right
                        nodes = [
                            (Position3D(placement_x, placement_y, 0), "P", f"In_Control_{cnot_counter}"),
                            (Position3D(placement_x, placement_y, 1), "ZXX", ""),
                            (Position3D(placement_x, placement_y, 2), "ZXZ", ""),
                            (Position3D(placement_x, placement_y, 3), "P", f"Out_Control_{cnot_counter}"),
                            (Position3D(placement_x, placement_y+1, 1), "ZXX", ""),
                            (Position3D(placement_x, placement_y+1, 2), "ZXZ", ""),
                            (Position3D(placement_x+1, placement_y+1, 0), "P", f"In_Target_{cnot_counter}"),
                            (Position3D(placement_x+1, placement_y+1, 1), "ZXZ", ""),
                            (Position3D(placement_x+1, placement_y+1, 2), "ZXZ", ""),
                            (Position3D(placement_x+1, placement_y+1, 3), "P", f"Out_Target_{cnot_counter}"),
                        ]
                        for pos, kind, label in nodes:
                            g.add_cube(pos, kind, label)

                        pipes = [(0, 1), (1, 2), (2, 3), # Control
                                (1, 4), (4, 5), # Ancilla
                                (5, 8), # Merge
                                (6, 7), (7, 8), (8, 9) # Target
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
                            (Position3D(placement_x+1, placement_y, 1), "ZXZ", ""),
                            (Position3D(placement_x+1, placement_y, 2), "ZXX", ""),
                            (Position3D(placement_x+1, placement_y+1, 0), "P", f"In_Target_{cnot_counter}"),
                            (Position3D(placement_x+1, placement_y+1, 1), "ZXZ", ""),
                            (Position3D(placement_x+1, placement_y+1, 2), "ZXX", ""),
                            (Position3D(placement_x+1, placement_y+1, 3), "P", f"Out_Target_{cnot_counter}"),
                        ]

                        for pos, kind, label in nodes_2:
                            g.add_cube(pos, kind, label)

                        pipes_2 = [(0, 1), (1, 2), (2, 3),
                                    (1, 4), (4, 5),
                                    (5, 8),
                                    (6, 7), (7, 8), (8, 9)
                                    ]

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
            stim_circuit = compiled_graph.generate_stim_circuit(
                k = distance_scale,
                manhattan_radius=2
            )
            
            # Compute partitions
            # TODO: This needs to be fixed for multiple cnots
            if distance_scale == 2:
                partitions = [
                    # Control patch
                    {
                        "indices": [
                            0, 1, 2, 3, 4, 5,
                            12, 13, 14, 15, 16,
                            23, 24, 25, 26, 27, 28,
                            35, 36, 37, 38, 39,
                            46, 47, 48, 49, 50, 51,
                            58, 59, 60, 61, 62,
                            69, 70, 71, 72, 73, 74,
                            81, 82, 83, 84, 85,
                            92, 93, 94, 95, 96, 97,
                            104, 105, 106, 107, 108,
                            115, 116, 117, 118, 119, 120
                        ],
                        "width": 6,
                        "height": 11,
                        "distance": 5,
                        "type": "rotated_surface_code"
                    },
                    # Ancilla Patch
                    {
                        "indices": [
                            6, 7, 8, 9, 10, 11,
                            18, 19, 20, 21, 22,
                            29, 30, 31, 32, 33, 34,
                            41, 42, 43, 44, 45,
                            52, 53, 54, 55, 56, 57,
                            64, 65, 66, 67, 68,
                            75, 76, 77, 78, 79, 80,
                            87, 88, 89, 90, 91,
                            98, 99, 100, 101, 102, 103,
                            110, 111, 112, 113, 114,
                            121, 122, 123, 124, 125, 126
                        ],
                        "width": 6,
                        "height": 11,
                        "distance": 5,
                        "type": "rotated_surface_code"
                    },
                    # Target Patch
                    {
                        "indices": [
                            132, 133, 134, 135, 136, 137,
                            138, 139, 140, 141, 142,
                            143, 144, 145, 146, 147, 148,
                            149, 150, 151, 152, 153,
                            154, 155, 156, 157, 158, 159,
                            160, 161, 162, 163, 164,
                            165, 166, 167, 168, 169, 170,
                            171, 172, 173, 174, 175,
                            176, 177, 178, 179, 180, 181,
                            182, 183, 184, 185, 186,
                            187, 188, 189, 190, 191, 192
                        ],
                        "width": 6,
                        "height": 11,
                        "distance": 5,
                        "type": "rotated_surface_code"
                    },
                    # CA_Patch
                    {
                        "indices": [17, 40, 63, 86, 109],
                        "width": 6,
                        "height": 1,
                        "distance": 5,
                        "type": "rotated_surface_code_ancilla"
                    },
                    # AT_Patch
                    {
                        "indices": [127, 128, 129, 130, 131],
                        "width": 1,
                        "height": 6,
                        "distance": 5,
                        "type": "rotated_surface_code_ancilla"
                    }
                ]
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
        stim_circuit = compiled_graph.generate_stim_circuit(
            k = distance_scale,
            manhattan_radius=2
        )

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
        stim_circuit = compiled_graph.generate_stim_circuit(
            k = distance_scale,
            manhattan_radius=2
        )

        return stim_to_qiskit(stim_circuit), stim_circuit

    def simple_circuit(self):
        from topologiq.scripts.runner import runner
        from topologiq.utils.interop_pyzx import pyzx_g_to_simple_g
        from topologiq.utils.utils_zx_graphs import kind_to_zx_type

        def steane_circuit_qiskit():
            """Function to generate the Steane code encoding circuit. """

            """
            qc = QuantumCircuit(10)

            qc.h(0)
            qc.cx(0, 3)
            qc.cx(0, 4)
            qc.cx(0, 5)
            qc.cx(0, 6)
            qc.h(0)

            qc.h(1)
            qc.cx(1, 3)
            qc.cx(1, 4)
            qc.cx(1, 7)
            qc.cx(1, 8)
            qc.h(1)

            qc.h(2)
            qc.cx(2, 3)
            qc.cx(2, 5)
            qc.cx(2, 7)
            qc.cx(2, 9)
            qc.h(2)
            """
            qc = QuantumCircuit(2)

            qc.h(0)
            qc.h(1)
            

            return qc

        # 1. Qiskit
        # Generate circuit
        base_circuit = steane_circuit_qiskit()
        #print(base_circuit)

        # Convert circuit to qasm
        qasm_str = qasm2.dumps(base_circuit)

        # 2. PyZX
        # Qasm to pyZX. WHY????
        zx_circuit = zx.Circuit.from_qasm(qasm_str)
        zx_graph = zx_circuit.to_graph()
        zx.draw(zx_graph, labels = True)
        # Optimize
        # Apply states
        num_apply_state = zx_graph.num_inputs()
        zx_graph.apply_state('0' * num_apply_state)
        # Apply post-select only to the outputs of the ancilla qubits
        zx_graph.apply_effect('000///////')
        #zx_graph.apply_effect('///')
        #zx.draw(zx_graph, labels = True)
        zx.full_reduce(zx_graph)
        zx.to_rg(zx_graph)

        random.seed(12)
        #zx.draw(zx_graph, labels = True, auto_layout=True)
        fig_data = zx.draw_matplotlib(zx_graph, labels=True)
        plt.savefig("data/circuits/QECCircuit/zx.png")


        # 3. topologiq
        simple_graph = pyzx_g_to_simple_g(zx_graph)

        for k, v in simple_graph.items():
                print(f"{k}: {v}")
        
        # Parameters & hyper-parameters
        circuit_name = f"steane_from_qiskit"
        visualisation = "final"  # Calls 3D visualisation at the end. `None` to deactivate.
        animation = None  # Change to "GIF" or "MP4" for a summary animation (significant runtime costs).

        VALUE_FUNCTION_HYPERPARAMS = (
            -1,  # Weight for lenght of path
            -1,  # Weight for number of "beams" broken by path
        )

        kwargs = {
            "weights": VALUE_FUNCTION_HYPERPARAMS,
            "length_of_beams": 9,
        }

        # Run topologiq
        simple_graph_after_use, edge_pths, lattice_nodes, lattice_edges = runner(
            simple_graph,  # The PyZX input graph, simplified
            circuit_name,  # The name of the circuit
            visualise=(visualisation, animation),
            fig_data=fig_data,
            **kwargs
        )

        # Print output
        if lattice_nodes and lattice_edges:
            print("\nCubes in final output:")
            for k, v in lattice_nodes.items():
                print(f"{k}:{v}")

            print("\nPipes in final output:")
            for k, v in lattice_edges.items():
                print(f"{k}:{v}")

        else:
            print("WARNING! Some key objects needed to create the block_graph do not exist. Please check that topologiq ran and succeeded.")
    


        # 4. tQEC
        def _int_position_before_scale(pos: FloatPosition3D, pipe_length: float) -> Position3D:
            return Position3D(
                x=round_or_fail(pos.x / (1 + pipe_length), atol=0.35),
                y=round_or_fail(pos.y / (1 + pipe_length), atol=0.35),
                z=round_or_fail(pos.z / (1 + pipe_length), atol=0.35),
            )


        def _offset_y_cube_position(pos: FloatPosition3D, pipe_length: float) -> FloatPosition3D:
            if np.isclose(pos.z - 0.5, np.floor(pos.z), atol=1e-9):
                pos = pos.shift_by(dz=-0.5)
            return FloatPosition3D(pos.x, pos.y, pos.z / (1 + pipe_length))



        if lattice_nodes and lattice_edges:

            pipe_length: float | None = None
            parsed_ports: list[FloatPosition3D] = []
            parsed_cubes: list[tuple[FloatPosition3D, CubeKind]] = []
            parsed_pipes: list[tuple[FloatPosition3D, PipeKind, int]] = []

            for v in lattice_nodes.values():
                coords = v[0]
                translation = FloatPosition3D(*coords)

                if v[1] != "ooo":
                    # NB! Need to figure out what to do with "ooo" blocks, which is what topologiq uses to denote a port.
                    kind = block_kind_from_str(v[1].upper())

                    if isinstance(kind, CubeKind):
                        parsed_cubes.append((translation, kind))

                else:
                    parsed_ports.append(translation)

            for (src, tgt), v in lattice_edges.items():
                kind = block_kind_from_str(v[0].upper())
                if isinstance(kind, PipeKind):
                    src_pos = lattice_nodes[src][0]
                    tgt_pos = lattice_nodes[tgt][0]

                    shift_coords_from_src = tuple([(u-v)/3 for u, v in zip(tgt_pos, src_pos)])
                    directional_multiplier = int(sum(shift_coords_from_src))

                    coords = [u+v for u, v in zip(src_pos, shift_coords_from_src)]
                    translation = FloatPosition3D(*coords)

                    parsed_pipes.append((translation, kind, directional_multiplier))

            # Construct graph
            # Create graph
            block_graph = BlockGraph(circuit_name)
            pipe_length = 2.0

            # Add cubes
            for pos, cube_kind in parsed_cubes:
                if isinstance(cube_kind, YHalfCube):
                    block_graph.add_cube(
                        _int_position_before_scale(_offset_y_cube_position(pos, pipe_length), pipe_length),
                        cube_kind,
                    )
                else:
                    block_graph.add_cube(_int_position_before_scale(pos, pipe_length), cube_kind)
            port_index = 0

            # Add pipes
            for pos, pipe_kind, directional_multiplier in parsed_pipes:
                head_pos = _int_position_before_scale(
                    pos.shift_in_direction(pipe_kind.direction, -1 * directional_multiplier),
                    pipe_length,
                )
                tail_pos = head_pos.shift_in_direction(pipe_kind.direction, 1 * directional_multiplier)

                if head_pos not in block_graph:
                    block_graph.add_cube(head_pos, Port(), label=f"Port{port_index}")
                    port_index += 1
                if tail_pos not in block_graph:
                    block_graph.add_cube(tail_pos, Port(), label=f"Port{port_index}")
                    port_index += 1
                block_graph.add_pipe(head_pos, tail_pos, pipe_kind)

            html = block_graph.view_as_html()
            with open(f"data/circuits/QECCircuit/{circuit_name}_open_ports.html","w") as f:
                f.write(str(html))
                f.close()

            filled_block_graphs = block_graph.fill_ports_for_minimal_simulation()
            for i, block_graph in enumerate(filled_block_graphs):

                for j, correlation_surface in enumerate(block_graph.observables):
                    html = block_graph.graph.view_as_html(
                        pop_faces_at_directions=("-Y", "+X"),
                        show_correlation_surface=block_graph.observables[j],
                    )

                    with open(f"data/circuits/QECCircuit/{circuit_name}_filled_{i}-{j}.html","w") as f:
                        f.write(str(html))
                        f.close()

         

        def graph_for_given_basis(observable_basis: Basis) -> BlockGraph | None:

            filled_graphs = filled_block_graphs
            assert len(filled_graphs) == 2
            if observable_basis == Basis.X:
                return filled_graphs[0].graph
            elif observable_basis == Basis.Z:
                return filled_graphs[1].graph


        def get_stim_circuit():

            block_graph_for_computation = graph_for_given_basis(Basis.X)
            if block_graph_for_computation:
                compiled_graph = compile_block_graph(block_graph_for_computation)
                stim_circuit = compiled_graph.generate_stim_circuit(
                    k=1, noise_model=NoiseModel.uniform_depolarizing(p=0.001)
                )
                print("\nBasis X:\n")
                basis_x_circuit = stim_circuit
                #print(stim_circuit)

            block_graph_for_computation = graph_for_given_basis(Basis.Z)
            if block_graph_for_computation:
                compiled_graph = compile_block_graph(block_graph_for_computation)
                stim_circuit = compiled_graph.generate_stim_circuit(
                    k=1, noise_model=NoiseModel.uniform_depolarizing(p=0.001)
                )
                print("\nBasis Z:\n")
                #print(stim_circuit)
                basis_z_circuit = stim_circuit

            return basis_x_circuit, basis_z_circuit

        basis_x_circuit, basis_z_circuit = get_stim_circuit()
        with open(f"data/circuits/QECCircuit/{circuit_name}_x.stim","w") as f:
            basis_x_circuit.to_file(f)
        
        with open(f"data/circuits/QECCircuit/{circuit_name}_z.stim","w") as f:
            basis_z_circuit.to_file(f)

        # Convert stim circuit to qiskit circuit
        return stim_to_qiskit(basis_x_circuit)
    