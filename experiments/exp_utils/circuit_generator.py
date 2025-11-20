from __future__ import annotations

import sys
import os
import logging
#sys.path.append(os.path.join(os.getcwd(), "glue/eccentric_bench/"))
#sys.path.append(os.path.join(os.getcwd(), "../eccentric_bench/external/qiskit_qec/src"))
#sys.path.append(os.path.join(os.getcwd(), "../eccentric_bench/external/qiskit_qec/"))

import qiskit
import numpy as np


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
from tqec.utils.position import FloatPosition3D, Position3D
from tqec.utils.scale import round_or_fail
from tqec.gallery import cnot, three_cnots, memory
from tqec.gallery.steane_encoding import steane_encoding
from experiments.exp_utils.circuit_utils import stim_to_qiskit

import stim

# Plotting
import matplotlib.pyplot as plt


class GenericCircuit():
    def __init__(self, nq: int):
        self.num_qubits = nq

    def generate_circuit(self, num_patches):
        
        # Generate GHZ circuit
        ghz = QuantumCircuit(self.num_qubits)
        # Apply H on qubit 0
        ghz.h(0)
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

    def single_cnot(self, distance_scale: int = 1):
        """Generate single logical CNOT with lattice surgery.

        Code adapted from: https://tqec.github.io/tqec/gallery/cnot.html

        :param distance_scale: Scale of surface code patch, defaults to 1
        :type distance_scale: int, optional
        """

        # TODO: add option for manhattan radius

        graph = cnot(Basis.X)
        compiled_graph = compile_block_graph(graph)
        stim_circuit = compiled_graph.generate_stim_circuit(
            k = distance_scale,
            manhattan_radius=2
        )

        return stim_to_qiskit(stim_circuit), stim_circuit

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
    