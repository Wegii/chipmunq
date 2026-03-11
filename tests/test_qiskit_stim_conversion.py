import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))

import stim
from experiments.exp_utils.circuit_noise import get_noise_model
# TQEC noise model
from tqec.utils.noise_model import NoiseModel

# Qiskit QEC
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit
from glue.qiskit_qec.stim_tools import get_stim_circuits_with_detectors

from experiments.exp_utils.circuit_generator import QECMemory
from experiments.exp_utils.transpilation_utils import *


def _output_to_file(content: str, filename: str = "") -> None:
    """Write a string to a file.

    If file exists, it will be overwritten.

    :param content: Content to write to file
    :type content: str
    :param filename: File to write to, defaults to ""
    :type filename: str, optional
    """

    if filename != "":
        with open(filename, "w") as f:
            print(content, file=f)
        

def test_simple_conversion():
    stim_ex1 = stim.Circuit('''
                QUBIT_COORDS(0, 0) 0
                QUBIT_COORDS(2, 0) 1
                QUBIT_COORDS(0, 2) 3
                QUBIT_COORDS(2, 2) 4           
                            
                H 0
                CX 0 1
                            
                TICK
                M 0 1
                DETECTOR(0, 0, 0) rec[-1] rec[-2]
                SHIFT_COORDS(0, 0)
                OBSERVABLE_INCLUDE(0) rec[-2]       
                TICK
                            
                CX 0 3
                SWAP 0 5
                CX 1 4
                ''')

    stim_code = StimCodeCircuit(stim_circuit = stim_ex1)

    stim_ex1_after_workflow = get_stim_circuits_with_detectors(stim_code.qc)[0][0]
    print(stim_ex1)
    print("\n\nAfterwards: ")
    print(stim_ex1_after_workflow)


def test_stim_conversion():
    """ Test conversion and transpilation of stim circuits
    
    Questions:
        - Is the transpilation correct (SWAP gates added)
        - How is the noise added to the circuit
        - Do all gates (single and two-qubit) receive the correct noise/error (stim noise gate)
    """


    # Define generic chiplet backend
    num_qubits = 100
    n_inter = 5
    small_backend = BackendChipletV2((2, 2, 10, 10), n_inter)

    circuit_generator = QECMemory(num_qubits)
    # Selects the maximum code distance given the number of qubits
    # Note: This is a stim circuit!
    surface_code_distance_3_circuit = circuit_generator.generate_code_memory('surface', 1, distance_scale = 1)
    _output_to_file(surface_code_distance_3_circuit, "tests/data/circuits/surface_code_distance_3")


    # Transpile using SABRE (we assume that the qiskit implementation is correct, as it should)
    # Stim to qiskit
    surface_code_distance_3_stimcircuit = StimCodeCircuit(stim_circuit = surface_code_distance_3_circuit)
    # Transpilation to backend
    surface_code_distance_3_transpiled_qiskit = sabre_transpilation(surface_code_distance_3_stimcircuit.qc, small_backend)
    # Qiskit to stim
    surface_code_distance_3_transpiled_stim = get_stim_circuits_with_detectors(surface_code_distance_3_transpiled_qiskit)[0][0]
    _output_to_file(surface_code_distance_3_transpiled_stim, "tests/data/circuits/surface_code_distance_3_transpiled")


    # Noise model: custom
    p = 1e-4
    noise_model = get_noise_model("constant", None, p, None)

    # Add noise to non-transpiled version of the stim code
    surface_code_distance_3_circuit_custom_noisy = noise_model.noisy_circuit(surface_code_distance_3_circuit)
    _output_to_file(surface_code_distance_3_circuit_custom_noisy,
                    "tests/data/circuits/surface_code_distance_3_custom_noisy")

    # Add noise to transpiled version of stim code
    surface_code_distance_3_circuit_custom_noisy_transpiled = noise_model.noisy_circuit(
        surface_code_distance_3_transpiled_stim)
    _output_to_file(surface_code_distance_3_circuit_custom_noisy_transpiled,
                    "tests/data/circuits/surface_code_distance_3_custom_noisy_transpiled")

    # Noise model: TQEC
    tqec_noise_model = NoiseModel.uniform_depolarizing

    # Add noise to non-transpiled version of the stim code
    #surface_code_distance_3_circuit_tqec_noisy = tqec_noise_model(p).noisy_circuit(surface_code_distance_3_circuit)
    #_output_to_file(surface_code_distance_3_circuit_tqec_noisy,
    #                "tests/data/circuits/surface_code_distance_3_tqec_noisy")
    
    # Add noise to transpiled version of stim code
    surface_code_distance_3_circuit_tqec_noisy_transpiled = tqec_noise_model(p).noisy_circuit(
        surface_code_distance_3_transpiled_stim)
    _output_to_file(surface_code_distance_3_circuit_tqec_noisy_transpiled,
                    "tests/data/circuits/surface_code_distance_3_tqec_noisy_transpiled")


if __name__ == "__main__":
    # test_simple_conversion()

    test_stim_conversion()

    # test_tqec_conversion()

