import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))

import stim
# TQEC noise model
from tqec.utils.noise_model import NoiseModel


def check_tqec_noise_model():
    """ Test the noise model from tqec.
    
    Check what kind of constraints and what structure a stim circuit must adhere to.
    """

    # Circuit that should fail, since one instruction uses some qubits multiple times
    stim_incorrect_circuit = stim.Circuit('''
            SWAP 14 15 13 14 12 13 11 12 10 11 15 16 14 15 13 14
            ''')
    
    stim_correct_circuit = stim.Circuit('''
            SWAP 14 15
            TICK
            SWAP 13 14
            TICK
            SWAP 12 13
            TICK
            SWAP 11 12
            TICK
            SWAP 10 11
            TICK
            SWAP 15 16
            TICK
            SWAP 14 15
            TICK
            SWAP 13 14
            ''')
    
    p = 1e-4
    tqec_noise_model = NoiseModel.uniform_depolarizing

    # This should throw an error
    try:
        _ = tqec_noise_model(p).noisy_circuit(stim_incorrect_circuit)
        print("Noisy circuit constructed successfully")
    except ValueError: print("Operation collisions")

    try:
        _ = tqec_noise_model(p).noisy_circuit(stim_correct_circuit)
        print("Noisy circuit constructed successfully")
    except ValueError as e: print(e)


if __name__ == "__main__":
    check_tqec_noise_model()