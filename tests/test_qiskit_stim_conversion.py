import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))

import stim
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit
from glue.qiskit_qec.stim_tools import get_stim_circuits_with_detectors


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


def test_tqec_conversion():
    # Test conversion of a lattice surgery circuit
    #     
    pass


if __name__ == "__main__":
    test_simple_conversion()