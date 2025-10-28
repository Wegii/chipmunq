import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))

import stim
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit
from glue.qiskit_qec.stim_tools import get_stim_circuits, get_stim_circuits_with_detectors

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
            CX 1 4

            ''')

stim_code = StimCodeCircuit(stim_circuit = stim_ex1)
#print(stim_ex1)
#print(stim_code)

stim_ex1_after_workflow = get_stim_circuits_with_detectors(stim_code.qc)[0][0]
#print("\n\nAfterwards: ")
#print(stim_ex1_after_workflow)

"""
# --- Define custom metadata instructions --- #
class DetectorInstruction(Instruction):
    def __init__(self, coords, rec_indices):
        params = {"coords": coords, "rec_indices": rec_indices}
        super().__init__("DETECTOR", 1, 0, [params])  # apply to all qubits for visualization


class ShiftCoordsInstruction(Instruction):
    def __init__(self, shift_vector):
        params = {"shift_vector": shift_vector}
        super().__init__("SHIFT_COORDS", 3, 0, [params])  # 3 qubits, no clbits


# --- Create circuit --- #
qreg = QuantumRegister(3)
creg = ClassicalRegister(3)
qc = QuantumCircuit(qreg, creg)

# 1. CNOT 0 2
qc.cx(0, 2)

# 2. CNOT 1 2
qc.cx(1, 2)

# 3. MR 2 (measure qubit 2)
qc.measure(2, 2)

# 4. DETECTOR(10.5, 0) rec[-1] rec[-2]
det_inst = DetectorInstruction(coords=(10.5, 0), rec_indices=[-1, -2])
qc_reg = qc.qregs[0]
for q in qc_reg:
    print(q)
    qc.append(det_inst, qargs=[q],)
#qc.append(det_inst, qargs=[qreg[1]])

# 5. SHIFT_COORDS(0, 1)
shift_inst = ShiftCoordsInstruction(shift_vector=(0, 1))
qc.append(shift_inst, qargs=[qreg[0], qreg[1], qreg[2]])

print(qc)
"""


