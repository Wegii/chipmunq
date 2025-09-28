import qiskit
from qiskit.qasm2 import dump
from StabCode import extract
import os

stab_path = "code/8_3_2Code.stab"

codeName, _ = os.path.splitext(os.path.basename(stab_path))
output_path = "circ/" + codeName +'.qasm'

Stab, dataNum = extract(stab_path)

p = 'Shor'

if p == "Shor":
    Data = qiskit.circuit.QuantumRegister(dataNum, 'data')
    QC = qiskit.QuantumCircuit(Data)
    for k, stab in enumerate(Stab):
        anc_name = 'anc%d' % k
        anc_size = len(stab['Ctrl'])
        Anc = qiskit.circuit.QuantumRegister(anc_size, 'anc%d' % k)
        QC.add_register(Anc)

        QC.reset(Anc)
        for i in range(anc_size):
            if i == 0: 
                QC.h(Anc[i])
            else:
                QC.cx(Anc[i-1], Anc[i])

        for i in range(anc_size):
            _, q_i, ctrl, _ = stab['Ctrl'][i]
            if ctrl[0] == "X":
                QC.cx(Anc[i], Data[q_i])
            elif ctrl[0] == "Z":
                QC.cz(Anc[i], Data[q_i])

        for i in reversed(range(anc_size)):
            if i == 0: 
                QC.h(Anc[i])
            else:
                QC.cx(Anc[i-1], Anc[i])
        QC.reset(Anc)

    with open(output_path, 'w') as f:
        dump(QC, f)

elif p == "Flag-Bridge":
    Data = qiskit.circuit.QuantumRegister(dataNum, 'data')
    Ancilla = qiskit.circuit.QuantumRegister(len(Stab), 'anc')
    # measure_bit = qiskit.circuit.ClassicalRegister(1, 'meas')
    QC = qiskit.QuantumCircuit(Data, Ancilla) # measure_bit)

    for k, stab in enumerate(Stab):
        synd_qubit = Ancilla[k]
        QC.reset(synd_qubit)
        QC.h(synd_qubit)

        for _, q_i, ctrl, _ in stab['Ctrl']:
            if ctrl[0] == "X":
                QC.cx(synd_qubit, Data[q_i])
            elif ctrl[0] == "Z":
                QC.cz(synd_qubit, Data[q_i])

        QC.h(synd_qubit)
        QC.reset(synd_qubit)
        # QC.measure(synd_qubit, measure_bit[0])

    with open(output_path, 'w') as f:
        dump(QC, f)
