import os
import argparse
import numpy as np
import json
from StabCode import extract
from Stage1_Map_ver3 import S1_transpile
from Stage2_Schedule_ver3 import S2_transpile, toQASM
from qiskit import QuantumCircuit

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('Code', help='path to input stabilizer code file')
    parser.add_argument('Arch', help='path to input architecture file')
    parser.add_argument('-p', '--protocal', default='Flag-Bridge', help='name of measurment protocal')
    parser.add_argument('-L', '--Len', type=int, default=0, help='initial bridge size')
    parser.add_argument('-mL', '--maxL', type=int, default=None, help='maximum size for the ancilla block')
    parser.add_argument('-mT', '--maxT', type=int, default=None, help='upper bound for time step')
    parser.add_argument('-n', '--chunkNum', type=int, default=None, help='chunk num')
    parser.add_argument('-S1', '--runStage1', action='store_true', help='Run Stage 1')
    parser.add_argument('-S2', '--runStage2', action='store_true', help='Run Stage 2')

    args = parser.parse_args()

    codeName, _ = os.path.splitext(os.path.basename(args.Code))
    stabilizers, dataNum = extract(args.Code)
    code = (dataNum, stabilizers)

    # if args.chunkSize is None:
    #     chunkSize = len(stabilizers)
    # else:
    #     chunkSize = args.chunkSize

    if args.chunkNum is None:
        chunkNum = 1
    else:
        chunkNum = args.chunkNum

    archName, _ = os.path.splitext(os.path.basename(args.Arch))
    with open(args.Arch, 'r') as f:
        arch = json.load(f)
        CG = np.array(arch['CG'])
        idx2coord = arch['idx2coord']

    base = codeName + '-' + archName

    physNum = len(CG)
    print("\n# Physical Qubit: %d" % physNum)

    print('\n\nStage 1:')
    if args.runStage1 == True:
        print('Running')
        S1_transpile(code, CG, args.protocal, args.maxL, args.Len, chunkNum, 'aux_files/clause_'+base, 'aux_files/sol_'+base, 'output/S1_'+base)
    with open('output/S1_'+base+'.json', 'r') as f:
        result = json.load(f)

    physNum = len(CG)
    print("\n# Physical Qubit: %d" % physNum)

    print('\n\nStage 2:')
    if args.runStage2 == True:
        print('Running')
        S2_transpile(result['chunkNum'], result['Data'], result['Stab'], CG, idx2coord, result['Collison_list'], args.maxT, 'output/S2_'+base)

    with open('output/S2_'+base+'.json', 'r') as f:
        result = json.load(f)    

    print('\nResult:')
    print('\nData Mapping:')
    for data in range(dataNum):
        print('q[%s]' % data, end='\t')
        for k in range(result['chunkNum']):
            print('-> %d' % (result['Data'][k][data]), end='\t')
        print()

    cnotNum = 0

    print('\nSwap Layer:')
    for k in range(result['chunkNum']):
        # print('Chunk %d <-> Chunk %d:' % (k, (k+1)%result['chunkNum']))
        for s in result['Swap_layer'][k]:
            # print('SWAP(%d, %d)' % (s[1][1], s[1][2]), end='\t')
            cnotNum += 3
        print()

    print('\nBridge Allocating: Stabilizer -> Syndrome qubit -> Ancilla bridge')
    for k in range(result['chunkNum']):
        for k, stab in enumerate(result['Stab'][k]):
            print('%s -> %s -> %s' % (stab['name'], stab['synd'], set(stab['Ancilla'])))
            cnotNum += len(stab['Ancilla']) * 2 - 2

    physNum = len(CG)
    print("\n# Physical Qubit: %d" % physNum)
    print("Code Density: %.2f" % (2*(sum(len(stab['Ctrl']) for stab in stabilizers))/(dataNum+len(stabilizers))))
    print('# Extra Cnot Gates: %d' % cnotNum)

    QC = QuantumCircuit(physNum, physNum)
    for i in range(result['chunkNum']):
        QC = QC.compose(toQASM(len(CG), result['Circuit'][i]))
        QC = QC.compose(toQASM(len(CG), result['Swap_layer'][i]))

    print('Circuit Depth: %d' % QC.depth())
    print()