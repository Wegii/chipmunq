import re
import argparse
import random
import numpy as np


def extract(fname):
    '''
        Input parsing
    '''
    Stab = []
    stabNo = 0
    dataNo = 0
    # ctrlNo = 0

    with open(fname) as f:
        qreg_line = f.readline()
        dataNum = int(re.match(r'qreg q\[(\d+)\]', qreg_line).group(1))
        for line in f:
            name = re.match(r'stabilizer\s+(\w+).*', line)
            if name:
                match = re.findall(r'\(([XYZ]), q\[(\d+)\], (\d+)\)', line)
                if match: 
                    S_k = {'name':name.group(1), 'Data':[], 'Ancilla':[], 'Ctrl':[]}
                    for Pauli, dataNo, t in match:
                        dataNo = int(dataNo)
                        t = int(t)
                        S_k['Data'].append(dataNo)
                        # CP gate: [stabNo, dataNo, [P, ctrl, tgt], t]
                        S_k['Ctrl'].append([stabNo, dataNo, [Pauli, None, None], t])
                    Stab.append(S_k)
                    stabNo += 1
                else:
                    match = re.findall(r'\(([XYZ]), q\[(\d+)\]\)', line)
                    S_k = {'name':name.group(1), 'Data':[], 'Ancilla':[], 'Ctrl':[]}
                    for Pauli, dataNo in match:
                        dataNo = int(dataNo)
                        S_k['Data'].append(dataNo)

                        # CP gate: [stabNo, dataNo, [P, ctrl, tgt], t]
                        S_k['Ctrl'].append([stabNo, dataNo, [Pauli, None, None], None])

                    Stab.append(S_k)
                    stabNo += 1
    random.seed(42)
    random.shuffle(Stab)
    return Stab, dataNum


def surface_code(fname, d=3):
    def coord2index(i, j):        
        return (d * i + j)

    with open(fname, "w") as f:
        StabNum = 0
        f.write('qreg q[%d]\n' % (d * d))

        for i in range(d - 1):
            for j in range(d - 1):
                StabNum += 1
                f.write('stabilizer S%d ' % StabNum)
                
                top = coord2index(i, j)
                left = coord2index(i + 1, j)
                right = coord2index(i, j + 1)
                down = coord2index(i + 1, j + 1)

                if (i + j) % 2:
                    for k, idx in enumerate([top, right, left, down]):
                        f.write('(X, q[{idx}], {k})'.format(idx=idx, k=k))
                        if k == 3:
                            f.write('\n')
                        else:
                            f.write(', ')
                else:
                    for k, idx in enumerate([top, left, right, down]):
                        f.write('(Z, q[{idx}], {k})'.format(idx=idx, k=k))
                        if k == 3:
                            f.write('\n')
                        else:
                            f.write(', ')
        
        for j in range(d // 2):
            StabNum += 1
            f.write('stabilizer S%d ' % StabNum)
            f.write('(X, q[{i0}], 2), (X, q[{i1}], 3)\n'.format(i0 = coord2index(0, j*2), i1 = coord2index(0, j*2+1)))

            StabNum += 1
            f.write('stabilizer S%d ' % StabNum)
            if d % 2 == 1:
                f.write('(X, q[{i0}], 0), (X, q[{i1}], 1)\n'.format(i0 = coord2index(d-1,  j*2+1), i1 = coord2index(d-1, j*2+2)))
            else:
                f.write('(X, q[{i0}], 0), (X, q[{i1}], 1)\n'.format(i0 = coord2index(d-1,  j*2), i1 = coord2index(d-1, j*2+1)))
        
        for i in range((d - 1) // 2):
            StabNum += 1
            f.write('stabilizer S%d ' % StabNum)
            f.write('(Z, q[{i0}], 2), (Z, q[{i1}], 3)\n'.format(i0 = coord2index(i*2+1, 0), i1 = coord2index(i*2+2, 0)))

            StabNum += 1
            f.write('stabilizer S%d ' % StabNum)
            if d % 2 == 1:
                f.write('(Z, q[{i0}], 0), (Z, q[{i1}], 1)\n'.format(i0 = coord2index(i*2, d-1), i1 = coord2index(i*2+1, d-1)))
            else:
                f.write('(Z, q[{i0}], 0), (Z, q[{i1}], 1)\n'.format(i0 = coord2index(i*2+1, d-1), i1 = coord2index(i*2+2, d-1)))


def steane_code(fname):
    groups = [[0, 1, 2, 3], [0, 1, 4, 5], [0, 2, 4, 6]] 
    with open(fname, "w") as f:
        StabNum = 0
        f.write('qreg q[%d]\n' % (7))

        for i in range(3):
            StabNum += 1
            f.write('stabilizer S%d ' % StabNum)
            f.write('(X, q[{i0}]), (X, q[{i1}]), (X, q[{i2}]), (X, q[{i3}])\n'.format(i0 = groups[i][0], i1 = groups[i][1], i2 = groups[i][2], i3 = groups[i][3]))

            StabNum += 1
            f.write('stabilizer S%d ' % StabNum)
            f.write('(Z, q[{i0}]), (Z, q[{i1}]), (Z, q[{i2}]), (Z, q[{i3}])\n'.format(i0 = groups[i][0], i1 = groups[i][1], i2 = groups[i][2], i3 = groups[i][3]))


def LDPC_code(fname):
    X = np.array(
        [
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        ]
    )

    Z = np.array(
        [
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1]
        ]
    )

    with open(fname, "w") as f:
        f.write('qreg q[%d]\n' % (18))
        StabNum = 0
        m, n = X.shape

        for i in range(m):
            StabNum += 1
            f.write('stabilizer S%d ' % StabNum)
            for k in range(n):
                if X[i][k] == 1:
                    f.write('(X, q[{i}]), '.format(i=k))
            f.write('\n')

            StabNum += 1
            f.write('stabilizer S%d ' % StabNum)
            for k in range(n):
                if Z[i][k] == 1:
                    f.write('(Z, q[{i}]), '.format(i=k))
            f.write('\n')

            
def bacon_shor_code(fname, m=3, n=3):
    def coord2index(i, j):        
        return (n * i + j)
    with open(fname, "w") as f:
        StabNum = 0
        for i in range(m-1):
            StabNum += 1
            f.write('stabilizer S%d ' % StabNum)
            for j in range(n):
                f.write('({op}, q[{i0}]), ({op}, q[{i1}]), '.format(op = 'X', i0 = coord2index(i, j), i1 = coord2index(i+1, j)))
            f.write('\n')
        for j in range(n-1):
            StabNum += 1
            f.write('stabilizer S%d ' % StabNum)
            for i in range(m):
                f.write('({op}, q[{i0}]), ({op}, q[{i1}]), '.format(op = 'Z', i0 = coord2index(i, j), i1 = coord2index(i, j+1)))
            f.write('\n')
            
def _8_3_2_code(fname):
    group = ['ZZZZIIII', 'ZZIIZZII', 'ZIZIZIZI', 'IIIIZZZZ', 'XXXXXXXX']
    with open(fname, "w") as f:
        f.write('qreg q[%d]\n' % (8))
        StabNum = 0
        for g in group:
            StabNum += 1
            f.write('stabilizer S%d ' % StabNum)
            for k, op in enumerate(g):
                if op != 'I':
                    f.write('({op}, q[{i0}]), '.format(op = op, i0 = k))
            f.write('\n')
            
def _16_4_3_code(fname):
    group = [[1,2,3,6,7,9], [2,4,6,8], [3,5,7,10], [6,8,9,11,12,14], [7,9,10,12,13,15], [12,14,15,16]]
    with open(fname, "w") as f:
        f.write('qreg q[%d]\n' % (16))
        StabNum = 0
        for g in group:
            StabNum += 1

            f.write('stabilizer S%d ' % StabNum)
            for i in g: 
                f.write('({op}, q[{i0}]), '.format(op = 'X', i0 = i-1))
            f.write('\n')    

            StabNum += 1
            f.write('stabilizer S%d ' % StabNum)
            for i in g: 
                f.write('({op}, q[{i0}]), '.format(op = 'Z', i0 = i-1))
            f.write('\n')

def _513_code(fname):
    group = ['IXZZX', 'XZZXI', 'ZZXIX', 'ZXIXZ']
    with open(fname, "w") as f:
        StabNum = 0
        for g in group:
            StabNum += 1
            f.write('stabilizer S%d ' % StabNum)
            for k, op in enumerate(g):
                if op != 'I':
                    f.write('({op}, q[{i0}]), '.format(op = op, i0 = k))
            f.write('\n')
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fname', help='path to stabilizer code file')
    parser.add_argument('-c', '--code', help='name of code')
    parser.add_argument('-d', '--dist', type=int, help='distance of the code')
    args = parser.parse_args()

    if args.code == 'SurfaceCode':
        a = surface_code(args.fname, d=args.dist)
    elif args.code == 'SteaneCode':
        a = steane_code(args.fname)
    # elif args.code == 'BaconShorCode':
    #     a = bacon_shor_code(args.fname, m=args.row, n=args.col)
    elif args.code == '8_3_2Code':
        a = _8_3_2_code(args.fname)
    elif args.code == '16_4_3Code':
        _16_4_3_code(args.fname)
    elif args.code == '513Code':
        a = _513_code(args.fname)
    elif args.code == 'LDPCCode':
        LDPC_code(args.fname)
    else:
        print('error')
        exit(-1)
    
