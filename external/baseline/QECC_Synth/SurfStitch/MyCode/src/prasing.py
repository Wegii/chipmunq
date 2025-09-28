import re

def extract(fname):
    '''
        Input parsing
    '''
    Stab = []
    data2cir = []
    cir2data = {}
    stabNo = 0
    dataNo = 0
    ctrlNo = 0
    with open(fname) as f:
        for line in f:
            name = re.match(r'stabilizer\s+(\w+).*', line)
            if name:
                match = re.findall(r'\(([XYZ]), q\[(\d+)\]\)', line)
                if not match:
                    match = re.findall(r'\(([XYZ]), q\[(\d+)\], \d\)', line)
                S_k = {'name':name.group(1), 'Data':[], 'Ancilla':[], 'Ctrl':[]}
                for Pauli, cirQubit in match:
                    if cirQubit not in data2cir:
                        data2cir.append(cirQubit)
                        cir2data[cirQubit] = dataNo
                        dataNo += 1
                    # Data
                    i = cir2data[cirQubit]
                    S_k['Data'].append(i)
                    # CP gate: [stabNo, dataNo, [P, ctrl, tgt], t]
                    S_k['Ctrl'].append([stabNo, i, [Pauli, None, None]])
                Stab.append(S_k)
                stabNo += 1
    return Stab, data2cir, cir2data

def extract2(fname, result):
    '''
        Input parsing
    '''
    with open(fname) as f:
        k = 0
        for line in f:
            name = re.match(r'stabilizer\s+(\w+).*', line)
            if name:
                match = re.findall(r'\(([XYZ]), q\[(\d+)\], (\d)\)', line)
                for _, cirQubit, t in match:
                    # Data
                    i = result['data2cir'].index(cirQubit)
                    ind = result['Stab'][k]['Data'].index(i)
                    result['Stab'][k]['Ctrl'][ind].append(int(t))
                k += 1
    return result

