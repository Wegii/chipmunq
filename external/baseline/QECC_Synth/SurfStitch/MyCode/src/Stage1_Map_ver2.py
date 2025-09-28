import time
import json
import subprocess
import numpy as np
from lit2num import LiteralCode

SAT_GLOBAL = True

class Stage1_Solver():
    def __init__(self, code, CG, protocal, maxL, wcnfName, solName, timeout=None) -> None:
        self.literalCode = LiteralCode()
        self.dataNum, self.Stab = code
        self.stabNum = len(self.Stab)
        self.physNum = len(CG)
        self.Data = [-1] * self.dataNum         # Data Qubit Mapping
        self.CG = CG
        
        self.wcnfName = wcnfName + '.wcnf'
        self.solName = solName + '.txt'
        
        self.protocal = protocal        # Stabilizer Measurement protocal
        
        if maxL is None:
            self.maxL = self.physNum - self.dataNum
        else: 
            self.maxL = maxL
            
        # init self.L
        # if prot == 'Shor':
        #     self.L = max(len(stab['Data']) for stab in self.Stab) - 1
        # elif prot == 'Flag-Bridge':
        #     self.L = 0
        # else: 
        #     raise ValueError("Undefined measurement protocal")
        
        if protocal == 'Shor':
            self.L = [len(stab['Data']) - 1 for stab in self.Stab]
        elif protocal == 'Flag-Bridge':
            self.L = [0] * self.stabNum
        else: 
            raise ValueError("Undefined measurement protocal")
        
        self.timeout = timeout
        
       
    ## Constraint Generation ##
    
    def generateAndWriteClauses(self):
        ''' 
            Writes the constraints corresponding to a particular MaxSat Instance to the wcnf file
        '''
        with open(self.wcnfName, 'w') as self.clause_f:
            self.writeMapConstraint()
            self.writeAllocConstraint()
            self.writeInjectivityConstraint()
            self.writeConnectivityConstraint()
            self.writeProtocalConstraint()
            self.writeOptimizationConstraints()



        
    ## Hard Constraint ##
    
    def writeMapConstraint(self):
        # Data qubit q is mapped to exactly one physical qubit
        self.literalCode.append('q', (self.dataNum, self.physNum))
        for i in range(self.dataNum):
            self.writeHardClause([
                [('q', i, j), True] for j in range(self.physNum)
            ])
            self.writeNoMoreConConstraint([
                ('q', i, j) for j in range(self.physNum)
            ])
            
        # Physical qubit p is assigned to no more than one data qubit
        for j in range(self.physNum):
            self.writeNoMoreConConstraint([
                ('q', i, j) for i in range(self.dataNum)
            ])
                

    def writeAllocConstraint(self): 
        # At least one physical qubit is allocated to stabilizer s
        self.literalCode.append('s', (self.stabNum, self.physNum))
        # for k in range(self.stabNum):
        #     self.writeHardClause([
        #         [('s', k, j), True] for j in range(self.physNum)
        #     ])   
        
        
        for k in range(self.stabNum):
            b_lenName = 'b_len' + '(%d)' % k
            self.literalCode.append(b_lenName, (self.L[k], self.physNum))
            # When physical qubit p is allocated to s, p is assigned to at least one layer from the L layers of stabilizer s
            for j in range(self.physNum):
                self.writeHardClause([
                    [('s', k, j), False]
                ] + [
                    [(b_lenName, l, j), True] for l in range(self.L[k])
                ])
            
            # l-th layer's ancilla qubit of stabilizer s is mapped to exactly one physical qubit
            for l in range(self.L[k]): 
                # self.writeHardClause([
                #     [(b_lenName, l, j), True] for j in range(self.physNum)
                # ])
                self.writeNoMoreConConstraint([
                    (b_lenName, l, j) for j in range(self.physNum)
                ])
        

    def writeInjectivityConstraint(self):
        # When physical qubit p is assigned to a data qubit, it can not be allocated to any stabilizers 
        for j in range(self.physNum):
            for i in range(self.dataNum):
                for k in range(self.stabNum):
                    self.writeHardClause([
                        [('q', i, j), False], 
                        [('s', k, j), False]
                    ])


    def writeConnectivityConstraint(self):
        # Physical qubits allocated to stabilizer s are connective 
        for k in range(self.stabNum):
            b_conName = 'b_con' + '(%d)' % k
            self.literalCode.append(b_conName, (self.physNum, self.physNum, self.L[k]))
            
            for j in range(self.physNum):
                # BFS check for stabilizer s, starting at physical qubit p
                for j2 in range(self.physNum):
                    if j2 != j:
                        self.writeHardClause([
                            [(b_conName, j, j2, 0), False]
                        ])
                        
                    nonzeroIndices = np.argwhere(self.CG[:, j2]>0)
                    for l in range(self.L[k]-1):
                        self.writeHardClause([
                            [(b_conName, j, j2, l+1), False], 
                            [('s', k, j2), True]
                        ])
                        
                        self.writeHardClause([
                            [(b_conName, j, j2, l+1), False], 
                            [(b_conName, j, j2, l), True]
                        ] + [
                            [(b_conName, j, int(p), l), True] for p in nonzeroIndices
                        ])
                        
                    self.writeHardClause([
                        [('s', k, j), False], 
                        [('s', k, j2), False], 
                        [(b_conName, j, j2, self.L[k]-1), True]
                    ])


    def writeProtocalConstraint(self):
        if self.protocal == 'Shor':
            # ctrl(k, i, [P, j1, j2]) <-> ST(k, i, j1)
            self.literalCode.append('ST', (self.stabNum, self.dataNum, self.physNum))
            for k in range(self.stabNum):
                for i in self.Stab[k]['Data']:
                    for j in range(self.physNum):
                        # ST(k, i, j1) -> anc(k, j1)
                        self.writeHardClause([
                            [('ST', k, i, j), False], 
                            [('s', k, j), True]
                        ])
                        
                        # ST(k, i, j1) -> Ngbr(j1, j2)
                        nonzeroIndices = np.argwhere(self.CG[j, :]>0)
                        self.writeHardClause([
                            [('ST', k, i, j), False]
                        ] + [
                            [('q', i, int(p)), True] for p in nonzeroIndices
                        ])
                        
                    # At least one ST qubit is assigned to data qubit q
                    self.writeHardClause([
                        [('ST', k, i, j), True] for j in range(self.physNum)
                    ])
            
            # In each stabilizer, ST qubit st is assigned to no more than one data qubit
            for k in range(self.stabNum):
                for j in range(self.physNum):
                    dataQubits = self.Stab[k]['Data']
                    self.writeNoMoreConConstraint([
                        ('ST', k, dataQubits[i], j) for i in range(len(dataQubits))
                    ])
                        
        elif self.protocal == 'Flag-Bridge':
            # In each stabilizer, data qubit q is mapped to be adjacent to at least one of the ancilla qubits
            for k in range(self.stabNum):
                for i in self.Stab[k]['Data']:
                    for j in range(self.physNum):
                        nonzeroIndices = np.argwhere(self.CG[:, j]>0)
                        self.writeHardClause([
                            [('q', i, j), False]
                        ] + [
                            [('s', k, int(p)), True] for p in nonzeroIndices
                        ])


    def writeNoMoreConConstraint(self, lits):
        # No more than one lit in the lits can be assigned the value True
        N = len(lits)
        litName = self.literalCode.auxiliary((N + 1, ))
        for n in range(N):
            self.writeHardClause([
                [(litName, n), True], 
                [(litName, n + 1), False]
            ]) 
                
            self.writeHardClause([
                [lits[n], False], 
                [(litName, n), True]
            ])
                
            self.writeHardClause([
                [lits[n], False],
                [(litName, n + 1), False]
            ])
        

    ## Soft Constraint ##

    def writeOptimizationConstraints(self):
        w = 1
        # P2: Incompatible stabilizer pair
        self.literalCode.append('in-cmpt', (self.stabNum, self.stabNum))
        for k in range(self.stabNum):
            for k2 in range(k):
                for j in range(self.physNum):
                    self.writeHardClause([
                        [('s', k, j), False], 
                        [('s', k2, j), False], 
                        [('in-cmpt', k, k2), True]
                    ])
                    
                self.writeSoftClause(1, [[('in-cmpt', k, k2), False]])
                # w += 1
        
        # # P2: Incompatible stabilizer pair
        # for k in range(self.stabNum):
        #     for j in range(self.physNum):
        #         for k2 in range(k):
        #             self.writeSoftClause(1, [
        #                 [('s', k, j), False], 
        #                 [('s', k2, j), False]
        #             ])
        #             # w += 1
        
        # P1: Total ancilla block size
        for k in range(self.stabNum):
            for j in range(self.physNum):
                self.writeSoftClause(w, [[('s', k, j), False]])
                    
        
    def writeHardClause(self, clause):
        self.clause_f.write('h ')
        for lit, value in clause:
            self.clause_f.write(str(self.literalCode.encode(lit, value)) + ' ')
        self.clause_f.write('0\n')


    def writeSoftClause(self, weight, clause):
        self.clause_f.write(str(weight) + ' ')
        for lit, value in clause:
            self.clause_f.write(str(self.literalCode.encode(lit, value)) + ' ')
        self.clause_f.write('0\n')


    ## Reading MaxSat solver output ##
    
    def readMaxSatOutput(self, litList=['q', 's']):
        result = {}
        with open(self.solName) as f:
            for line in f:
                if line.startswith('v'):
                    lits = list(line.split()[1])
                    for litName in litList:
                        shifted, size, _,= self.literalCode.alloc[litName]
                        L = [self.literalCode.decode(i, int(lits[i-1])) for i in range(shifted, min(shifted + size, len(lits) + 1))]
                        
                        result[litName] = [lit[2] for lit in filter(lambda x: x[0] and (x[1] == litName), L)]
        return result
    
    
    ## Solving ##
    
    def solve(self):
        result = {}
        t_s = time.time()
        while not result:
            self.L = [l + 1 for l in self.L]
            print(' Running for L =', self.L)
            if min(self.L) > self.maxL:
                raise ValueError("UNsatisfied with maximum ancilla block size %d" % self.maxL)

            self.literalCode.reset()
            self.generateAndWriteClauses()
            if SAT_GLOBAL:
                if self.timeout:
                    rem_time = self.timeout - (time.time() - t_s)
                else:
                    rem_time = None

                try:
                    # p = subprocess.Popen(['lib/NuWLS-c/bin/nuwls-c_static', self.wcnfName],  stdout=open(self.solName, 'w')) # 43200
                    p = subprocess.Popen(['timeout', '-s', '15', '14400', './lib/NuWLS-c/bin/nuwls-c_static', self.wcnfName],  stdout=open(self.solName, 'w'))
                    p.wait(rem_time)
                except subprocess.TimeoutExpired:
                    p.terminate()
                    raise TimeoutError
                
            result = self.readMaxSatOutput()

        for i, j in result['q']:
            self.Data[i] = j
        
        for k, j in result['s']:
            self.Stab[k]['Ancilla'].append(j)
        
        self.Collison_list = self.readMaxSatOutput(['in-cmpt'])['in-cmpt']
        
        if self.protocal == 'Shor':
            ST_result = self.readMaxSatOutput(['ST'])['ST']
            for k, i, j in ST_result:
                stab = self.Stab[k]
                ind = stab['Data'].index(i)
                stab['Ctrl'][ind][2][1:3] = j, self.Data[i]
        elif self.protocal == 'Flag-Bridge':
            for k in range(self.stabNum):
                stab = self.Stab[k]
                for ind in range(len(stab['Data'])):
                    i = stab['Data'][ind]
                    Data_i = self.Data[i]
                    for j in np.argwhere(self.CG[:, Data_i]>0):
                        if int(j) in self.Stab[k]['Ancilla']:
                            stab['Ctrl'][ind][2][1:3] = int(j), Data_i
                            break

        print()
        print('Timecost:', time.time() - t_s)
        print('Length:', self.L)
        print("Total ancilla block size:", sum(len(stab['Ancilla']) for stab in self.Stab))
                

def S1_transpile(code, CG, protocal, maxL, wcnfName, solName, S1_resultName):
    solver = Stage1_Solver(code, CG, protocal, maxL, wcnfName, solName)
    solver.solve()
    
    result = {
        'Data': solver.Data,
        'Stab': solver.Stab,
        'Collison_list': solver.Collison_list
    }
    
    with open(S1_resultName+'.json', 'w') as f:
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)
        json.dump(result, f, cls=NpEncoder)
    

    # return result
    # Data_Map = 'Data: \n' 
    # for data, phys in enumerate(solver.Data):
    #     Data_Map += ('q[%s] -> %d \n' % (solver.data2cir[data], phys))
    
    # Stab_Alloc = 'Stab_Alloc: \n'
    # for stab in solver.Stab:
    #     Stab_Alloc += '%s ->' % stab['name']
    #     for phys in stab['Ancilla']:
    #         Stab_Alloc += ' %d' % phys
    #     Stab_Alloc += '\n'
    
    # print('Max Length:', solver.L, '\n')
    # print(Data_Map, Stab_Alloc, sep='\n')
    
    
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('Code', help='path to input stabilizer code file')
#     parser.add_argument('Arch', help='path to input architecture file')
#     parser.add_argument('-p', '--protocal', default='Flag-Bridge', help='name of measurment protocal')
#     parser.add_argument('-mL', '--maxL', type=int, default=None, help='maximum size for the ancilla block')
    
#     args = parser.parse_args()
            
#     base, _ = os.path.splitext(os.path.basename(args.StabCode))
    
#     result = S1_transpile(args.StabCode, arch, index2coor, 'S1_clause_'+base, 'S1_sol_'+base, 'S1_result_'+base, prot=args.prot, maxL=args.maxL)