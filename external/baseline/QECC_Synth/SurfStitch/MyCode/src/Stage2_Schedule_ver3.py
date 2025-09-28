import os
import time
import json
import math
import argparse
import numpy as np
from collections import Counter
from pysat.solvers import Solver
import networkx as nx
from qiskit import QuantumCircuit
from sympy.combinatorics import Permutation

from lit2num import LiteralCode

SAT_GLOBAL = True
        
class Stage2_Solver():
    def __init__(self, Data, Stab, CG, cList, maxT=None, maxTime=None) -> None:
        self.literalCode = LiteralCode()
        self.Data = Data
        self.Stab = Stab
        
        self.CG = np.array(CG)
        self.cList = cList
        
        self.stabNum = len(self.Stab)
        self.dataNum = len(Data)
        self.physNum = len(CG)
        
        self.Cir = [[] for _ in range(self.physNum)]
        self.CtrlP = [[] for _ in range(self.physNum)]
        for k in range(self.stabNum):
            Anc_k =  self.Stab[k]['Ancilla']
            Ctrl_k = self.Stab[k]['Ctrl']
            
            for ind in range(len(Anc_k)):
                phys = Anc_k[ind]
                self.Cir[phys].extend([
                    ('reset_%d' % k, ind),
                    ('Mz_%d' % k, ind)
                ])
                for ind2 in range(len(Anc_k)):
                    self.Cir[phys].extend([
                        ('encode_%d' % k, ind, ind2),
                        ('decode_%d' % k, ind, ind2)
                    ])
                    if ind != ind2:
                        phys2 = Anc_k[ind2]
                        self.Cir[phys2].extend([
                            ('encode_%d' % k, ind, ind2),
                            ('decode_%d' % k, ind, ind2)
                        ])
                
            for ind in range(len(Ctrl_k)):
                ctrl = Ctrl_k[ind]
                phys1, phys2 = ctrl[2][1:3]
                self.Cir[phys1].append(('ctrl_%d' % k, ind))
                self.Cir[phys2].append(('ctrl_%d' % k, ind))
                self.CtrlP[phys2].append(('ctrl_%d' % k, k, ind))
        
        if cList:
            self.T = maxT #max(Counter([stab for pair in cList for stab in pair]).values()) * 5 + 5
        else:
            self.T = maxT
        self.maxT = maxT
        self.inc = 1.3
        
        self.maxTime = None # maxTime
        
        
    def InComm(self, stab1, stab2):
        InComm = []
        for i in stab1['Data']:
            if i in stab2['Data']:
                idx1 =  stab1['Data'].index(i)
                idx2 =  stab2['Data'].index(i)
                if stab1['Ctrl'][idx1][2][0] != stab2['Ctrl'][idx2][2][0]:
                    InComm.append((idx1, idx2))
        return InComm
    
    
    ## Constraint Generation ##
    
    def generateAndWriteClauses(self):
        self.writeAssignmentConstraint()
        self.writeCollisionConstraint()
        
        self.writeResetConstraint()
        self.writeEncodeConstraint()
        self.writeCtrlConstraint()
        self.writeDecodeConstraint()
        self.writeMzConstraint()
        
        self.writeIncompatibleConstraint()
        self.writePropertyConstraint()

        # if not self.Stab[0]["Ctrl"][0][-1] is None:
        #     self.writeProtocolConstraint()
 
    ## Hard Constraint ##
    
    def writeAssignmentConstraint(self):
        for k in range(self.stabNum):
            Anc_k = self.Stab[k]['Ancilla']
            L = len(Anc_k)
            
            # reset, measure
            for litName in ['reset_%d' % k, 'Mz_%d' % k]:
                self.literalCode.append(litName, (L, self.T))
                for idx in range(L):
                    self.writeHardClause([
                        [(litName, idx, t), True] for t in range(self.T)
                    ])
                    self.writeNoMoreConConstraint([
                        (litName, idx, t) for t in range(self.T)
                    ])
                    
            # encode, decode
            for litName in ['encode_%d' % k, 'decode_%d' % k]:
                self.literalCode.append(litName, (L, L, self.T))
                for idx in range(L):
                    # (litName, idx, idx2, t): CNOT(idx2, idx) -> t
                    self.writeHardClause([
                        [(litName, idx, idx2, t), True] for idx2 in range(L) for t in range(self.T)
                    ])
                    self.writeNoMoreConConstraint([
                        (litName, idx, idx2, t) for idx2 in range(L) for t in range(self.T)
                    ])
            
            # ctrl
            Ctrl_k = self.Stab[k]['Ctrl']
            litName = 'ctrl_%d' % k
            self.literalCode.append(litName, (len(Ctrl_k), self.T))
            for idx in range(len(Ctrl_k)):
                self.writeHardClause([
                    [(litName, idx, t), True] for t in range(self.T)
                ])
                self.writeNoMoreConConstraint([
                    (litName, idx, t) for t in range(self.T)
                ])
                
                  
    def writeCollisionConstraint(self):                
        for j in range(self.physNum):
            for t in range(self.T):
                self.writeNoMoreConConstraint([
                        (*cir, t) for cir in self.Cir[j]
                    ])


    def writeResetConstraint(self):
        for k in range(self.stabNum):
            litName = 'reset_%d' % k
            Anc_k = self.Stab[k]['Ancilla']
            L = len(Anc_k)
            for idx in range(L):
                # reset cannot be the last operartion
                self.writeHardClause([
                    [(litName, idx, self.T - 1), False]
                ])
                # reset phys[idx] is followed by CNOT(idx2, idx)  
                for t in range(self.T - 1):
                    self.writeHardClause([
                        [(litName, idx, t), False]
                    ] + [
                        [('encode_%d' % k, idx, idx2, t + 1), True] for idx2 in range(L)
                    ])
    
    
    def writeMzConstraint(self):
        for k in range(self.stabNum):
            litName = 'Mz_%d' % k
            Anc_k = self.Stab[k]['Ancilla']
            L = len(Anc_k)
            for idx in range(L):
                self.writeHardClause([
                    [(litName, idx, 0), False]
                ])
                for t in range(1, self.T):
                    self.writeHardClause([
                        [(litName, idx, t), False]
                    ] + [
                        [('decode_%d' % k, idx, idx2, t - 1), True] for idx2 in range(L)
                    ])

    
    def writeEncodeConstraint(self):
        for k in range(self.stabNum):
            litName = 'encode_%d' % k
            Anc_k = self.Stab[k]['Ancilla']
            L = len(Anc_k)
            G = self.CG[np.ix_(Anc_k, Anc_k)]
            
            # One root
            self.writeNoMoreConConstraint([
                (litName, idx, idx, t) for idx in range(L) for t in range(self.T)
            ])
            
            # Exactly one CNOT(idx2, idx) exsist for phys[idx]
            for idx in range(L):
                nonzeroIndices = [int(p[0]) for p in np.argwhere(G[:, idx]>0)] 
                self.writeHardClause([
                    [(litName, idx, idx2, t), True] for idx2 in nonzeroIndices + [idx] for t in range(self.T)
                ])
                self.writeNoMoreConConstraint([
                    (litName, idx, idx2, t) for idx2 in nonzeroIndices + [idx] for t in range(self.T)
                ])
            
            # Tree Structure
            self.literalCode.append('tree_en_%d' % k, (L, self.T))
            self.literalCode.append('b_tree_en_%d' % k, (L, L, self.T))
            for idx in range(L):
                self.writeHardClause([
                    [('tree_en_%d' % k, idx, 0), False]
                ])
                self.writeHardClause([
                    [('tree_en_%d' % k, idx, self.T - 1), True]
                ])
                for t in range(self.T - 1):
                    for idx2 in range(L):
                        if idx2 == idx:
                            self.writeHardClause([
                                [('b_tree_en_%d' % k, idx, idx, t + 1), False],
                                [(litName, idx, idx, t + 1), True]
                            ])
                        else: 
                            self.writeHardClause([
                                [('b_tree_en_%d' % k, idx, idx2, t + 1), False],
                                [(litName, idx, idx2, t + 1), True]
                            ])
                            self.writeHardClause([
                                [('b_tree_en_%d' % k, idx, idx2, t + 1), False],
                                [('tree_en_%d' % k, idx2, t), True]
                            ])
                    self.writeHardClause([
                        [('tree_en_%d' % k, idx, t + 1), False],
                        [('tree_en_%d' % k, idx, t), True]
                    ] + [
                        [('b_tree_en_%d' % k, idx, idx2, t + 1), True] for idx2 in range(L)
                    ])
    
    
    def writeDecodeConstraint(self):
        for k in range(self.stabNum):
            litName = 'decode_%d' % k
            Anc_k = self.Stab[k]['Ancilla']
            L = len(Anc_k)
            G = self.CG[np.ix_(Anc_k, Anc_k)]
            
            # One root
            self.writeNoMoreConConstraint([
                (litName, idx, idx, t) for idx in range(L) for t in range(self.T)
            ])
        
            # Exactly one CNOT
            for idx in range(L):
                nonzeroIndices = [int(p[0]) for p in np.argwhere(G[:, idx]>0)] 
                self.writeHardClause([
                    [(litName, idx, idx2, t), True] for idx2 in nonzeroIndices + [idx] for t in range(self.T)
                ])
                self.writeNoMoreConConstraint([
                    (litName, idx, idx2, t) for idx2 in nonzeroIndices + [idx] for t in range(self.T)
                ])
            
            # Tree Structure
            self.literalCode.append('tree_de_%d' % k, (L, self.T))
            self.literalCode.append('b_tree_de_%d' % k, (L, L, self.T))
            for idx in range(L):
                self.writeHardClause([
                    [('tree_de_%d' % k, idx, 0), True]
                ])
                self.writeHardClause([
                    [('tree_de_%d' % k, idx, self.T - 1), False]
                ])
                for t in range(1, self.T):
                    for idx2 in range(L):
                        if idx2 == idx:
                            self.writeHardClause([
                                [('b_tree_de_%d' % k, idx, idx, t - 1), False],
                                [(litName, idx, idx, t - 1), True]
                            ])
                        else: 
                            self.writeHardClause([
                                [('b_tree_de_%d' % k, idx, idx2, t - 1), False],
                                [(litName, idx, idx2, t - 1), True]
                            ])
                            self.writeHardClause([
                                [('b_tree_de_%d' % k, idx, idx2, t - 1), False],
                                [('tree_de_%d' % k, idx2, t), True]
                            ])
                    self.writeHardClause([
                        [('tree_de_%d' % k, idx, t - 1), False],
                        [('tree_de_%d' % k, idx, t), True]
                    ] + [
                        [('b_tree_de_%d' % k, idx, idx2, t - 1), True] for idx2 in range(L)
                    ])
                    
    
    def writeCtrlConstraint(self):
        for k in range(self.stabNum):
            ctrlName = 'ctrl_%d' % k
            encodeName = 'encode_%d' % k
            decodeName = 'decode_%d' % k
            
            Anc_k = self.Stab[k]['Ancilla']
            Ctrl_k = self.Stab[k]['Ctrl']
            
            # Encode is ahead of Ctrl
            b_tlName = 'b_TL_en_ctrl_%d' % k
            self.literalCode.append(b_tlName, (len(Anc_k), self.T))
            for idx_anc in range(len(Anc_k)):
                # [1, 1, ..., 1, 0, 0, ..., 0]
                for t in range(self.T - 1):
                    self.writeHardClause([
                        [(b_tlName, idx_anc, t), True],
                        [(b_tlName, idx_anc, t + 1), False] 
                    ]) 
            self.code2tl(Anc_k, encodeName, b_tlName, True)
            self.ctrl2tl(Anc_k, Ctrl_k, ctrlName, b_tlName, False)
            
            # Ctrl is ahead of Decode
            b_tlName = 'b_TL_ctrl_de_%d' % k
            self.literalCode.append(b_tlName, (len(Anc_k), self.T))
            for idx_anc in range(len(Anc_k)):
                # [1, 1, ..., 1, 0, 0, ..., 0]
                for t in range(self.T - 1):
                    self.writeHardClause([
                        [(b_tlName, idx_anc, t), True],
                        [(b_tlName, idx_anc, t + 1), False] 
                    ]) 
            self.ctrl2tl(Anc_k, Ctrl_k, ctrlName, b_tlName, True)
            self.code2tl(Anc_k, decodeName, b_tlName, False)

            # Encode is ahead of Decode
            b_tlName = 'b_TL_en_de_%d' % k
            self.literalCode.append(b_tlName, (len(Anc_k), self.T))
            for idx_anc in range(len(Anc_k)):
                # [1, 1, ..., 1, 0, 0, ..., 0]
                for t in range(self.T - 1):
                    self.writeHardClause([
                        [(b_tlName, idx_anc, t), True],
                        [(b_tlName, idx_anc, t + 1), False] 
                    ]) 
            self.code2tl(Anc_k, encodeName, b_tlName, True)
            self.code2tl(Anc_k, decodeName, b_tlName, False)
        
        
    def writeIncompatibleConstraint(self):
        self.literalCode.append('ahead', (self.stabNum, self.stabNum))
        
        for k1, k2 in self.cList:
            stab1 = self.Stab[k1]
            stab2 = self.Stab[k2]
            
            self.writeHardClause([
                [('ahead', k1, k2), True], 
                [('ahead', k2, k1), True]
            ]) 
            
            for j in set(stab1['Ancilla']) & set(stab2['Ancilla']):
                idx1 = stab1['Ancilla'].index(j)
                idx2 = stab2['Ancilla'].index(j)
                
                b_LitName = 'b_ahead_(%d, %d)_%d' % (k1, k2, j)
                
                self.literalCode.append(b_LitName, (self.T, ))
                for t in range(self.T - 1):
                    self.writeHardClause([
                        [(b_LitName, t), True], 
                        [(b_LitName, t + 1), False]
                    ]) 
                
                for t in range(self.T):
                    # ahead(k1, k2)
                    self.writeHardClause([
                        [('ahead', k1, k2), False], 
                        [('Mz_%d' % k1, idx1, t), False], 
                        [(b_LitName, t), True]
                    ])
                    self.writeHardClause([
                        [('ahead', k1, k2), False], 
                        [('reset_%d' % k2, idx2, t), False], 
                        [(b_LitName, t), False]
                    ])
                    
                    # ahead(k2, k1)
                    self.writeHardClause([
                        [('ahead', k2, k1), False], 
                        [('Mz_%d' % k2, idx2, t), False], 
                        [(b_LitName, t), True]
                    ])       

                    self.writeHardClause([
                        [('ahead', k2, k1), False], 
                        [('reset_%d' % k1, idx1, t), False], 
                        [(b_LitName, t), False]
                    ])


    def writePropertyConstraint(self):
        for k1 in range(self.stabNum):
            for k2 in range(k1):
                stab1, stab2 = self.Stab[k1], self.Stab[k2]
                InComm = self.InComm(stab1, stab2)
                
                if InComm and (k1, k2) not in self.cList:
                    N = len(InComm)
                    
                    litName1 = 'ctrl_%d' % k1
                    litName2 = 'ctrl_%d' % k2
                    b_propName = 'b_prop' + '(%d, %d)' % (k1, k2)
                    
                    self.literalCode.append(b_propName, (N, self.T))
                    for n in range(N):
                        idx1, idx2 = InComm[n]
                        for t in range(self.T):
                            self.writeHardClause([
                                [(litName1, idx1, t), False],
                                [(b_propName, n, t), False]
                            ])
                            self.writeHardClause([
                                [(litName2, idx2, t), False],
                                [(b_propName, n, t), True]
                            ])

                        for t in range(1, self.T):
                            self.writeHardClause([
                                [(litName1, idx1, t), True],
                                [(litName2, idx2, t), True], 
                                [(b_propName, n, t - 1), False], 
                                [(b_propName, n, t), True] 
                            ])
                            self.writeHardClause([
                                [(litName1, idx1, t), True],
                                [(litName2, idx2, t), True], 
                                [(b_propName, n, t - 1), True], 
                                [(b_propName, n, t), False] 
                            ])
                    
                    b_addName = 'b_add' + '(%d, %d)' % (k1, k2)
                    self.literalCode.append(b_addName, (N + 1, ))
                    self.writeHardClause([
                        [(b_addName, 0), False]
                    ])
                    self.writeHardClause([
                        [(b_addName, N), False]
                    ])
                    for n in range(N):
                        # prop_n + add_n = add_n + 1
                        self.writeHardClause([
                            [(b_propName, n, self.T - 1), False], [(b_addName, n), False],
                            [(b_addName, n + 1), False]
                        ])
                        self.writeHardClause([
                            [(b_propName, n, self.T - 1), False], [(b_addName, n), True],
                            [(b_addName, n + 1), True]
                        ])
                        self.writeHardClause([
                            [(b_propName, n, self.T - 1), True], [(b_addName, n), False],
                            [(b_addName, n + 1), True]
                        ])
                        self.writeHardClause([
                            [(b_propName, n, self.T - 1), True], [(b_addName, n), True],
                            [(b_addName, n + 1), False]
                        ])
            
    def writeProtocolConstraint(self):                
        for j in range(self.physNum):
            L = []
            for cir in self.CtrlP[j]:
                ctrlName, k, ind = cir
                ctrl = self.Stab[k]["Ctrl"][ind]
                t = ctrl[-1]
                L.append(((ctrlName, ind), t))
            if L:
                L.sort(key= lambda x : x[-1])
                for i in range(len(L) - 1):
                    ctrl1 = L[i][0]
                    ctrl2 = L[i+1][0]
                    
                    b_LitName = 'b_ahead2_(%d, %d)' % (j, i)
                    self.literalCode.append(b_LitName, (self.T, ))
                    
                    for t in range(self.T - 1):
                        self.writeHardClause([
                            [(b_LitName, t), True], 
                            [(b_LitName, t + 1), False]
                        ]) 
                    
                    for t in range(self.T):
                        self.writeHardClause([
                            [(*ctrl1, t), False], 
                            [(b_LitName, t), True]
                        ])
                        self.writeHardClause([
                            [(*ctrl2, t), False],
                            [(b_LitName, t), False]
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
    
    
    def code2tl(self, Anc_k, litName, b_tlName, value):
        L = len(Anc_k)
        for idx1 in range(L):
            for idx2 in range(L):
                for t in range(self.T):
                    self.writeHardClause([
                        [(litName, idx1, idx2, t), False], 
                        [(b_tlName, idx1, t), value]
                    ])
                    
                    self.writeHardClause([
                        [(litName, idx2, idx1, t), False], 
                        [(b_tlName, idx1, t), value]
                    ])


    def ctrl2tl(self, Anc_k, Ctrl_k, litName, b_tlName, value):
        for idx_anc in range(len(Anc_k)):
            anc = Anc_k[idx_anc]
            for idx_ctrl in range(len(Ctrl_k)):
                if Ctrl_k[idx_ctrl][2][1] == anc:
                    for t in range(self.T):
                        self.writeHardClause([
                            [(litName, idx_ctrl, t), False], 
                            [(b_tlName, idx_anc, t), value]
                        ])

    
    def writeHardClause(self, clause):
        self.SATsolver.add_clause([self.literalCode.encode(lit, value) for lit, value in clause])
    
    
    ## Reading MaxSat solver output ##
    
    def readPySatOutput(self, litList):
        result = {}
        lits = self.SATsolver.get_model()
        for litName in litList:
            shifted, size, _,= self.literalCode.alloc[litName]
            L = [self.literalCode.decode(i, int(lits[i-1])) for i in range(shifted, min(shifted + size, len(lits) + 1))]
            
            result[litName] = [lit[2] for lit in filter(lambda x: x[0] and (x[1] == litName), L)]

        return result
    
    
    ## Solving ##
    
    def solve(self):
        t_s = time.time()
        solved = False
        while True:
            print("\tRunning for T =", self.T)
            with Solver() as self.SATsolver:
                
                self.literalCode.reset()
                self.generateAndWriteClauses()
                if self.SATsolver.solve():
                
                    solved = True
                    Circuit = []
                    Synd_qubits = []
                    
                    for k in range(self.stabNum): #[1, 4]: #range(self.stabNum):
                        Anc_k = self.Stab[k]['Ancilla']
                        Ctrl_k = self.Stab[k]['Ctrl']
                        litList = ['reset_%d' % k, 'encode_%d' % k, 'ctrl_%d' % k, 'decode_%d' % k, 'Mz_%d' % k]

                        result = self.readPySatOutput(litList)
                        
                        for idx, t in result['reset_%d' % k]:
                            Circuit.append((t, ['reset', Anc_k[idx]]))
                        
                        for idx, t in result['Mz_%d' % k]:
                            Circuit.append((t, ['Mz', k, Anc_k[idx]]))
                            
                        for idx1, idx2, t in result['encode_%d' % k]:
                            if idx1 == idx2:
                                Circuit.append((t, ['H', Anc_k[idx1]]))                  
                            else: 
                                Circuit.append((t, ['X', Anc_k[idx2], Anc_k[idx1]]))
                                
                        for idx1, idx2, t in result['decode_%d' % k]:
                            if idx1 == idx2:
                                Circuit.append((t, ['H', Anc_k[idx1]]))
                                Synd_qubits.append(Anc_k[idx1])                   
                            else: 
                                Circuit.append((t, ['X', Anc_k[idx2], Anc_k[idx1]]))   
                        
                        for idx, t in result['ctrl_%d' % k]:
                            Circuit.append((t, Ctrl_k[idx][2]))
                        
                    Circuit.sort(key=lambda x: x[0])
                    QC = toQASM(self.physNum, Circuit)
                    self.T = QC.depth() - 1
                else:
                    # if self.maxT:
                    #     if self.T != self.maxT:
                    #         self.T = min(math.ceil(self.T * self.inc), self.maxT)
                    #     else:
                    #         raise ValueError("UNsatisfied with time step bound %d" % self.maxT)
                    # else: 
                    #     self.T = math.ceil(self.T * self.inc)
                    if solved:
                        ops = dict(QC.count_ops())
                        print()
                        print(' Timecost:', time.time() - t_s)
                        print(' Gate count:', ops)
                        print(' Controlled gate count:', (ops['cx'] if 'cx' in ops.keys() else 0) + (ops['cz'] if 'cz' in ops.keys() else 0))
                        print(' Circuit depth:', QC.depth())
                        break

                    if self.maxT:
                        if self.T != self.maxT:
                            self.T = min(math.ceil(self.T * self.inc), self.maxT)
                        else:
                            raise ValueError("UNsatisfied with time step bound %d" % self.maxT)
                    else: 
                        self.T = math.ceil(self.T * self.inc)
                        
                if self.maxTime and time.process_time() > t_s + self.maxTime:
                    raise TimeoutError
                
        return Circuit, Synd_qubits


def S2_transpile(chunkNum, Data, Stab, CG, idx2coord, Collison_list, maxT, S2_resultName):
    dataNum = len(Data[0])
    G = nx.from_numpy_array(CG)

    Swap_layer = []
    for k in range(chunkNum):
        Swap_layer.append([])

        map_source = Data[k].copy()
        map_target = Data[(k+1)%chunkNum].copy()

        Swaps= []
        while set(map_target).difference(map_source):
            q = set(map_target).difference(map_source).pop()
            idx = map_target.index(q)
            Swaps.append((map_source[idx], map_target[idx]))
            map_source[idx] = map_target[idx]
        if map_source != map_target:
            pattern = []
            for idx in range(dataNum):
                if map_source[idx] == map_target[idx]:
                    pattern.append(idx)
                else:
                    pattern.append(map_target.index(map_source[idx]))
            for idx0, idx1 in Permutation(pattern).transpositions():
                Swaps.append((map_source[idx0], map_source[idx1]))

        map_source = Data[k].copy()
        for q0, q1 in Swaps:
            path = nx.dijkstra_path(G, q0, q1)
            for i, j in zip(path[:-1]+list(reversed(path[:-2])), path[1:]+list(reversed(path[1:-1]))):
                if i in map_source or j in map_source:
                    map_source = [i if q == j else (j if q == i else q) for q in map_source]
                    Swap_layer[-1].append((None, ['SWAP', i, j]))
        assert map_source == Data[(k+1)%chunkNum].copy()

    result = {
        'chunkNum': chunkNum,
        'idx2coord': idx2coord,
        'Stab': Stab,
        'Data': Data,
        'Swap_layer': Swap_layer
    }

    Circuit = [None] * chunkNum
    for i in range(chunkNum):
        print("\n Chunk %d:\n" % i)
        solver = Stage2_Solver(Data[i], Stab[i], CG, Collison_list[i], maxT=maxT)
        Circuit[i], Synd_qubits = solver.solve()
        for k, synd in enumerate(Synd_qubits):
            result['Stab'][i][k]['synd'] = synd

    result['Circuit'] = Circuit

    physNum = len(CG)
    QC = QuantumCircuit(physNum, physNum)
    for i in range(chunkNum):
        QC = QC.compose(toQASM(len(CG), result['Circuit'][i]))
        QC = QC.compose(toQASM(len(CG), result['Swap_layer'][i]))
    ops = dict(QC.count_ops())
    print()
    print('Total Gate count:', ops)
    print(
        'Total Controlled gate count:', 
        (ops['cx'] if 'cx' in ops.keys() else 0) + (ops['cz'] if 'cz' in ops.keys() else 0) + (3 * ops['swap'] if 'swap' in ops.keys() else 0)
    )
    print('Total Circuit depth:', QC.depth())
    
    with open(S2_resultName+'.json', 'w') as f:
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


def toQASM(physNum, Circuit):
    QC = QuantumCircuit(physNum, physNum)
    n = 0 
    for _, cir in Circuit:
        if cir[0] == 'reset':
            QC.reset(cir[1])
        elif cir[0] == 'H':
            QC.h(cir[1])
        elif cir[0] == 'X':
            n += 1
            QC.cx(cir[1], cir[2])
        elif cir[0] == 'Y':
            n += 1
            QC.cy(cir[1], cir[2])
        elif cir[0] == 'Z':
            n += 1
            QC.cz(cir[1], cir[2])    
        elif cir[0] == 'Mz':
            QC.measure(cir[2], cir[2])
        elif cir[0] == 'SWAP':
            QC.swap(cir[1], cir[2])

    # QC.draw('latex_source', filename='./file.tex')
    return QC


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('S1_result', help='path to input Tree_result')
#     parser.add_argument('-l', '--maxT', type=int, default=None, help='upper bound for time step')
#     parser.add_argument('-t', '--timeout', type=int, default=5,help='maximum run time for SAT in seconds')
#     args = parser.parse_args()

#     base, _ = os.path.splitext(os.path.basename(args.S1_result)[len('S1_result_'):])
    
#     S2_transpile(args.S1_result, args.maxT, 'S2_result_'+base, maxTime=args.timeout)