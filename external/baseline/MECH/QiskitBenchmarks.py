from random import choices, sample
from collections import defaultdict
from math import *
from Chiplet import *

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreSwap, NoiseAdaptiveLayout, CXCancellation, CommutativeCancellation, InverseCancellation, CommutativeInverseCancellation
from qiskit.transpiler.passes.layout import SabreLayout
from qiskit.compiler import transpile

def genQiskitQFT(qubit_num, line_num):
    qiskit_circuit = QuantumCircuit(line_num)
    for c in range(qubit_num):
        for t in range(c+1, qubit_num):
            qiskit_circuit.rz(pi/3, t)
            qiskit_circuit.cx(c,t)
            qiskit_circuit.rz(pi/3, t)
            qiskit_circuit.cx(c,t)
    print('{} gates'.format(qubit_num**2))
    return qiskit_circuit

def genQiskitVQE(qubit_num, line_num):
    qiskit_circuit = QuantumCircuit(line_num)
    for i in range(qubit_num):
        for j in range(i+1, qubit_num):
            qiskit_circuit.cx(i,j)
    print('{} gates'.format(qubit_num**2/2))
    return qiskit_circuit

def genQiskitFullyConnectedCPhase(qubit_num, line_num):
    qiskit_circuit = QuantumCircuit(line_num)
    for i in range(qubit_num):
        for j in range(i+1, qubit_num):
            qiskit_circuit.cp(pi/2,i,j)
    print('{} gates'.format(len(qiskit_circuit)))
    return qiskit_circuit

def genQiskitQAOA(qubit_num, line_num, percentage=0.5, sorted=True):
    all_possible_gates = [(i,j) for i in range(qubit_num) for j in range(i+1, qubit_num)]
    gates = list(set(sample(all_possible_gates, k=int(len(all_possible_gates)*percentage))))
    if sorted:
        gates.sort()
    print("{}/{} gates selected : {} ... {}".format(len(gates), len(all_possible_gates), gates[:10], gates[-10:]))
    
    qiskit_circuit = QuantumCircuit(line_num)
    for c,t in gates:
        qiskit_circuit.cx(c,t)
        qiskit_circuit.rz(pi/3, t)
        qiskit_circuit.cx(c,t)
    return qiskit_circuit

def genQiskitBV(qubit_num, line_num, percentage=0.5, sorted=True):
    all_possible_gates = [(0,i) for i in range(1,qubit_num)]
    gates = list(set(sample(all_possible_gates, k=int(len(all_possible_gates)*percentage))))
    if sorted:
        gates.sort()
    print("{}/{} gates selected : {} ... {}".format(len(gates), len(all_possible_gates), gates[:10], gates[-10:]))
    
    qiskit_circuit = QuantumCircuit(line_num)
    for c,t in gates:
        qiskit_circuit.cx(c,t)
    return qiskit_circuit


def count_norm_cnots(G, sabre_circ, cross_chip_ratio=7.4):
    within_chip_cnots = 0
    cross_chip_cnots = 0
    for gate in sabre_circ:
        if gate.operation.num_qubits < 2:
            continue
        q1, q2 = gate.qubits
        q1, q2 = q1.index, q2.index
        idx_qubit_dict = gen_idx_qubit_dict(G)
        edge = (idx_qubit_dict[q1], idx_qubit_dict[q2])
        if gate.operation.name in ['cx', 'cp'] and G.edges[edge]['type'] == 'on_chip':
            within_chip_cnots += 1
        elif gate.operation.name == 'swap' and G.edges[edge]['type'] == 'on_chip':
            within_chip_cnots += 3
        elif gate.operation.name in ['cx', 'cp'] and G.edges[edge]['type'] == 'cross_chip':
            cross_chip_cnots += 1
        elif gate.operation.name == 'swap' and G.edges[edge]['type'] == 'cross_chip':
            cross_chip_cnots += 3
    return within_chip_cnots, cross_chip_cnots, within_chip_cnots + cross_chip_cnots * cross_chip_ratio

def backend_from_weighted_coupling_map(weighted_coupling_map):
    from qiskit.providers.models.backendproperties import Nduv, Gate
    from qiskit.providers.models import BackendProperties, BackendConfiguration
    from qiskit.providers import BackendV2
    import datetime
    curtime = datetime.datetime(3,8,10)
    # readout error of qubits
    num_physical_qubits = max([max(weighted_edge[:2]) for weighted_edge in weighted_coupling_map]) + 1
    qubits_info = [[Nduv(curtime,'readout_error','',0.0)] for i in range(num_physical_qubits)]
    # cx error of qubits
    gates_info = []
    for weighted_edge in weighted_coupling_map:
        gates_info.append(Gate(weighted_edge[:2],'cx',[Nduv(curtime,'gate_error','',weighted_edge[2])], name=f"cx_{weighted_edge[0]}_{weighted_edge[1]}"))
        gates_info.append(Gate(weighted_edge[:2][::-1],'cx',[Nduv(curtime,'gate_error','',weighted_edge[2])], name=f"cx_{weighted_edge[1]}_{weighted_edge[0]}"))
    backend = BackendProperties('QubitMapping','0.0.1',curtime,qubits_info,gates_info,[])
#     backend = BackendConfiguration(qubits=qubits_info, gates=gates_info)
    return backend

def Qiskit_experiments(structure, chiplet_array_dim, chiplet_size, data_qubit_num=None, benchmarks=['vqe','qft','qaoa','bv'], cross_link_sparsity=None, cross_chip_gate_weight = 7.4, iterations=1, mode='sabre'):
    if data_qubit_num is None:
        G=gen_chiplet_array(structure, chiplet_array_dim[0], chiplet_array_dim[1], chiplet_size[0], chiplet_size[1], cross_link_sparsity=1)
        gen_highway_layout(G)
        data_qubit_num = len(G.nodes) - len(get_highway_qubits(G))

    G=gen_chiplet_array(structure, chiplet_array_dim[0], chiplet_array_dim[1], chiplet_size[0], chiplet_size[1], cross_link_sparsity=cross_link_sparsity)

    qubit_idx_dict = gen_qubit_idx_dict(G)
    idx_qubit_dict = gen_idx_qubit_dict(G)
    regular_coupling = list([qubit_idx_dict[n1], qubit_idx_dict[n2]] for n1,n2 in G.edges)
    regular_coupling += list([qubit_idx_dict[n2], qubit_idx_dict[n1]] for n1,n2 in G.edges)

    link_err_rate = lambda connection_type: cross_chip_gate_weight * 1e-3 if connection_type == 'cross_chip' else 1e-3
    weighted_regular_coupling = [[edge[0],edge[1], link_err_rate(G.edges[(idx_qubit_dict[edge[0]], idx_qubit_dict[edge[1]])]['type'])] for edge in regular_coupling]
    backend = backend_from_weighted_coupling_map(weighted_regular_coupling)

    cross_chip_ratio = cross_chip_gate_weight
    benchmark_program_dict = {'vqe': genQiskitFullyConnectedCPhase, 'qft': genQiskitFullyConnectedCPhase, 'qaoa': genQiskitQAOA, 'bv': genQiskitBV}

    result = defaultdict(dict)
    for b in benchmarks:
        print('executing {}-qubit {} program on {}-qubit chiplets'.format(data_qubit_num, b, len(G.nodes)))
        if b == 'qft' and 'vqe' in benchmarks:
            continue

        include_qft = b == 'vqe' and 'qft' in benchmarks
        transpiled_depth = 0
        swap_decomposed_depth = 0
        cp_swap_decomposed_depth = 0
        within_chip_cnots = 0
        cross_chip_cnots = 0
        norm_cnots = 0
        within_chip_cnots_2 = 0
        cross_chip_cnots_2 = 0
        norm_cnots_2 = 0
        filter_function = lambda gate: gate.operation.num_qubits >= 2
        for i in range(iterations):
            # gen circuit
            qiskit_circuit = benchmark_program_dict[b](data_qubit_num, len(G.nodes))
            
            #optimization_level 2
            if mode == 'level_2':
                transpiled_circuit = transpile(qiskit_circuit, coupling_map=CouplingMap(regular_coupling), optimization_level=2)
            #optimization_level 3
            if mode == 'level_3':
                transpiled_circuit = transpile(qiskit_circuit, coupling_map=CouplingMap(regular_coupling), optimization_level=3)
            if mode == 'sabre':
                transpiled_circuit = PassManager([
                                            SabreLayout(coupling_map=CouplingMap(regular_coupling)),
                                            SabreSwap(coupling_map=CouplingMap(regular_coupling)),
                                            CXCancellation(),
                                            CommutativeCancellation(),
                                            CommutativeInverseCancellation()
                                            ]).run(qiskit_circuit)

            transpiled_depth += transpiled_circuit.depth()
            swap_decomposed_circuit = transpiled_circuit.decompose('swap')
            swap_decomposed_depth += swap_decomposed_circuit.depth(filter_function)
            
            w, c, n = count_norm_cnots(G, swap_decomposed_circuit, cross_chip_ratio)
            within_chip_cnots += w
            cross_chip_cnots += c
            norm_cnots += n
            
            if include_qft:
                cp_swap_decomposed_circuit = swap_decomposed_circuit.decompose('cp')
                cp_swap_decomposed_depth += cp_swap_decomposed_circuit.depth(filter_function)

                w2, c2, n2 = count_norm_cnots(G, cp_swap_decomposed_circuit, cross_chip_ratio)
                within_chip_cnots_2 += w2
                cross_chip_cnots_2 += c2
                norm_cnots_2 += n2
        result[b] = {'depth': swap_decomposed_depth/iterations, 'eff_gate_num': norm_cnots/iterations, 'on-chip': within_chip_cnots/iterations, 'cross-chip': cross_chip_cnots/iterations, 
                     'cross_chip_gate_weight':cross_chip_gate_weight, 'sparisity':cross_link_sparsity, 'qubit_num':data_qubit_num, 'iterations':iterations, 'mode':mode}

        print('avg: decomposed_depth = {}, within_chip_cnots={}, cross_chip_cnots={}, norm_cnots = {}'.format(swap_decomposed_depth/iterations, within_chip_cnots/iterations, cross_chip_cnots/iterations, norm_cnots/iterations))

        if include_qft:
            print('avg: decomposed_depth = {}, within_chip_cnots={}, cross_chip_cnots={}, norm_cnots = {}'.format(cp_swap_decomposed_depth/iterations, within_chip_cnots_2/iterations, cross_chip_cnots_2/iterations, norm_cnots_2/iterations))
            result['qft'] = {'depth': cp_swap_decomposed_depth/iterations, 'eff_gate_num': norm_cnots_2/iterations, 'on-chip': within_chip_cnots_2/iterations, 'cross-chip': cross_chip_cnots_2/iterations, 
                     'cross_chip_gate_weight':cross_chip_gate_weight, 'sparisity':cross_link_sparsity, 'qubit_num':data_qubit_num, 'iterations':iterations, 'mode':mode}
    return result
