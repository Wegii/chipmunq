from collections import Counter, defaultdict
from random import sample
from Circuit import *
from Chiplet import *
from HighwayOccupancy import *
from Router import *

def VQE(data_qubit_num):
    program = []
    for i in range(data_qubit_num-1):
        control_block = ControlBlock(i, Counter(list(range(i+1, data_qubit_num))))
        program.append(control_block)
    return program

def QFT(data_qubit_num):
    program = []
    for i in range(data_qubit_num-1):
        control_block = ControlBlock(i, Counter(list(range(i+1, data_qubit_num))*2))
        program.append(control_block)
    return program

def QAOA(data_qubit_num, percentage=0.5):
    all_possible_gates = [(i,j) for i in range(data_qubit_num) for j in range(i+1, data_qubit_num)]
    gates = list(sample(all_possible_gates, k=int(len(all_possible_gates) * percentage)))
    gates.sort()
    print('{}/{} gates are selected: {}...{}'.format(len(gates), len(all_possible_gates), gates[:10], gates[-10:]))

    program = []
    control_target_dict = defaultdict(list)
    for control, target in gates:
        control_target_dict[control].append(target)
    for control, target_list in control_target_dict.items():
        control_block = ControlBlock(control, Counter(target_list*2))
        program.append(control_block)
    return program

def BV(data_qubit_num, percentage=0.5):
    all_possible_gates = [(0,i) for i in range(1, data_qubit_num)]
    gates = list(sample(all_possible_gates, k=int(len(all_possible_gates)*percentage)))
    gates.sort()
    
    program = []
    control_target_dict = defaultdict(list)
    for control, target in gates:
        control_target_dict[control].append(target)
    for control, target_list  in control_target_dict.items():
        control_block = ControlBlock(control, Counter(target_list))
        program.append(control_block)
    return program
    
def MECH_experiments(structure, chiplet_array_dim, chiplet_size, custom_highway_qubits=None, custom_highway_edges=[], benchmarks=['vqe','qft','qaoa','bv'], cross_link_sparsity=1, prep_period=None, meas_period=2, cross_chip_gate_weight = 7.4, meas_weight = 2.2):
    G=gen_chiplet_array(structure, chiplet_array_dim[0], chiplet_array_dim[1], chiplet_size[0], chiplet_size[1], cross_link_sparsity=cross_link_sparsity)
    if prep_period is None:
        if structure in ['square', 'hexagon']:
            prep_period = 13
        if structure in ['heavy_square', 'heavy_hexagon']:
            prep_period = 15

    if custom_highway_qubits is not None:
        custom_highway_layout(G, custom_highway_qubits, custom_highway_edges)
    else:
        gen_highway_layout(G)

    data_qubit_num = len(G.nodes) - len(get_highway_qubits(G))
    benchmark_program_dict = {'vqe': VQE(data_qubit_num), 'qft': QFT(data_qubit_num), 'qaoa': QAOA(data_qubit_num), 'bv': BV(data_qubit_num)}
    result = defaultdict(dict)

    for b in benchmarks:
        print('-'*16+'executing ', b)
        program = benchmark_program_dict[b]
        router = Router(G, prep_period=prep_period, meas_period=meas_period)

        for i, control_block in enumerate(program):
            if i%10 == 0:
                print('{}/{}'.format(i, len(program)))
            router.execute_control_multi_target_block(control_block, cross_chip_gate_weight, cross_chip_overhead = 3)

        for shuttle_idx in range(len(router.highway_manager.shuttle_stack)):
            for idx in range(router.highway_manager.get_shuttle_prep_start_time(shuttle_idx), router.highway_manager.get_shuttle_exec_start_time(shuttle_idx)):
                for line in range(len(router.circuit.circuit_lines)):
                    assert router.circuit.is_position_empty(line, idx)
            source_measured_qubits_dict = router.highway_manager.bridge_throughout_highway(router.circuit, shuttle_idx)
            router.highway_manager.reentangle_throughout_highway(router.circuit, shuttle_idx, source_measured_qubits_dict)

        for shuttle_idx in range(len(router.highway_manager.shuttle_stack)):
            router.highway_manager.measure_throughout_highway(router.circuit, shuttle_idx)
            

        on_chip_gate_num = 0
        cross_chip_gate_num = 0
        meas_num = 0

        for idx in range(router.circuit.depth):
            for line in range(len(router.circuit.circuit_lines)):
                if router.circuit.take_role(line, idx) == 'q':
                    meas_num += 1
                if router.circuit.take_role(line, idx) in ['t', 'mt']:
                    node = router.circuit.take_node(line, idx)
                    if isinstance(node,  OpNode):
                        control_line = node.control
                    if isinstance(node, MOpNode):
                        control_line = node.shared
                    control_qubit, target_qubit = router.highway_manager.idx_qubit_dict[control_line], router.highway_manager.idx_qubit_dict[line]
                    
                    if router.chip.has_edge(control_qubit, target_qubit) and router.chip.edges[(control_qubit, target_qubit)]['type'] == 'cross_chip':
                        cross_chip_gate_num += 1
                    else:
                        on_chip_gate_num += 1

        eff_gate_num = on_chip_gate_num + cross_chip_gate_num * cross_chip_gate_weight + meas_num * meas_weight
        result[b] = {'depth': router.circuit.depth, 'eff_gate_num': eff_gate_num, 'on-chip': on_chip_gate_num, 'cross-chip': cross_chip_gate_num, 'meas_num': meas_num, 'weight':(cross_chip_gate_weight, meas_weight), 
                     'periods':(prep_period, meas_period), 'shuttle_num': len(router.highway_manager.shuttle_stack), 'sparsity':cross_link_sparsity}

    return result