from collections import defaultdict, deque
from copy import deepcopy
from random import shuffle
from numpy import *


class OpNode:
    def __init__(self, *qargs):
        self.depth = None
        self.qargs = qargs
        self.qubit_num = len(qargs)
        assert self.qubit_num <=2, 'OpNodes should not have more than 2 qubits'
        if self.qubit_num == 1:
            self.q = self.qargs[0]
        elif self.qubit_num == 2:
            self.control, self.target = self.qargs
    def get_the_other_qubit(self, qubit):
        if self.control == qubit:
            return self.target
        else:
            assert self.target == qubit, "Qubit {} is not in {}".format(qubit, self)
            return self.control
    def __repr__(self):
        if self.qubit_num == 1:
            return ' G{} '.format(self.q)
        else:
            return 'C{},{}'.format(self.qargs[0], ','.join([str(i) for i in self.qargs[1:]]))

class MOpNode:
    def __init__(self, *qargs):
        self.depth = None
        self.qargs = list(qargs)
        self.qubit_num = len(qargs)
        self.shared, self.indiv = self.qargs[0], list(self.qargs[1:])
    def add_indiv(self, qubit):
        self.qargs.append(qubit)
        self.indiv.append(qubit)
    def remove_indiv(self, qubit):
        self.qargs.remove(qubit)
        self.indiv.remove(qubit)
    def __repr__(self):
        repr_str = 'M{}{}'.format(self.qargs[0], ''.join([str(i) for i in self.qargs[1:3]]))
        if len(self.qargs)>3:
            repr_str += ':'
        return repr_str[:5]

class Circuit: 
    def __init__(self, qubit_num):
        self.qubit_num = qubit_num
        self.circuit_lines = defaultdict(list) # each line contains a list of (node, role)
    def __copy__(self):
        new_instance = copy.deepcopy(self)
        return new_instance
    
    def get_line_depth(self, line):
        return len(self.circuit_lines[line])
    
    def take_node_with_role(self, circuit_line, idx):
        return circuit_line[idx] if idx < len(circuit_line) else None
    
    def take_node(self, line, idx):
        node_with_role = self.take_node_with_role(self.circuit_lines[line], idx)
        return node_with_role[0] if node_with_role is not None else None  
    
    def take_role(self, line, idx):
        node_with_role = self.take_node_with_role(self.circuit_lines[line], idx)
        return node_with_role[1] if node_with_role is not None else None  
    
    def is_position_empty(self, line, idx):
        return idx > self.get_line_depth(line) - 1 or self.take_node(line, idx) is None
    
    def add_node_with_role(self, line, idx, node, role):
        assert idx >= 0, "idx should >= 0"
        assert self.is_position_empty(line, idx), "Failed to add {} to ({},{}) because the position is already taken".format(node, line, idx)
        node.depth = idx + 1
        for i in range(self.get_line_depth(line), idx+1):
            self.circuit_lines[line].append(None)
        self.circuit_lines[line][idx] = (node, role)

    def delete_node_with_role(self, line, idx):
        self.circuit_lines[line][idx] = None
        while len(self.circuit_lines[line]) > 0 and self.circuit_lines[line][-1] is None:
            self.circuit_lines[line].pop()

    def is_node_the_same_gate(self, node, control, target):
        if isinstance(node, OpNode):
            if node.qubit_num == 1:
                return False
            else:
                return node.control == control and node.target == target
        return False
    
    def is_component_contained_in_mop(self, mop, control, target):
        if isinstance(mop, MOpNode):
            return mop.shared == control and target in mop.indiv
        return False
    
    def remove_control_target_at_index(self, control, target, idx):
        c_node = self.take_node(control, idx)
        if self.is_node_the_same_gate(c_node, control, target):
            self.delete_node_with_role(control, idx)
            self.delete_node_with_role(target, idx)
        else:
            assert self.is_component_contained_in_mop(c_node, control, target), 'Auto cancellation failed.'
            c_node.remove_indiv(target)
            self.delete_node_with_role(target, idx)
            if not c_node.indiv:
                self.delete_node_with_role(control, idx)

    def get_earliest_index_for_2qubit_op(self, op, auto_commuting, auto_cancellation, min_idx=0, max_idx=None, addition_condition=lambda idx: True):
        control, target = op.control, op.target
        if max_idx is None:
            max_idx = max(self.get_line_depth(control), self.get_line_depth(target))
        
        earliest_idx = -1
        for idx in range(max_idx, min_idx-1, -1):
            c_node, c_role = self.take_node(control, idx), self.take_role(control, idx)
            t_node, t_role = self.take_node(target, idx), self.take_role(target, idx)
                
            if self.is_node_the_same_gate(c_node, control, target) or self.is_component_contained_in_mop(c_node, control, target):
                if auto_cancellation:
                    return idx
                elif self.is_node_the_same_gate(c_node, control, target):
                    return earliest_idx
                
            if c_node is None and t_node is None and addition_condition(idx):
                earliest_idx = idx
            elif not auto_commuting or c_role not in {None, 'c', 'mc'} or t_role not in {None, 't', 'mt'}:
                break # not commutable
        return earliest_idx
    
    def get_earliest_index_for_2qubit_component(self, op, auto_commuting, auto_cancellation, min_idx=0, max_idx=None, addition_condition=lambda idx: True):
        control, target = op.control, op.target
        if max_idx is None:
            max_idx = max(self.get_line_depth(control), self.get_line_depth(target))
        
        earliest_idx = -1
        for idx in range(max_idx, min_idx-1, -1):
            c_node, c_role = self.take_node(control, idx), self.take_role(control, idx)
            t_node, t_role = self.take_node(target, idx), self.take_role(target, idx)
            if self.is_node_the_same_gate(c_node, control, target) or self.is_component_contained_in_mop(c_node, control, target):
                if auto_cancellation:
                    return idx
                elif c_role == 'mc':
                    return earliest_idx
            
            if c_node is None and t_node is None and addition_condition(idx):
                earliest_idx = idx
            elif c_role == 'mc' and t_node is None and addition_condition(idx):
                if not auto_commuting:
                    return idx
                else:
                    earliest_idx = idx
            elif not auto_commuting or c_role not in {None, 'c', 'mc'} or t_role not in {None, 't', 'mt'}:
                break # not commutable
        return earliest_idx
        
    @property
    def depth(self):
        if not self.circuit_lines:
            return 0
        return max([len(circuit_line) for circuit_line in self.circuit_lines.values()])
    
    def get_mop_depth(self, line):
        idx = -1
        for idx in range(self.get_line_depth(line)-1, -1, -1):
            if self.take_role(line, idx) in {'mc', 'mt'}:
                break
        return idx + 1

    @property
    def mop_depth(self):
        return max([self.get_mop_depth(line) for line in range(self.qubit_num)])

    
    def add_1qubit_op(self, op, depth=None):
        if depth is not None:
            idx = depth - 1
        else:
            idx = self.get_line_depth(op.q)
        self.add_node_with_role(op.q, idx, op, 'q')


    def add_2qubit_op(self, op, depth=None, auto_commuting=False, auto_cancellation=False, min_idx=0, max_idx=None):
        if depth is None:
            earliest_idx = self.get_earliest_index_for_2qubit_op(op, auto_commuting=auto_commuting, auto_cancellation=auto_cancellation, min_idx=min_idx, max_idx=max_idx)
        else:
            earliest_idx = depth - 1

        if auto_cancellation:
            c_node = self.take_node(op.control, earliest_idx)
            if self.is_node_the_same_gate(c_node, op.control, op.target) or self.is_component_contained_in_mop(c_node, op.control, op.target):
                self.remove_control_target_at_index(op.control, op.target, earliest_idx)
                return
            
        self.add_node_with_role(op.control, earliest_idx, op, 'c')
        self.add_node_with_role(op.target, earliest_idx, op, 't')
            

    def add_mqubit_op(self, mop, depth=None):
        if depth is None:
            depth = max([self.get_line_depth(qubit) for qubit in mop.qargs]) + 1
        self.add_node_with_role(mop.shared, depth-1, mop, 'mc')
        for line in mop.indiv:
            self.add_node_with_role(line, depth-1, mop, 'mt')

    def add_mop_2qubit_component(self, op, depth=None, auto_commuting=False, auto_cancellation=False):
        if depth is None:
            earliest_index = self.get_earliest_index_for_2qubit_component(op, auto_commuting=auto_commuting, auto_cancellation=auto_cancellation)
            earliest_depth = earliest_index + 1
        else: 
            earliest_depth = depth

        if auto_cancellation:
            c_node = self.take_node(op.control, earliest_depth - 1)
            if self.is_node_the_same_gate(c_node, op.control, op.target) or self.is_component_contained_in_mop(c_node, op.control, op.target):
                self.remove_control_target_at_index(op.control, op.target, earliest_depth - 1)
                return

        if self.is_position_empty(op.control, earliest_depth-1) and self.is_position_empty(op.target, earliest_depth-1):
            self.add_mqubit_op(MOpNode(op.control, op.target), earliest_depth)
        else:
            mop = self.take_node(op.control, earliest_depth-1)
            assert isinstance(mop, MOpNode), "Failed to update {} with {} because it is not a MOpNode".format(mop, op)
            assert mop.shared == op.control, "Failed to update {} with {} because their control qubits are different {}".format(mop, op)
            mop.add_indiv(op.target)
            self.add_node_with_role(op.target, earliest_depth-1, mop, 'mt')


    def __repr__(self):
        def pad(s):
            return s[:8].ljust(8)
        def with_hlines(s):
            return pad('-'+s+'-')
        def with_control_style(s, color):
            fg = lambda text, color: "\033[0;3{}m{}\033[0m".format(color,text)
            return fg(pad('-'+s+'-'), color)
        def with_target_style(s, color):
            fg = lambda text, color: "\033[4;3{}m{}\033[0m".format(color,text)
            return fg(pad('|'+s+'|'),color)
        def with_control_multi_target_style(s, color, sym='|'):
            bg = lambda text, color: "\033[0;30;4{}m{}\033[0m".format(color,text)
            return bg(pad(sym+s+sym),color)
        def get_node_repr_on_line(op, line):
            if op is None:
                return with_hlines(pad(' '))
            if isinstance(op, OpNode):
                if op.qubit_num == 2:
                    if op.control == line:
                        op_repr = with_control_style(op.__repr__(), (op.control+op.target)%6+1)
                    elif op.target == line:
                        op_repr = with_target_style(op.__repr__(), (op.control+op.target)%6+1)
                else:
                    op_repr = with_hlines(op.__repr__())
                return op_repr
            if isinstance(op, MOpNode):
                if op.shared == line:
                    op_repr = with_control_multi_target_style(op.__repr__(), op.shared%6+1, '-')
                else:
                    op_repr = with_control_multi_target_style(op.__repr__(), op.shared%6+1)
                return op_repr
                
        represened_lines = []
        if not self.circuit_lines:
            max_len = 0
        else:
            max_len = max([len(circuit_line) for circuit_line in self.circuit_lines.values()])
        for line in range(self.qubit_num):
            circuit_line_len = len(self.circuit_lines[line])
            l=[get_node_repr_on_line(self.take_node(line, i), line) for i in range(circuit_line_len)] + [' ' for i in range(max_len - circuit_line_len)]
            represened_lines.append(l)
        
        number_each_line = 9
        string_blocks = []
        print(max_len)
        for i in range((max_len - 1)//number_each_line + 1):
            a,b=i * number_each_line, (i+1) * number_each_line
            depth_info = 'slice\t' + '  '.join([pad(str(n)) for n in range(a, b)]) + '\n'
            this_block = [str(line) + '-->\t' + '  '.join(represened_lines[line][a: min(b, len(self.circuit_lines[line]))]) for line in range(len(represened_lines))]
            string_blocks.append(depth_info + '\n'.join(this_block))
            
        circuit_repr = ('\n'+'-'*100+'\n').join(string_blocks)
        if len(circuit_repr)>500000:
            print('OUT OF LIMIT: only part of the circuit is represented')
        return circuit_repr[:500000]

        