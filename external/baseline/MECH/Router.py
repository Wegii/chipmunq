from collections import Counter
from Chiplet import *
from Circuit import *
from HighwayOccupancy import *

class ControlBlock:
    def __init__(self, control, target_counter=Counter()):
        self.control = control
        self.target_counter = target_counter

class Router:
    def __init__(self, chip, initial_v2p=None, circuit=None, highway_manager=None, prep_period=2, meas_period=2):
        self.chip = chip
        self.circuit = Circuit(len(chip.nodes)) if circuit is None else circuit
        self.highway_manager = HighwayManager(chip, prep_period, meas_period) if highway_manager is None else highway_manager
        self._v2p = self.highway_manager.idx_qubit_dict.copy() if initial_v2p is None else initial_v2p
        self._p2v = {p:v for v,p in self._v2p.items()}
        self.draw_figures = False

    def swap(self, p1, p2):
        line1, line2 = self.highway_manager.qubit_idx_dict[p1], self.highway_manager.qubit_idx_dict[p2]
        self.highway_manager.execute_on_local(self.circuit, OpNode(line1, line2), auto_commuting=True, auto_cancellation=False)
        self.highway_manager.execute_on_local(self.circuit, OpNode(line2, line1), auto_commuting=True, auto_cancellation=False)
        self.highway_manager.execute_on_local(self.circuit, OpNode(line1, line2), auto_commuting=True, auto_cancellation=False)
        v1, v2 = self._p2v[p1], self._p2v[p2]
        self._p2v[p1], self._p2v[p2] = v2, v1
        self._v2p[v1], self._v2p[v2] = p2, p1

    def swap_along_path_unless_encoutering_control(self, path, p_control=None):
        blocked_by_p_control = False
        for step in range(1, len(path)):
            from_qubit, to_qubit = path[step-1], path[step]
            if to_qubit == p_control:
                blocked_by_p_control = True
                break
            self.swap(from_qubit, to_qubit)
        return blocked_by_p_control

    def path_cost(self, path, cross_chip_gate_weight, cross_chip_overhead = 3):
        arrival_depth = 0
        num_of_cross_chip_links = 0
        for step in range(1, len(path)):
            from_qubit, to_qubit = path[step-1], path[step]
            if self.chip.edges[(from_qubit, to_qubit)]['type'] == 'cross_chip':
                num_of_cross_chip_links += 1
            line1, line2 = self.highway_manager.qubit_idx_dict[from_qubit], self.highway_manager.qubit_idx_dict[to_qubit]
            cur_depth = max(self.circuit.get_line_depth(line1), self.circuit.get_line_depth(line2))
            arrival_depth = max(arrival_depth + 3, cur_depth + 3)
        return arrival_depth + 3 * num_of_cross_chip_links * (cross_chip_gate_weight - 1) * cross_chip_overhead

    def execute_control_multi_target_block(self, control_block, cross_chip_gate_weight, cross_chip_overhead = 3):
        v_control, v_targets_dict = control_block.control, control_block.target_counter
        remaining_v_targets = set(v_targets_dict.keys())

        # move control qubit to highway
        p_control = self._v2p[v_control]
        if get_distance_to_highway(self.chip, p_control) > 1:
            entrances_with_paths = find_possible_entrances_with_paths(self.chip, p_control)
            best_entrance, best_path = min(entrances_with_paths, key=lambda entrances_with_path: self.path_cost(entrances_with_path[1], cross_chip_gate_weight, cross_chip_overhead))
            self.swap_along_path_unless_encoutering_control(best_path)
            assert get_distance_to_highway(self.chip, self._v2p[v_control])

        # move target qubits to highway from near to far
        blocked_by_p_control = False
        while remaining_v_targets:
            nearest_target = min(remaining_v_targets, key=lambda v_target: get_distance_to_highway(self.chip, self._v2p[v_target]))
            p_control, p_nearest_target = self._v2p[v_control], self._v2p[nearest_target]
            if not is_qubit_next_to_highway(self.chip, p_nearest_target): #TODO: exclude the highway entrance occupied by control
                entrances_with_paths = find_possible_entrances_with_paths(self.chip, p_nearest_target) #!TODO: don't occupy the position of control qubit!
                best_entrance, best_path = min(entrances_with_paths, key=lambda entrances_with_path: self.path_cost(entrances_with_path[1], cross_chip_gate_weight, cross_chip_overhead))
                blocked_by_p_control = self.swap_along_path_unless_encoutering_control(best_path, p_control)
                
                assert self._v2p[v_control] == p_control
                p_nearest_target = best_path[-1]
                assert is_qubit_next_to_highway(self.chip, p_nearest_target) or are_regularly_connected(self.chip, p_control, p_nearest_target)
            p_control, p_nearest_target = self._v2p[v_control], self._v2p[nearest_target]
            line1, line2 = self.highway_manager.qubit_idx_dict[p_control], self.highway_manager.qubit_idx_dict[p_nearest_target]
            
            if blocked_by_p_control:
                for i in range(v_targets_dict[nearest_target]):
                    self.highway_manager.execute_on_local(self.circuit, OpNode(line1, line2))
            else:
                for i in range(v_targets_dict[nearest_target]):
                    self.highway_manager.execute_on_highway(self.circuit, OpNode(line1, line2))
            remaining_v_targets.remove(nearest_target)

