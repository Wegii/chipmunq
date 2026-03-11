from qiskit.providers import BackendV2
import random

random.seed(123)

import numpy as np
from typing import Optional, Dict, Set, Tuple, List, Union
import stim


def get_noise_model(error_type: str, qt=None, p: float = None, backend=None, remote=None):
    """Generate circuit level noise.

    Code adapted from https://github.com/aswierkowska/ECCentric/blob/main/noise/utils.py

    Args:
        error_type (str): _description_
        p (float, optional): _description_. Defaults to None.
        remote (_type_, optional): _description_. Defaults to None.

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """

    if p:
        if error_type == "modsi1000":
            return NoiseModel(
                sq=p / 10,
                tq=p,
                idle=p / 10,
                measure=2 * p,
                reset=2 * p,
                qt=qt,
                remote=remote,
                noisy_gates={
                    "CZ": p,
                    "R": 2 * p,
                    "M": 5 * p,
                },
            )

    raise NotImplementedError


SQ_OPS = {"C_XYZ", "C_ZYX", "H", "H_YZ", "I", "X"}
TQ_OPS = {"CX", "CY", "CZ", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ"}
RESET_OPS = {"R", "RX", "RY"}
MEASURE_OPS = {"M", "MX", "MY"}
ANNOTATION_OPS = {"OBSERVABLE_INCLUDE", "DETECTOR", "SHIFT_COORDS", "QUBIT_COORDS", "TICK"}
SWAP_OPS = {"SWAP"}


class NoiseModel:
    """Noise model based on: https://github.com/Strilanc/honeycomb_threshold/blob/main/src/noise.py"""

    def __init__(
        self,
        sq: Optional[float] = 0,
        tq: Optional[float] = 0,
        idle: Optional[float] = 0,
        crosstalk: Optional[float] = 0,
        leakage: Optional[float] = 0,
        leakage_propagation: Optional[float] = 0,
        reset: Optional[float] = 0,
        measure: Optional[float] = 0,
        shuttle: Optional[float] = 0,
        remote: Optional[float] = 0,
        noisy_gates: Dict[str, float] = {},
        gate_times: Optional[Dict[str, float]] = {},
        qt=None,
        backend: Optional[BackendV2] = None,
        use_correlated_parity_measurement_errors: bool = False,
    ):
        self.sq = sq
        self.tq = tq
        self.idle = idle
        self.crosstalk = crosstalk
        self.leakage = leakage
        self.leakage_propagation = leakage_propagation if leakage_propagation != 0 else leakage
        self.reset = reset
        self.measure = measure
        self.shuttle = shuttle
        # Remote can either be one single number, or dictionary containing remote pairs and their noise level. If remote
        # is a single number, and no qubit tracking is provided, no remote connections are taken into consideration
        self.remote = remote
        self.noisy_gates = noisy_gates
        self.gate_times = gate_times
        # If there is no qubit tracking provided, it is still possible to utilize remote_gates functionality. For this
        # provided the remote_gates and their noise strength in the self.remote property
        self.qt = qt
        self.backend = backend
        self.use_correlated_parity_measurement_errors = use_correlated_parity_measurement_errors

    def add_qubit_error(self, circuit: stim.Circuit, qubits: List[stim.GateTarget], gate_duration: float) -> None:
        # https://arxiv.org/pdf/1404.3747

        if self.backend == None or self.idle != 0:
            return
        for qubit in qubits:
            qubit_properties = self.backend.qubit_properties(qubit.value)
            t1 = qubit_properties.t1
            t2 = qubit_properties.t2

            # Safety check
            if t1 <= 0 or t2 <= 0:
                continue

            p_x = 0.25 * (1 - np.exp(-gate_duration / t1))
            p_y = 0.25 * (1 - np.exp(-gate_duration / t1))
            p_z = (1 - np.exp(-gate_duration / t2)) / 2 - (1 - np.exp(-gate_duration / t1)) / 4

            p_x = np.clip(p_x, 0.0, 1.0)
            p_y = np.clip(p_y, 0.0, 1.0)
            p_z = np.clip(p_z, 0.0, 1.0)

            circuit.append_operation("PAULI_CHANNEL_1", [qubit], [p_x, p_y, p_z])

    def add_crosstalk_errors(self, touched_qubits: set[int], post: stim.Circuit):
        if self.backend == None:
            return
        # TODO: could be enhanced by correlated errors
        already_noised = set()
        for q in touched_qubits:
            for neighbor in self.qt.get_neighbours(q):
                if (
                    neighbor in touched_qubits
                    and (neighbor, q) not in already_noised
                    and (q, neighbor) not in already_noised
                ):
                    noise_op = "X_ERROR" if random.random() < 0.5 else "Z_ERROR"
                    post.append_operation(noise_op, [stim.target_qubit(neighbor)], self.crosstalk)
                    already_noised.add((q, neighbor))
                    already_noised.add((neighbor, q))

    def propagate_leakage(self, circuit: stim.Circuit, pair: List[int]):
        if any(self.qt.check_leaked(q) for q in pair) and random.random() < self.leakage_propagation:
            for q in pair:
                if not self.qt.check_leaked(q):
                    pauli = random.choice(["X_ERROR", "Y_ERROR", "Z_ERROR"])
                    circuit.append_operation(pauli, [q], 1.0)
                    self.qt.leak_qubit(q)

    # TODO: add targets type
    def add_leakage_errors(self, circuit: stim.Circuit, targets):
        qubit_targets = [t for t in targets if t.is_qubit_target]
        for target in qubit_targets:
            if random.random() < self.leakage:
                pauli = random.choice(["X_ERROR", "Y_ERROR", "Z_ERROR"])
                circuit.append_operation(pauli, [target], 1.0)
                self.qt.leak_qubit(target.value)

    def is_remote(self, pair: List[int]) -> bool:
        # if self.qt == None or self.backend == None:
        #    return False

        if self.qt == None:
            if self.remote == None or isinstance(self.remote, (int, float)):
                return False
            else:
                # pair is a list, but remote needs this as pair
                return tuple(pair) in self.remote

        phy_q1 = self.qt.get_layout_postion(pair[0])
        phy_q2 = self.qt.get_layout_postion(pair[1])
        if (phy_q1, phy_q2) in self.backend.get_remote_gates or (phy_q2, phy_q1) in self.backend.get_remote_gates:
            return True
        else:
            return False

    def get_gate_time(self, op: stim.CircuitInstruction, pair: Optional[List[int]] = None) -> Union[float, None]:
        if self.gate_times == {}:
            return 0

        if pair:
            if self.backend and self.is_remote(pair):
                if self.backend.name == "FakeQuantinuumApollo" or self.backend.name == "FakeInfleqtion":
                    return self.gate_times["REMOTE"] * self.qt.get_euclidian_distance(pair[0], pair[1])
                else:
                    return self.gate_times["REMOTE"]
            else:
                return self.gate_times["TQ"]

        if op.name in SQ_OPS:
            return self.gate_times["SQ"]
        elif op.name in RESET_OPS:
            return self.gate_times["R"]
        elif op.name in MEASURE_OPS:
            return self.gate_times["M"]
        raise NotImplementedError(f"Gate time not defined for op: {repr(op)}")

    def noisy_op(self, op: stim.CircuitInstruction, ancilla: int) -> Tuple[stim.Circuit, stim.Circuit, stim.Circuit]:
        pre = stim.Circuit()
        mid = stim.Circuit()
        post = stim.Circuit()
        targets = op.targets_copy()
        args = op.gate_args_copy()

        if self.leakage > 0:
            self.add_leakage_errors(post, targets)

        if op.name in SQ_OPS:
            if op.name in self.noisy_gates:
                post.append_operation("DEPOLARIZE1", targets, self.noisy_gates[op.name])
            elif self.sq != 0:
                post.append_operation("DEPOLARIZE1", targets, self.sq)
            self.add_qubit_error(post, targets, self.get_gate_time(op))
        elif op.name in TQ_OPS or op.name in SWAP_OPS:
            if op.name in self.noisy_gates:
                p = self.noisy_gates[op.name]
            else:
                p = self.tq
            for i in range(0, len(targets), 2):
                pair = [targets[i].value, targets[i + 1].value]
                if self.leakage > 0:
                    self.propagate_leakage(post, pair)
                if self.is_remote(pair):
                    if self.shuttle:
                        post.append_operation("DEPOLARIZE2", pair, self.shuttle)
                    elif self.remote:
                        if isinstance(self.remote, (int, float)):
                            post.append_operation("DEPOLARIZE2", pair, self.remote)
                        else:
                            # Get strength based on the noise level of this pair
                            noise = self.remote[tuple(pair)]
                            post.append_operation("DEPOLARIZE2", pair, noise)
                    else:
                        post.append_operation("DEPOLARIZE2", pair, p)
                else:
                    post.append_operation("DEPOLARIZE2", pair, p)
                self.add_qubit_error(post, targets, self.get_gate_time(op, pair))
        elif op.name in RESET_OPS:
            for q in targets:
                if self.qt != None:
                    self.qt.reset_qubit(q.value)
            if op.name in self.noisy_gates:
                post.append_operation(
                    "Z_ERROR" if op.name.endswith("X") else "X_ERROR", targets, self.noisy_gates[op.name]
                )
            elif self.reset != 0:
                post.append_operation("Z_ERROR" if op.name.endswith("X") else "X_ERROR", targets, self.reset)
            self.add_qubit_error(post, targets, self.get_gate_time(op))
        elif op.name in MEASURE_OPS:
            if op.name in self.noisy_gates:
                p = self.noisy_gates[op.name]
            else:
                p = self.measure
            pre.append_operation("Z_ERROR" if op.name.endswith("X") else "X_ERROR", targets, p)
            # self.add_qubit_error(post, targets, self.get_gate_time(op))
        elif op.name == "MPP":
            # Our circuits never contain MPP after translations
            assert len(targets) % 3 == 0 and all(t.is_combiner for t in targets[1::3]), repr(op)
            assert args == [] or args == [0]
            if op.name in self.noisy_gates:
                p = self.noisy_gates[op.name]
            else:
                p = self.measure

            if self.use_correlated_parity_measurement_errors:
                for k in range(0, len(targets), 3):
                    mid += parity_measurement_with_correlated_measurement_noise(
                        t1=targets[k], t2=targets[k + 2], ancilla=ancilla, mix_probability=p
                    )
                return pre, mid, post

            else:
                pre.append_operation("DEPOLARIZE2", [t.value for t in targets if not t.is_combiner], p)
                args = [p]
        mid.append_operation(op.name, targets, args)
        return pre, mid, post

    def noisy_circuit(self, circuit: stim.Circuit, *, qs: Optional[Set[int]] = None) -> stim.Circuit:
        result = stim.Circuit()
        ancilla = circuit.num_qubits

        current_moment_pre = stim.Circuit()
        current_moment_mid = stim.Circuit()
        current_moment_post = stim.Circuit()
        used_qubits: Set[int] = set()
        measured_qubits: Set[int] = set()
        reset_qubits: Set[int] = set()

        if qs is None:
            qs = set(range(circuit.num_qubits))

        def flush():
            nonlocal result
            if not current_moment_mid and self.idle == 0:
                return

            idle_qubits = sorted(qs - used_qubits)
            if idle_qubits and self.idle > 0:
                current_moment_post.append_operation("DEPOLARIZE1", idle_qubits, self.idle)

            result += current_moment_pre
            result += current_moment_mid
            result += current_moment_post
            used_qubits.clear()
            current_moment_pre.clear()
            current_moment_mid.clear()
            current_moment_post.clear()
            measured_qubits.clear()
            reset_qubits.clear()

        for op in circuit:
            if isinstance(op, stim.CircuitRepeatBlock):
                flush()
                result += self.noisy_circuit(op.body_copy(), qs=qs) * op.repeat_count
            elif isinstance(op, stim.CircuitInstruction):
                if op.name == "TICK":
                    flush()
                    result.append_operation("TICK", [])
                    continue

                if op.name in SWAP_OPS and self.qt != None:
                    self.qt.update_stim_swaps(op)

                pre, mid, post = self.noisy_op(op, ancilla)
                current_moment_pre += pre
                current_moment_mid += mid
                current_moment_post += post

                touched_qubits = {
                    t.value
                    for t in op.targets_copy()
                    if t.is_x_target or t.is_y_target or t.is_z_target or t.is_qubit_target
                }
                if op.name in ANNOTATION_OPS:
                    touched_qubits.clear()

                if self.crosstalk > 0:
                    self.add_crosstalk_errors(touched_qubits, post)

                used_qubits |= touched_qubits
                if op.name in MEASURE_OPS:
                    measured_qubits |= touched_qubits
                if op.name in RESET_OPS:
                    reset_qubits |= touched_qubits
            else:
                raise NotImplementedError(repr(op))
        flush()
        return result


def mix_probability_to_independent_component_probability(mix_probability: float, n: float) -> float:
    return 0.5 - 0.5 * (1 - mix_probability) ** (1 / 2 ** (n - 1))


def parity_measurement_with_correlated_measurement_noise(
    *, t1: stim.GateTarget, t2: stim.GateTarget, ancilla: int, mix_probability: float
) -> stim.Circuit:

    ind_p = mix_probability_to_independent_component_probability(mix_probability, 5)

    # Generate all possible combinations of (non-identity) channels.  Assumes triple of targets
    # with last element corresponding to measure qubit.
    circuit = stim.Circuit()
    circuit.append_operation("R", [ancilla])
    if t1.is_x_target:
        circuit.append_operation("XCX", [t1.value, ancilla])
    if t1.is_y_target:
        circuit.append_operation("YCX", [t1.value, ancilla])
    if t1.is_z_target:
        circuit.append_operation("ZCX", [t1.value, ancilla])
    if t2.is_x_target:
        circuit.append_operation("XCX", [t2.value, ancilla])
    if t2.is_y_target:
        circuit.append_operation("YCX", [t2.value, ancilla])
    if t2.is_z_target:
        circuit.append_operation("ZCX", [t2.value, ancilla])

    first_targets = ["I", stim.target_x(t1.value), stim.target_y(t1.value), stim.target_z(t1.value)]
    second_targets = ["I", stim.target_x(t2.value), stim.target_y(t2.value), stim.target_z(t2.value)]
    measure_targets = ["I", stim.target_x(ancilla)]

    errors = []
    for first_target in first_targets:
        for second_target in second_targets:
            for measure_target in measure_targets:
                error = []
                if first_target != "I":
                    error.append(first_target)
                if second_target != "I":
                    error.append(second_target)
                if measure_target != "I":
                    error.append(measure_target)

                if len(error) > 0:
                    errors.append(error)

    for error in errors:
        circuit.append_operation("CORRELATED_ERROR", error, ind_p)

    circuit.append_operation("M", [ancilla])

    return circuit
