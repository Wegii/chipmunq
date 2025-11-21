# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name, disable=no-name-in-module, disable=unused-argument

"""Tools to use functionality from Stim."""
from typing import Union, List, Dict
from math import log as loga
from stim import Circuit as StimCircuit
from stim import target_rec as StimTarget_rec
from qiskit import QuantumCircuit


def get_stim_circuits_with_detectors(
    circuit: Union[QuantumCircuit, List]
):
    """Converts compatible qiskit circuits to stim circuits.
       Dictionaries are not complete. For the stim definitions see:
       https://github.com/quantumlib/Stim/blob/main/doc/gates.md

        Note: This is an improved version, that also supports detectors gates (detector, observable, etc.) inside of an
             circuit. The detectors are not simply added at the end of the circuit, but rather at the specified location.
             For this to work, a general qiskit circuit with generalized gates (representing e. g. the detector)
             is needed. These gates are utilized and added by the StimCodeCircuit.

    Args:
        circuit: Compatible gates are Paulis, controlled Paulis, h, s,
        and sdg, swap, reset, measure and barrier. Compatible noise operators
        correspond to a single or two qubit pauli channel.
        detectors: A list of measurement comparisons. A measurement comparison
        (detector) is either a list of measurements given by a the name and index
        of the classical bit or a list of dictionaries, with a mandatory clbits
        key containing the classical bits. A dictionary can contain keys like
        'qubits', 'time', 'basis' etc.
        logicals: A list of logical measurements. A logical measurement is a
        list of classical bits whose total parity is the logical eigenvalue.
        Again it can be a list of dictionaries.

    Returns:
        stim_circuits, stim_measurement_data
    """

    stim_circuits = []
    stim_measurement_data = []
    if isinstance(circuit, QuantumCircuit):
        circuit = [circuit]
    for circ in circuit:
        stim_circuit = StimCircuit()

        qiskit_to_stim_dict = {
            "id": "I",
            "x": "X",
            "y": "Y",
            "z": "Z",
            "h": "H",
            "s": "S",
            "sdg": "S_DAG",
            "cx": "CX",
            "cy": "CY",
            "cz": "CZ",
            "swap": "SWAP",
            "reset": "R",
            "measure": "M",
            "barrier": "TICK",
        }

        # Instructions specific to detectors/measurements
        stim_detector_gates = [
            "DETECTOR",
            "SHIFT_COORDS",
            "OBSERVABLE_INCLUDE",
            "QUBIT_COORDS"
        ]

        measurement_data = []
        qreg_offset = {}
        creg_offset = {}
        prevq_offset = 0
        prevc_offset = 0

        for instruction in circ.data:
            inst = instruction.operation
            qargs = instruction.qubits
            cargs = instruction.clbits
            for qubit in qargs:
                if qubit._register.name not in qreg_offset:
                    qreg_offset[qubit._register.name] = prevq_offset
                    prevq_offset += qubit._register.size
            for bit in cargs:
                if bit._register.name not in creg_offset:
                    creg_offset[bit._register.name] = prevc_offset
                    prevc_offset += bit._register.size

            qubit_indices = [
                qargs[i]._index + qreg_offset[qargs[i]._register.name] for i in range(len(qargs))
            ]

            # Gates and measurements
            if inst.name in qiskit_to_stim_dict:
                #print(inst)
                if len(cargs) > 0:  # keeping track of measurement indices in stim
                    measurement_data.append([cargs[0]._register.name, cargs[0]._index])

                if qiskit_to_stim_dict[inst.name] == "TICK":  # barrier
                    stim_circuit.append("TICK")
                elif hasattr(inst, "condition") and inst.condition is not None:  # handle c_ifs
                    if inst.name in "xyz":
                        if inst.condition[1] == 1:
                            clbit = inst.condition[0]
                            stim_circuit.append(
                                qiskit_to_stim_dict["c" + inst.name],
                                [
                                    StimTarget_rec(
                                        measurement_data.index(
                                            [clbit._register.name, clbit._index]
                                        )
                                        - len(measurement_data)
                                    ),
                                    qubit_indices[0],
                                ],
                            )
                            #stim_circuit.append("TICK")
                        else:
                            raise Exception(
                                "Classically controlled gate must be conditioned on bit value 1"
                            )
                    else:
                        raise Exception(
                            "Classically controlled " + inst.name + " gate is not supported"
                        )
                else:  # gates/measurements acting on qubits
                    stim_circuit.append(qiskit_to_stim_dict[inst.name], qubit_indices)
                    # Add barrier to two qubit gates, in order to not have stim combining these gates again
                    #if inst.name in ["swap", "cx", "cy", "cz"]:
                    #    stim_circuit.append("TICK")
            elif inst.name in stim_detector_gates:
                if inst.name == "QUBIT_COORDS":
                    # NOTE: ignore these for now, since stimcircuit has issues converting this back
                    stim_circuit.append("QUBIT_COORDS", [inst.params[0]['index']], inst.params[0]['coords'])
                    #stim_circuit.append("TICK")

                    pass
                elif inst.name == "DETECTOR" or inst.name == "OBSERVABLE_INCLUDE":
                    stim_record_targets = []
                    for rec in inst.params[0]['rec_indices']:
                        stim_record_targets.append(
                            StimTarget_rec(rec)
                        )
                    stim_circuit.append(inst.name, stim_record_targets, inst.params[0]['coords'])
                elif inst.name == "SHIFT_COORDS":
                    stim_circuit.append("SHIFT_COORDS", [], inst.params[0]['shift_vector'])
            else:
                raise Exception("Unexpected operations: " + str([inst, qargs, cargs]))

        compact_circ = StimCircuit()
        #for layer in collect_circuit_layers(stim_circuit):
        for layer in collect_circuit_layers_with_ticks(stim_circuit):
            compact_circ += layer
            compact_circ.append_operation("TICK")

        stim_circuits.append(compact_circ)
        stim_measurement_data.append(measurement_data)
    

    return stim_circuits, stim_measurement_data


def collect_circuit_layers_with_ticks(circ: StimCircuit) -> list[StimCircuit]:
    """Split a Stim circuit into parallel-executable layers between ticks

    Split a Stim circuit into parallel-executable layers while keeping DETECTOR, OBSERVABLE_INCLUDE, SHIFT_COORDS,
    QUBIT_COORDS in the same relative layer defined by TICK boundaries.

    :param circ: Stim circuit to process
    :type circ: StimCircuit
    :return: list of circuit layers. All instructions in one layer can be executed in parallel.
    :rtype: list[StimCircuit]
    """

    # Split circuit into tick-delimited layers
    tick_layers = [[]]
    for instr in circ:
        if instr.name == "TICK":
            tick_layers.append([])
        else:
            tick_layers[-1].append(instr)

    # ensure no trailing empty layer
    if tick_layers and len(tick_layers[-1]) == 0:
        tick_layers.pop()

    # Iterate over tick layers
    for layer_instrs in tick_layers:
        circ_cpy = StimCircuit()

    # Iterate over each tick-defined layer
    final_layers: list[StimCircuit] = []
    for layer_instrs in tick_layers:
        circ_cpy = StimCircuit()
        for instr in layer_instrs:
            circ_cpy.append_operation(instr)

        sublayers = collect_circuit_layers(circ_cpy)

        # append the resulting sublayers
        final_layers.extend(sublayers)

    return final_layers


def collect_circuit_layers(circ: StimCircuit) -> list[StimCircuit]:
    """Collect all layers that can be executed in parallel.

    Adapted from:
    - https://github.com/munich-quantum-toolkit/qecc/blob/main/src/mqt/qecc/circuit_synthesis/circuit_utils.py

    :param circ: Stim circuit to process
    :type circ: StimCircuit
    :raises ValueError: _description_
    :return: list of circuit layers. All instructions in one layer can be executed in parallel.
    :rtype: list[StimCircuit]
    """


    # Copy the circuit and separate all instructions by ticks
    circ_cpy = StimCircuit()
    for instr in circ:
        # Moved outside grouping, since these operations to not act on qubits, but carry additional parameters
        if (instr.name == "QUBIT_COORDS" or
            instr.name == "DETECTOR" or
            instr.name == "OBSERVABLE_INCLUDE" or
            instr.name == "SHIFT_COORDS"):
            circ_cpy.append_operation(instr)
            circ_cpy.append_operation("TICK", [])
            continue

        for grp in instr.target_groups():
            qubits = [q.qubit_value for q in grp]
            circ_cpy.append_operation(instr.name, qubits)
            circ_cpy.append_operation("TICK", [])


    # Now work with the copied circuit
    circ = circ_cpy
    n_qubits = circ.num_qubits
    layers = []

    while len(circ) > 0:
        layer = StimCircuit()
        # Track used qubits in this layer
        qubit_layer_used = [False] * n_qubits 
        # Track instructions to delete after adding them to the layer
        instr_to_delete = []  
        idx = 0

        while idx < len(circ):
            instr = circ[idx]

            # Skip TICK instructions
            while instr is not None and instr.name == "TICK" and idx < len(circ):
                circ.pop(idx)
                instr = circ[idx] if idx < len(circ) else None

            if instr is None:  # No more instructions to process
                break
            
            if (instr.name == "QUBIT_COORDS" or
                instr.name == "DETECTOR" or
                instr.name == "OBSERVABLE_INCLUDE" or
                instr.name == "SHIFT_COORDS"):
                # Simply append these instructions
                layer.append_operation(instr)
                instr_to_delete.append(idx)
            else:

                qubits = [q.qubit_value for q in instr.targets_copy()]

                # Check if any qubit from this instruction is already used in the layer
                if not any(qubit_layer_used[q] for q in qubits):
                    layer.append_operation(instr.name, qubits)
                    instr_to_delete.append(idx)  # Mark this instruction for removal

                # Mark the qubits used in this instruction
                for q in qubits:
                    qubit_layer_used[q] = True

            idx += 1

        # Add the layer to the list
        layers.append(layer)

        # Remove the instructions that were added to the layer
        for n_deleted, gate_idx in enumerate(instr_to_delete):
            circ.pop(gate_idx - n_deleted)

    return layers
def collect_circuit_layers_with_ticks(circ: StimCircuit) -> list[StimCircuit]:
    """Split a Stim circuit into parallel-executable layers between ticks

    Split a Stim circuit into parallel-executable layers while keeping DETECTOR, OBSERVABLE_INCLUDE, SHIFT_COORDS,
    QUBIT_COORDS in the same relative layer defined by TICK boundaries.

    :param circ: Stim circuit to process
    :type circ: StimCircuit
    :return: list of circuit layers. All instructions in one layer can be executed in parallel.
    :rtype: list[StimCircuit]
    """

    # Split circuit into tick-delimited layers
    tick_layers = [[]]
    for instr in circ:
        if instr.name == "TICK":
            tick_layers.append([])
        else:
            tick_layers[-1].append(instr)

    # ensure no trailing empty layer
    if tick_layers and len(tick_layers[-1]) == 0:
        tick_layers.pop()

    # Iterate over tick layers
    for layer_instrs in tick_layers:
        circ_cpy = StimCircuit()

    # Iterate over each tick-defined layer
    final_layers: list[StimCircuit] = []
    for layer_instrs in tick_layers:
        circ_cpy = StimCircuit()
        for instr in layer_instrs:
            circ_cpy.append_operation(instr)

        sublayers = collect_circuit_layers(circ_cpy)

        # append the resulting sublayers
        final_layers.extend(sublayers)

    return final_layers


def collect_circuit_layers(circ: StimCircuit) -> list[StimCircuit]:
    """Collect all layers that can be executed in parallel.

    Adapted from:
    - https://github.com/munich-quantum-toolkit/qecc/blob/main/src/mqt/qecc/circuit_synthesis/circuit_utils.py

    :param circ: Stim circuit to process
    :type circ: StimCircuit
    :raises ValueError: _description_
    :return: list of circuit layers. All instructions in one layer can be executed in parallel.
    :rtype: list[StimCircuit]
    """


    # Copy the circuit and separate all instructions by ticks
    circ_cpy = StimCircuit()
    for instr in circ:
        # Moved outside grouping, since these operations to not act on qubits, but carry additional parameters
        if (instr.name == "QUBIT_COORDS" or
            instr.name == "DETECTOR" or
            instr.name == "OBSERVABLE_INCLUDE" or
            instr.name == "SHIFT_COORDS"):
            circ_cpy.append_operation(instr)
            circ_cpy.append_operation("TICK", [])
            continue

        for grp in instr.target_groups():
            qubits = [q.qubit_value for q in grp]
            circ_cpy.append_operation(instr.name, qubits)
            circ_cpy.append_operation("TICK", [])


    # Now work with the copied circuit
    circ = circ_cpy
    n_qubits = circ.num_qubits
    layers = []

    while len(circ) > 0:
        layer = StimCircuit()
        # Track used qubits in this layer
        qubit_layer_used = [False] * n_qubits 
        # Track instructions to delete after adding them to the layer
        instr_to_delete = []  
        idx = 0

        while idx < len(circ):
            instr = circ[idx]

            # Skip TICK instructions
            while instr is not None and instr.name == "TICK" and idx < len(circ):
                circ.pop(idx)
                instr = circ[idx] if idx < len(circ) else None

            if instr is None:  # No more instructions to process
                break
            
            if (instr.name == "QUBIT_COORDS" or
                instr.name == "DETECTOR" or
                instr.name == "OBSERVABLE_INCLUDE" or
                instr.name == "SHIFT_COORDS"):
                # Simply append these instructions
                layer.append_operation(instr)
                instr_to_delete.append(idx)
            else:

                qubits = [q.qubit_value for q in instr.targets_copy()]

                # Check if any qubit from this instruction is already used in the layer
                if not any(qubit_layer_used[q] for q in qubits):
                    layer.append_operation(instr.name, qubits)
                    instr_to_delete.append(idx)  # Mark this instruction for removal

                # Mark the qubits used in this instruction
                for q in qubits:
                    qubit_layer_used[q] = True

            idx += 1

        # Add the layer to the list
        layers.append(layer)

        # Remove the instructions that were added to the layer
        for n_deleted, gate_idx in enumerate(instr_to_delete):
            circ.pop(gate_idx - n_deleted)

    return layers