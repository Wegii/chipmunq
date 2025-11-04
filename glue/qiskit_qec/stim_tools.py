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
                    stim_circuit.append("TICK")
            elif inst.name in stim_detector_gates:
                if inst.name == "QUBIT_COORDS":
                    # NOTE: ignore these for now, since stimcircuit has issues converting this back
                    #stim_circuit.append("QUBIT_COORDS", [inst.params[0]['index']], inst.params[0]['coords'])
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

        stim_circuits.append(stim_circuit)
        stim_measurement_data.append(measurement_data)

    return stim_circuits, stim_measurement_data

