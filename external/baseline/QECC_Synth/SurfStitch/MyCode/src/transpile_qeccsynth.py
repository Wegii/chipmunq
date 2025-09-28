import numpy as np
import json
import logging
from qiskit import QuantumCircuit

# QECC-Synth
from StabCode import extract, surface_code
from Stage1_Map_ver3 import S1_transpile
from Stage2_Schedule_ver3 import S2_transpile, toQASM


def transpile_circuit_QECCSynth(code_distance: int, architecture: tuple[np.array, list], archName: str) -> QuantumCircuit:
    """Generate a qiskit Circuit given a surface code (distance) and a hardware architecture (coupling graph).

    Attention: This algorithm takes an QEC code and architecture as input. The code must be expressed in terms of
               stabilizers. Thus, this implementation currently only support the surface code. For this, the code
               distance must be provided in order to construct the QEC code here.
    
    Args:
        code_distance (int): Distance of the surface code. The code and circuit is constructed in this function.
                             It is not possible to pass a generic circuit!
        architecture (tuple[np.array, list]): Tuple containing coupling graph and index2coordinates dictionary
        archName (str): Name of architecture. Used for output files

    Returns:
        QuantumCircuit: Qiskit QuantumCircuit object with constructed circuit
    """

    logging.info("Starting algorithm: QECC-Synth")

    # Mapping protocol
    protocol = 'Flag-Bridge'
    # maximum size for the ancilla block (If none, then this value is computed)
    maxL = None
    # upper bound for time step (minutes or seconds?)
    maxT = 50 
    # not defined what this is ...
    Len = None 
    # Not defined what this is...
    chunkNum = 1 
    # Location of ouput files
    base_output = 'data/qecc_synth_output/'

    # Generate surface code (written as stabilizers to a file), given code distance
    stabilizer_file = f'{base_output}stab/surface_code_distance_{code_distance}'
    codeName = f'surface_{code_distance}'
    surface_code(stabilizer_file, code_distance)
    stabilizers, dataNum = extract(stabilizer_file)
    code = (dataNum, stabilizers)
    
    # Coupling graph and Qubit mapping
    CG, idx2coord = architecture

    run_name = codeName + '-' + archName

    # 1. Stage
    S1_transpile(code,
                 CG,
                 protocol,
                 maxL,
                 Len,
                 chunkNum,
                 base_output + 'aux_files/clause_' + run_name,
                 base_output + 'aux_files/sol_' + run_name,
                 base_output + 'output/S1_' + run_name)
    with open(base_output + 'output/S1_' + run_name + '.json', 'r') as f:
        result = json.load(f)

    
    # 2. Stage
    S2_transpile(result['chunkNum'],
                 result['Data'],
                 result['Stab'],
                 CG,
                 idx2coord,
                 result['Collison_list'],
                 maxT,
                 base_output + 'output/S2_' + run_name)

    with open(base_output + 'output/S2_' + run_name + '.json', 'r') as f:
        result = json.load(f)    

    # Construct circuit
    physNum = len(CG)
    QC = QuantumCircuit(physNum, physNum)   
    for i in range(result['chunkNum']):
        QC = QC.compose(toQASM(len(CG), result['Circuit'][i]))
        QC = QC.compose(toQASM(len(CG), result['Swap_layer'][i]))

    logging.info("Finished algorithm!")

    return QC