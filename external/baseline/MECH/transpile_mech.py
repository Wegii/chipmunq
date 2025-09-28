import qiskit.circuit
from Circuit import *
from Chiplet import *
from HighwayOccupancy import *
from Router import *
from MECHBenchmarks import *

from tqdm import tqdm
import matplotlib.pyplot as plt
import qiskit
import logging


def circuit_to_program(circuit: qiskit.circuit) -> list:
    """ Translate circuit to MECH program.

    MECH only considers two-qubit gates, which are represented as ControlBlocks acting on
    control and target qubits.

    Args:
        circuit (qiskit.circuit): Circuit to translate

    Returns:
        list: List of ControlBlocks
    """
    
    program = []
    for _, qargs, _ in circuit.data:
        
        # Add two-qubit gate    
        if len(qargs) == 2:
            control = circuit.qubits.index(qargs[0])
            target = circuit.qubits.index(qargs[1])
            program.append(ControlBlock(control, Counter(list([target]))))
        # Ignore one-qubit gates, as these are not relevant for mapping and routing

    return program


def program_to_circuit(router):
    """_summary_

    Args:
        router (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Construct circuit with correct number of qubits
    circuit = qiskit.QuantumCircuit(len(router.circuit.circuit_lines))

    # Iterate over depth
    for idx in range(router.circuit.depth):
        # Iterate over each qubit
        for line in range(len(router.circuit.circuit_lines)):
            #print(router.circuit.take_role(line, idx))
            
            if router.circuit.take_role(line, idx) == 'q':
                # Single qubit-gate
                node = router.circuit.take_node(line, idx)
                control_line = node.q
                control_qubit = router.highway_manager.idx_qubit_dict[control_line]
                # Measurement in X-basis
                circuit.H(control_qubit)
                circuit.reset(control_qubit)

            # Iterate over targets
            # TODO: WIP
            """
            if router.circuit.take_role(line, idx) in ['t', 'mt']:
                # Target qubit

                node = router.circuit.take_node(line, idx)
                if isinstance(node,  OpNode):
                    control_line = node.control
                if isinstance(node, MOpNode):
                    control_line = node.shared
                control_qubit, target_qubit = router.highway_manager.idx_qubit_dict[control_line], router.highway_manager.idx_qubit_dict[line]

                circuit.ControlledGate(control_qubit, target_qubit)
            """
    
    #print(on_chip_gate_num)
    print("Conversion succeeded")
    return circuit


def transpile_circuit_MECH(circuit: qiskit.circuit, G: nx.Graph) -> Router:
    """Perform mapping and routing with MECH

    Args:
        circuit (qiskit.circuit): _description_
        G (nx.Graph): _description_

    Returns:
        Router: _description_
    """
    
    logging.info("Starting algorithm: MECH")
    
    # Translate circuit to program (necessary for MECH)
    program = circuit_to_program(circuit)
    logging.debug("Program generated")
    
    # Set some parameters
    prep_period = 13
    meas_period = 2
    cross_chip_gate_weight = 7.4

    # Initialize router
    router = Router(G, prep_period=prep_period, meas_period=meas_period)

    # Perform local routing
    for i, control_block in enumerate(tqdm(program, total=len(program))):
        router.execute_control_multi_target_block(control_block, cross_chip_gate_weight, cross_chip_overhead = 3)

    # Perform routing using the highway
    for shuttle_idx in range(len(router.highway_manager.shuttle_stack)):
        for idx in range(router.highway_manager.get_shuttle_prep_start_time(shuttle_idx), router.highway_manager.get_shuttle_exec_start_time(shuttle_idx)):
            for line in range(len(router.circuit.circuit_lines)):
                assert router.circuit.is_position_empty(line, idx)

        source_measured_qubits_dict = router.highway_manager.bridge_throughout_highway(router.circuit, shuttle_idx)
        router.highway_manager.reentangle_throughout_highway(router.circuit, shuttle_idx, source_measured_qubits_dict)
    
    # Perform measurements
    for shuttle_idx in range(len(router.highway_manager.shuttle_stack)):
        router.highway_manager.measure_throughout_highway(router.circuit, shuttle_idx)


    # TODO: The constructed circuit can not yet be used to perform simulations. In the following, only the router, which keeps a circuit
    # consisting of ControlBlock, is returned. In a future version it should be possible to translate the program structure back to a 
    # qiskit.circuit.
    # transpiled_qc = program_to_circuit(router)

    logging.info("Finished algorithm!")

    return router