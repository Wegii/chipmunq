from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.layout import Layout


class GenericRouter(TransformationPass):

    def __init__(self, backend):
        super().__init__()

        #coupling_map (Union[CouplingMap, Target]): Directed graph represented a coupling map.
        self.backend = backend
        self.coupling_map = backend.coupling_map

    def _local_routing(self):
        # Intra-chiplet routing
        raise NotImplementedError

    def _global_routing(self):
        # Inter-chiplet routing
        raise NotImplementedError


class BasicSwapRouter(GenericRouter):
    """ Wrapper around qiskit.transpiler.passes.BasicSwap utilizing a custom layout"""

    def __init__(self, backend):
        super().__init__(backend)

    def run(self, dag):
        return self._local_routing(dag)

    def _local_routing(self, dag: DAGCircuit):
        """Perform local *and* global routing by naive SWAPgate insertion using a custom layout.

        Note: All of Qiskit’s built-in routing stages will additionally run the VF2PostLayout pass after routing. This
              might reassign the initial layout, if lower-error qubits can be found. Thus, we are not going to use the
              default implementation, as we want to keep the qubit mapping.

        This routing pass utilizes the same algorithm as in qiskit.transpiler.passes.BasicSwap

        :param dag: _description_
        :type dag: DAGCircuit
        :return: _description_
        :rtype: _type_
        """

        new_dag = dag.copy_empty_like()
        current_layout = self.property_set["layout"]
        canonical_register = dag.qregs["q"]

        # Implementation from: https://github.com/Qiskit/qiskit/blob/main/qiskit/transpiler/passes/routing/basic_swap.py
        for layer in dag.serial_layers():
            subdag = layer["graph"]

            for gate in subdag.two_qubit_ops():
                physical_q0 = current_layout[gate.qargs[0]]
                physical_q1 = current_layout[gate.qargs[1]]

                if self.coupling_map.distance(physical_q0, physical_q1) != 1:
                    # Insert a new layer with the SWAP(s).
                    swap_layer = DAGCircuit()
                    swap_layer.add_qreg(canonical_register)

                    path = self.coupling_map.shortest_undirected_path(physical_q0, physical_q1)
                    for swap in range(len(path) - 2):
                        connected_wire_1 = path[swap]
                        connected_wire_2 = path[swap + 1]

                        qubit_1 = current_layout[connected_wire_1]
                        qubit_2 = current_layout[connected_wire_2]

                        # create the swap operation
                        swap_layer.apply_operation_back(
                            SwapGate(), (qubit_1, qubit_2), cargs=(), check=False
                        )

                    # layer insertion
                    order = current_layout.reorder_bits(new_dag.qubits)
                    new_dag.compose(swap_layer, qubits=order)

                    # update current_layout
                    # TODO: is this necessary??
                    for swap in range(len(path) - 2):
                        current_layout.swap(path[swap], path[swap + 1])

            order = current_layout.reorder_bits(new_dag.qubits)
            new_dag.compose(subdag, qubits=order)

        # This pass must set the following property: self.property_set["final_layout"]
        self.property_set["final_layout"] = current_layout

        return new_dag


class SABRERouter(GenericRouter):
    """ Wrapper around qiskit.transpiler.passes.SabreSwap"""

    def __init__(self, coupling_map):
        super().__init__(coupling_map)

    def run(self, dag):
        # TODO: run SABRE swap algorithm

        # The SABRE algorithm can not perform global and local routing separately.
        #self._global_routing()
        self._local_routing()

    def _local_routing(self):
        # Intra
        # Local routing here also performs global routing.
        pass

    def _global_routing(self):
        # Inter
        # This is not implemented here, since SABRE is not capable of this
        pass




    
