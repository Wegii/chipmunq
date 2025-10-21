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

        current_layout = self.property_set["layout"]
        #new_dag = dag.copy_empty_like()
        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)
        

        for node in dag.topological_op_nodes():
                if len(node.qargs) == 2:
                    q0, q1 = [dag.qubits.index(q) for q in node.qargs]
                    
                    # Check distance in coupling map
                    if not self.coupling_map.distance(q0, q1) == 1:
                        # Find shortest path connecting both qubits
                        path = self.coupling_map.shortest_undirected_path(q0, q1)
                     
                        # Insert swaps along path except last edge
                        for i in range(len(path) - 2):
                            swap = SwapGate()
                            new_dag.apply_operation_back(
                                swap,
                                qargs=[new_dag.qubits[path[i]], new_dag.qubits[path[i+1]]]
                            )
                        
                        # Apply original gate
                        new_dag.apply_operation_back(node.op, qargs=[new_dag.qubits[path[-2]], new_dag.qubits[path[-1]]])

                        # SWAP backwards
                        for i in reversed(range(len(path) - 2)):
                            swap = SwapGate()
                            new_dag.apply_operation_back(
                                swap,
                                qargs=[new_dag.qubits[path[i]], new_dag.qubits[path[i+1]]]
                            )
                        
                    else:
                        # Local two-qubit gates
                        new_dag.apply_operation_back(node.op, qargs=node.qargs)
                        
                else:
                    # Single-qubit gates
                    new_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)
                    
        #from qiskit.visualization import dag_drawer
        #dag_drawer(dag, filename="data/backends/mapping/dag.png")
        #dag_drawer(new_dag, filename="data/backends/mapping/routed_dag.png")

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




    
