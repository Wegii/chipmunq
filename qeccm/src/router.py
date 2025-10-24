from threading import local
from numpy import single
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
        print("Starting routing")
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
        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)
        
        for node in dag.topological_op_nodes():               
                if len(node.qargs) == 2:
                    q0, q1 = node.qargs[0]._index, node.qargs[1]._index
                    
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

class ParallelSwapRouter(GenericRouter):
    """ Parallel implementation of BasicSwapRouter"""

    def __init__(self, backend):
        super().__init__(backend)

    def run(self, dag):
        print("Starting routing")

        current_layout = self.property_set["layout"]


        # Iterate over Chiplets
            # Get all nodes of this chiplet
            # Iterate over dag and extract 
            #   - Each node also gets a timing-position, in order to construct the dag afterwards
            #   - Local routing of Gates if source and target on chip: local_nodes
            #   - Global routing of gates if target outside of this chip: remote_nodes
            # local_routing of local_nodes
            # global_routing of remote_nodes
        local_dag_nodes, remote_dag_nodes, single_dag_nodes = self._dag_node_extraction()
        
        # Local routing on chiplet
        local_dag_instr = self._local_routing(local_dag_nodes)
        # Global routing between chiplet
        global_dag_instr = self._global_routing(remote_dag_nodes)

        # Construct dag given local and global routing instructions
        new_dag = self._build_parallel_dag(dag, local_dag_instr, global_dag_instr, single_dag_nodes)

        # Note: Layout transpilation pass needs to set this property
        self.property_set["final_layout"] = current_layout

        return new_dag

    def _build_parallel_dag(self, dag, local_dag, global_dag):
        # Sort local and global routing instructions based on timing-position

        # TODO: sort local and global
        # Combine local and global
        # Construct dag

        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

    def _dag_node_extraction(self, dag) -> tuple[list, list]:

        local_dag_nodes = []
        remote_dag_nodes = []
        single_dag_nodes = []

        for node in dag.topological_op_nodes():               
            if len(node.qargs) == 2:
                q0, q1 = node.qargs[0]._index, node.qargs[1]._index
                
                # Check if q1 or q2 not on this chip
                remote_dag_nodes.append("remote")

                local_dag_nodes.append("local")
                
            else:
                # Single-qubit gates
                #new_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)
                single_dag_nodes.append("asd")



        return local_dag_nodes, remote_dag_nodes, single_dag_nodes

    def _local_routing(self, local_dag_nodes: list):
        """ Perform local basic swap routing

        Note: Parallelization over all nodes

        :param dag: _description_
        :type dag: DAGCircuit
        :return: _description_
        :rtype: _type_
        """

        # TODO: parallelize routing

        # TODO: return list of dag instructions
        routed_local_dag_nodes = []


        return routed_local_dag_nodes

    def _global_routing(self, remote_dag_nodes: list):
        """Perform global basic swap routing

        Note: Parallelization over all nodes    
        
        :param remote_dag_nodes: _description_
        :type remote_dag_nodes: list
        :return: _description_
        :rtype: _type_
        """

        # TODO: parallelize routing

        # TODO: return list of dag instructions
        routed_remote_dag_nodes = []

        return routed_remote_dag_nodes


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




    
