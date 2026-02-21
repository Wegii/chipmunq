# Typing
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.dagcircuit import DAGCircuit

# Qiskit
from qiskit.transpiler.basepasses import TransformationPass

from qeccm.backends.BackendChipletV2 import BackendChipletV2


class GenericRouter(TransformationPass):
    def __init__(self, backend: BackendChipletV2):
        super().__init__()

        # coupling_map (Union[CouplingMap, Target]): Directed graph represented a coupling map.
        self.backend = backend
        # Select the coupling_map that has defective qubits removed
        self.coupling_map = backend.defective_coupling_map  # backend.coupling_map

    def _local_routing(self):
        # Intra-chiplet routing
        raise NotImplementedError

    def _global_routing(self):
        # Inter-chiplet routing
        raise NotImplementedError


class BasicSwapRouter(GenericRouter):
    """Wrapper around qiskit.transpiler.passes.BasicSwap utilizing a custom layout"""

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

        # Utilize mapping of physical qubit to partition mapping to check if qubits need routing. Qubits part of the
        # same partition do not need any routing
        pq_to_partition = self.property_set["pq_to_partition"]

        for node in dag.topological_op_nodes():
            if len(node.qargs) == 2:
                q0, q1 = node.qargs[0]._index, node.qargs[1]._index

                # Check distance in coupling map
                # if not self.coupling_map.distance(q0, q1) == 1:
                if pq_to_partition[q0] != pq_to_partition[q1]:
                    # Find shortest path connecting both qubits
                    path = self.coupling_map.shortest_undirected_path(q0, q1)
                    # TODO: this can be replaced with a easier calculation, by considering the grid layout of the
                    #       backend. Note: only works for grid layout then.

                    # Insert swaps along path except last edge
                    mid = len(path) // 2

                    # Route from start to middle
                    for i in range(mid - 1):
                        swap = SwapGate()
                        qubit_1 = path[i]  # current_layout[path[i]]
                        qubit_2 = path[i + 1]  # current_layout[path[i + 1]]

                        new_dag.apply_operation_back(
                            swap,
                            qargs=[new_dag.qubits[path[i]], new_dag.qubits[path[i + 1]]],
                            # SwapGate(), (qubit_1, qubit_2), cargs=(), check=False
                        )
                    # Route from end to middle
                    for i in range(len(path) - 1, mid, -1):
                        swap = SwapGate()
                        qubit_1 = path[i]  # current_layout[path[i]]
                        qubit_2 = path[i - 1]  # current_layout[path[i - 1]]
                        new_dag.apply_operation_back(
                            swap,
                            qargs=[new_dag.qubits[path[i]], new_dag.qubits[path[i - 1]]],
                            # SwapGate(), (qubit_1, qubit_2), cargs=(), check=False
                        )

                    # Apply original gate
                    # new_dag.apply_operation_back(node.op, qargs=[new_dag.qubits[path[-2]], new_dag.qubits[path[-1]]])
                    new_dag.apply_operation_back(
                        node.op, qargs=[new_dag.qubits[path[mid - 1]], new_dag.qubits[path[mid]]]
                    )

                    # Route backwards
                    for i in reversed(range(mid - 1)):
                        swap = SwapGate()
                        # new_dag.apply_operation_back(
                        #    swap,
                        #    qargs=[new_dag.qubits[path[i]], new_dag.qubits[path[i+1]]]
                        # )
                        qubit_1 = path[i]  # current_layout[path[i]]
                        qubit_2 = path[i + 1]  # current_layout[path[i + 1]]

                        new_dag.apply_operation_back(
                            swap,
                            qargs=[new_dag.qubits[path[i]], new_dag.qubits[path[i + 1]]],
                            # SwapGate(), (qubit_1, qubit_2), cargs=(), check=False
                        )

                    for i in reversed(range(len(path) - 1, mid, -1)):
                        # swap = SwapGate()
                        # new_dag.apply_operation_back(
                        #    swap,
                        #    qargs=[new_dag.qubits[path[i]], new_dag.qubits[path[i-1]]]
                        # )
                        swap = SwapGate()
                        qubit_1 = path[i]  # current_layout[path[i]]
                        qubit_2 = path[i - 1]  # current_layout[path[i - 1]]
                        new_dag.apply_operation_back(
                            swap,
                            qargs=[new_dag.qubits[path[i]], new_dag.qubits[path[i - 1]]],
                            # SwapGate(), (qubit_1, qubit_2), cargs=(), check=False
                        )

                else:
                    # Local two-qubit gates
                    new_dag.apply_operation_back(node.op, qargs=node.qargs)
            else:
                # Single-qubit gates
                new_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)

        # This pass must set the following property: self.property_set["final_layout"]
        self.property_set["final_layout"] = current_layout

        return new_dag


class CostRouter(GenericRouter):
    def __init__(self, backend: BackendChipletV2, alpha: float = 0.0, beta: float = 0.0):
        super().__init__(backend)

        self.alpha = alpha
        self.beta = beta

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        current_layout = self.property_set["layout"]

        # Route gates that are on a chip
        local_routed_dag = self._local_routing(dag)
        # Route gates across chips
        full_routed_dag = self._global_routing(local_routed_dag)

        # Set final layout
        self.property_set["final_layout"] = current_layout

        print("Routing done")

        return full_routed_dag

    def _local_routing(self, dag: DAGCircuit) -> DAGCircuit:
        # TODO: Simply iterate over dag and route all connections that are on chip

        print("Local routing")

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
                    if self.backend.get_chiplet_of_node(q0) == self.backend.get_chiplet_of_node(q1):
                        path = self.coupling_map.shortest_undirected_path(q0, q1)
                        new_dag = self._perform_routing_between_nodes(new_dag, path, node, "")

                    else:
                        # Remote gate. Routing performed during global_routing step
                        new_dag.apply_operation_back(node.op, qargs=node.qargs)
                else:
                    # Add local two-qubit gates that do not require routing
                    new_dag.apply_operation_back(node.op, qargs=node.qargs)
            else:
                # Add single-qubit gates to do not require routing
                new_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)

        return new_dag

    def _global_routing(self, dag: DAGCircuit) -> DAGCircuit:
        """_summary_

        path_cost = path_length + alpha*inter_chiplet_error + beta*inter_chiplet_utilization

        :param dag: _description_
        :type dag: DAGCircuit
        :return: _description_
        :rtype: DAGCircuit
        """
        print("Global cost routing")

        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        # Get all inter-chiplet connections and their respective noise from the backend
        inter_chiplet_connections = self.backend.inter_chiplet_connections
        # Keep track of utilization of inter-chiplet connection
        inter_chiplet_utilization = dict.fromkeys(inter_chiplet_connections, 0)

        for node in dag.topological_op_nodes():
            if len(node.qargs) == 2:
                q0, q1 = node.qargs[0]._index, node.qargs[1]._index

                # Check distance in coupling map
                if not self.coupling_map.distance(q0, q1) == 1:
                    if self.backend.get_chiplet_of_node(q0) != self.backend.get_chiplet_of_node(q1):
                        # Assume that we only need one inter-chiplet connection from one chip to the next, and not
                        # multiple inter-chiplet connections from the source to the target

                        # Shortest path
                        path = self.coupling_map.shortest_undirected_path(q0, q1)

                        # Get error of inter-chiplet connection
                        inter_chiplet_nodes = None
                        for i in range(len(path) - 1):
                            n0 = path[i]
                            n1 = path[i + 1]

                            # Found remote connection if the chiplets do not match
                            if self.backend.get_chiplet_of_node(n0) != self.backend.get_chiplet_of_node(n1):
                                inter_chiplet_nodes = (n0, n1)
                                break

                        # In case the connection is found, try to get the noise value from it. Flip source and target
                        # if the node could not be found
                        if inter_chiplet_nodes not in inter_chiplet_connections:
                            inter_chiplet_nodes = (inter_chiplet_nodes[1], inter_chiplet_nodes[0])

                        current_path_cost = (
                            len(path)
                            + self.alpha * inter_chiplet_connections[inter_chiplet_nodes]
                            + self.beta * (inter_chiplet_utilization[inter_chiplet_nodes] + 1)
                        )

                        best_path = path

                        inter_chiplet_connections_of_chiplet = self.backend.chiplet_to_inter_chiplet_connection[
                            self.backend.node_to_chiplet[q0]
                        ]

                        # In case the current node already is an inter_chiplet connection, skip.
                        if q0 not in inter_chiplet_connections_of_chiplet:
                            # Iterate over inter-chiplet connections of this chip, and see if it is possible to generate
                            # a better routing

                            # TODO: Add option to only select the k-nearest inter-chiplet connections
                            for cicc in inter_chiplet_connections_of_chiplet:
                                # Check if the selected inter_chiplet connection is not flagged as defective
                                if cicc not in self.backend.all_defective_qubits:
                                    # Route from source to inter-chiplet connection
                                    p0 = list(self.coupling_map.shortest_undirected_path(q0, cicc))
                                    # Route from inter-chiplet connection to target
                                    p1 = list(self.coupling_map.shortest_undirected_path(cicc, q1))
                                    # Combine into one full path
                                    p_combined = p0 + p1[1:]

                                    # Convert value to int, since this is a numpy int
                                    cicc = int(p1[0])
                                    cicc_2 = int(p1[1])
                                    new_inter_chiplet_nodes = (cicc, cicc_2)
                                    if new_inter_chiplet_nodes not in inter_chiplet_connections:
                                        new_inter_chiplet_nodes = (
                                            new_inter_chiplet_nodes[1],
                                            new_inter_chiplet_nodes[0],
                                        )

                                    if new_inter_chiplet_nodes not in inter_chiplet_connections:
                                        # The inter-chiplet connection does not exist
                                        continue

                                    path_cost = (
                                        len(p_combined)
                                        + self.alpha * inter_chiplet_connections[new_inter_chiplet_nodes]
                                        + self.beta * (inter_chiplet_utilization[new_inter_chiplet_nodes] + 1)
                                    )

                                    if path_cost < current_path_cost:
                                        current_path_cost = path_cost
                                        best_path = p_combined
                                        # print("Chosen a better path")

                        # Route path with lowest cost
                        new_dag = self._perform_routing_between_nodes(
                            new_dag=new_dag, path=best_path, gate_operation=node, method=""
                        )

                        # Update inter-chiplet connection utilization
                        inter_chiplet_utilization[inter_chiplet_nodes] += 1
                    else:
                        new_dag.apply_operation_back(node.op, qargs=node.qargs)
                else:
                    # Add local two-qubit gates that do not require routing
                    new_dag.apply_operation_back(node.op, qargs=node.qargs)
            else:
                # Add single-qubit gates to do not require routing
                new_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)

        return new_dag

    def _perform_routing_between_nodes(self, new_dag, path, gate_operation, method: str = "") -> DAGCircuit:
        # Insert swaps along path except last edge
        mid = len(path) // 2

        # Route from start to middle
        for i in range(mid - 1):
            swap = SwapGate()

            new_dag.apply_operation_back(swap, qargs=[new_dag.qubits[path[i]], new_dag.qubits[path[i + 1]]])
        # Route from end to middle
        for i in range(len(path) - 1, mid, -1):
            swap = SwapGate()
            new_dag.apply_operation_back(swap, qargs=[new_dag.qubits[path[i]], new_dag.qubits[path[i - 1]]])

        # Apply original gate
        new_dag.apply_operation_back(
            gate_operation.op, qargs=[new_dag.qubits[path[mid - 1]], new_dag.qubits[path[mid]]]
        )

        # Route backwards
        for i in reversed(range(mid - 1)):
            swap = SwapGate()

            new_dag.apply_operation_back(swap, qargs=[new_dag.qubits[path[i]], new_dag.qubits[path[i + 1]]])

        for i in reversed(range(len(path) - 1, mid, -1)):
            swap = SwapGate()

            new_dag.apply_operation_back(swap, qargs=[new_dag.qubits[path[i]], new_dag.qubits[path[i - 1]]])

        return new_dag


class AcceleratedBasicSwapRouter(GenericRouter):
    """Accelerated version of qiskit.transpiler.passes.BasicSwap utilizing a custom layout"""

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

        # Accelerated implementation
        from qiskit._accelerate.basic_swap import basic_routing

        # TODO: call basic swap
        new_dag = basic_routing(dag, self.backend.target)

        # This pass must set the following property: self.property_set["final_layout"]
        self.property_set["final_layout"] = current_layout

        return new_dag


class ParallelSwapRouter(GenericRouter):
    """Parallel implementation of BasicSwapRouter"""

    # Parallel implementation
    from joblib import Parallel, delayed

    def __init__(self, backend):
        super().__init__(backend)

    def run(self, dag):
        """Parallel version of SWAPRouter

        Note: Currently there are a couple of things missing, and this is not a correct implementation!!!
        Note: The parallel implementation is currently slower than the single-core implementation!!!

        Iterate over Chiplets and route all nodes of this chiplet:
            - Each node also gets a timing-position, in order to construct the dag afterwards
            - Local routing of Gates if source and target on chip: local_nodes
            - Global routing of gates if target outside of this chip: remote_nodes
            - local_routing of local_nodes
            - global_routing of remote_nodes

        :param dag: _description_
        :type dag: _type_
        :return: _description_
        :rtype: _type_
        """
        print("Starting routing")

        current_layout = self.property_set["layout"]

        local_dag_nodes, remote_dag_nodes, single_dag_instr = self._dag_node_extraction(dag)

        # Local routing on chiplet
        print("local routing")
        local_dag_instr = self._local_routing(dag, local_dag_nodes)
        # Global routing between chiplet
        print("global cost routing")
        global_dag_instr = self._global_routing(remote_dag_nodes)

        # Construct dag given local and global routing instructions
        print("dag construction")
        new_dag = self._build_parallel_dag(dag, local_dag_instr, global_dag_instr, single_dag_instr)

        # Note: Layout transpilation pass needs to set this property
        self.property_set["final_layout"] = current_layout

        return new_dag

    def _build_parallel_dag(self, dag, local_dag_instr, global_dag_instr, single_dag_instr):
        # Sort local and global routing instructions based on timing-position

        # TODO: sort local and global
        # Combine local and global and add single_instructions
        # Construct dag

        list_all_ops = []
        list_all_ops = local_dag_instr

        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        for sub_dag in list_all_ops:
            # new_dag.compose(sub_dag)
            for op, qargs in sub_dag:
                new_dag.apply_operation_back(op, qargs=qargs)

            # if len(dag_op.qargs) == 2:
            #    new_dag.apply_operation_back(dag_op)

        # TODO: asd

        return new_dag  # new_dag

    def _dag_node_extraction(self, dag) -> tuple[list, list]:

        local_dag_nodes = []
        remote_dag_nodes = []
        single_dag_nodes = []

        for i, node in enumerate(dag.topological_op_nodes()):
            if len(node.qargs) == 2:
                q0, q1 = node.qargs[0]._index, node.qargs[1]._index

                # Check if q1 or q2 not on this chip
                # remote_dag_nodes.append("remote")
                # Node on chip
                local_dag_nodes.append((i, node))
            else:
                # Single-qubit gates
                single_dag_nodes.append((i, node))

        # print(len(local_dag_nodes))
        # print(len(list(dag.op_nodes())))
        return local_dag_nodes, remote_dag_nodes, single_dag_nodes

    def _local_routing(self, dag, local_dag_nodes: list):
        """Perform local basic swap routing

        Note: Parallelization over all nodes

        :param dag: _description_
        :type dag: DAGCircuit
        :return: _description_
        :rtype: _type_
        """
        """
        # TODO: parallelize routing
        routed_local_dag_instr = []

        # Check distance in coupling map
        for (node_index, dag_node) in local_dag_nodes:
            q0, q1 = dag_node.qargs[0]._index, dag_node.qargs[1]._index
                
            ops_to_apply = []
            if not self.coupling_map.distance(q0, q1) == 1:
                
                # Find shortest path connecting both qubits
                path = self.coupling_map.shortest_undirected_path(q0, q1)
                swap = SwapGate()
                
                # Insert swaps along path except last edge
                for i in range(len(path) - 2):
                    #sub_dag.apply_operation_back(swap, qargs=[dag.qubits[path[i]], dag.qubits[path[i+1]]])
                    ops_to_apply.append((swap, [dag.qubits[path[i]], dag.qubits[path[i+1]]]))
                
                # Apply original gate
                #sub_dag.apply_operation_back(dag_node.op, qargs=[dag.qubits[path[-2]], dag.qubits[path[-1]]])
                ops_to_apply.append((dag_node.op, [dag.qubits[path[-2]], dag.qubits[path[-1]]]))

                # SWAP backwards
                for i in reversed(range(len(path) - 2)):
                    # sub_dag.apply_operation_back(swap, qargs=[dag.qubits[path[i]], dag.qubits[path[i+1]]])
                    ops_to_apply.append((swap, [dag.qubits[path[i]], dag.qubits[path[i+1]]]))

            else:
                # No swapping needed
                #sub_dag.apply_operation_back(dag_node.op, qargs=dag_node.qargs)
                ops_to_apply.append((dag_node.op, dag_node.qargs))

            # Append node swap subdag
            routed_local_dag_instr.append(ops_to_apply)
        """

        def route_node(node_index, dag_node):
            q0, q1 = dag_node.qargs[0]._index, dag_node.qargs[1]._index
            ops_to_apply = []

            # If not directly connected, find SWAP path
            if self.coupling_map.distance(q0, q1) != 1:
                path = self.coupling_map.shortest_undirected_path(q0, q1)
                swap = SwapGate()

                # Forward SWAPs
                for i in range(len(path) - 2):
                    ops_to_apply.append((swap, [dag.qubits[path[i]], dag.qubits[path[i + 1]]]))

                # Apply original 2-qubit operation
                ops_to_apply.append((dag_node.op, [dag.qubits[path[-2]], dag.qubits[path[-1]]]))

                # Backward SWAPs
                for i in reversed(range(len(path) - 2)):
                    ops_to_apply.append((swap, [dag.qubits[path[i]], dag.qubits[path[i + 1]]]))
            else:
                # No routing needed
                ops_to_apply.append((dag_node.op, dag_node.qargs))

            return (node_index, ops_to_apply)

        # Parallel processing of routing tasks
        results = Parallel(n_jobs=2, backend="loky", batch_size=int(dag.size() / 2))(
            delayed(route_node)(node_index, dag_node) for (node_index, dag_node) in local_dag_nodes
        )

        # Sort results back into the original order (to preserve DAG order)
        print("sorting")
        routed_local_dag_instr = [ops for _, ops in sorted(results, key=lambda x: x[0])]

        # List of dags containing swapping operations
        return routed_local_dag_instr

    def _global_routing(self, remote_dag_nodes: list):
        """Perform global basic swap routing

        Note: Parallelization over all nodes

        :param remote_dag_nodes: _description_
        :type remote_dag_nodes: list
        :return: _description_
        :rtype: _type_
        """
        # TODO: simple swap between the nodes

        # TODO: parallelize routing

        # TODO: return list of dag instructions
        routed_remote_dag_instr = []

        return routed_remote_dag_instr
