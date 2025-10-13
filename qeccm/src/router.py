from qiskit.transpiler.basepasses import TransformationPass


class GenericRouter(TransformationPass):

    def __init__(self, coupling_map):
        super().__init__()

        #coupling_map (Union[CouplingMap, Target]): Directed graph represented a coupling map.
        self.target = coupling_map

    def _local_routing(self):
        # Intra-chiplet routing
        raise NotImplementedError

    def _global_routing(self):
        # Inter-chiplet routing
        raise NotImplementedError


class BasicSwapRouter(GenericRouter):
    """ Wrapper around qiskit.transpiler.passes.BasicSwap"""

    def __init__(self, coupling_map):
        super().__init__(coupling_map)

    def run(self, dag):
        return self._local_routing(dag)

        # Set the routingin self.property_set["layout"]

    def _local_routing(self, dag):

        new_dag = dag.copy_empty_like()

        # Local routing here also performs global routing.

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




    
