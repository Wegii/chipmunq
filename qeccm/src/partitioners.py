from __future__ import annotations

# Qiskit transpiler
from qiskit.transpiler.basepasses import AnalysisPass

# Partitioning
import multiprocessing
#import mtkahypar as mtkahypar
import kahypar as kahypar
from qeccm.circuit.hypergraph_circuit import HypergraphCircuit, PartitionedHyperGraph

from qiskit.dagcircuit import DAGCircuit
from qeccm.backends.BackendChipletV2 import BackendChipletV2
from typing import List, Tuple
import numpy as np


class GenericHypergraphPartitioning(AnalysisPass):
    def __init__(self):
        super().__init__()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        # TODO: function that calculates (given a backend and circuit) into how many cuts it is necessary to partition the circuit
        pass


class KaHyParPartitioning(GenericHypergraphPartitioning):
    """ Hypergraph partitioning based on multilevel hypergraph partitioning framework KaHyPar

    See implementation details in `<https://kahypar.org/>`_ `<https://github.com/kahypar/mt-kahypar>`_
    """


    def __init__(self, backend: BackendChipletV2):
        """KaHyPar partitioning initializer"""
        super().__init__()

        self.backend = backend

        # Method for calculating the number of partitions
        self._calculate_partitions_method = "full"

        # Initialize KaHyPar
        self.khp_context = kahypar.Context()

        # TODO: change objective
        self.khp_context.loadINIconfiguration("qeccm/src/kahypar_config.ini")

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Partition a DAGCircuit into a optimal (calculated) number of partitions

        :param dag: _description_
        :type dag: DAGCircuit
        :return: _description_
        :rtype: _type_
        """
        
        # Calculate number of partitions
        self.kp, partition_sizes = self.calculate_partitions(dag)
        
        print(self.kp)
        if self.kp > 1:
            hgc = self.property_set['hyper_dag']

            # Translate vertices and edges from general hypergraph to KaHyPar specific format
            (index_vector, edge_vector) = self.property_set['hyper_dag_kahypar']
            num_vertices = hgc.get_num_vertices() 
            num_hyperedges = len(index_vector) - 1

            # For now, all hyperedges are assumed to have the same weight
            hyperedge_weights = [1 for i in range(0, num_hyperedges)]
            # Qubit vertices are given weight 1
            # Potentially vertices with high connectivity should get higher weight to connect these together
            vertex_weights = [1 for i in range(0, num_vertices)]

            self.khp_context.setK(self.kp)
            self.khp_context.setCustomTargetBlockWeights(partition_sizes)
            self.khp_context.suppressOutput(True)
            self.khp_context.setSeed(42)

            # Convert hypergraph to kahypar format
            kahypar_hg = kahypar.Hypergraph(
                num_vertices,
                num_hyperedges,
                index_vector,
                edge_vector,
                self.kp,
                hyperedge_weights,
                vertex_weights,
                )

            # Partition hypergraph
            kahypar.partition(kahypar_hg, self.khp_context)

            self.property_set["partitioned_hyper_dag"] = PartitionedHyperGraph(kahypar_hg)
        else:
            # TODO: 
            self.property_set["partitioned_hyper_dag"] = PartitionedHyperGraph(num_nodes = num_vertices)


        # TODO: Do some visualization, so see if for lattice surgery, it is possible to lay out the partitions without any
        #       edges intersecting each other. If there are intersecting edges, this is a big problem for the routing, 
        #       since these connections need to be routed through a whole other qpu.
        #       Goal: We do not want any intersection of edges of the partitioned graph

        return dag

    def calculate_partitions(self, dag: DAGCircuit) -> Tuple[int, List[int]]:
        """Calculate number of partitions

        This is based on:
            - Backend size of each QPU (are all QPUs the same size, and are all qubits working?)
            - For lattice surgery, detect the patch size and how many can be placed on the backend

        Different methods:
            - full: Fill single chips as much as possible (for lattice surgery potentially not ideal)
            - patch-aware: Keep patches together and distribute
        
        :param dag: _description_
        :type dag: DAGCircuit
        :return: _description_
        :rtype: Tuple[int, List[int]]
        """
        # TODO: Calculate into how many partitions the circuit should be split.
        #       This is based on a number of things:
        #           - Backend size of each QPU (are all QPUs the same size, and are all qubits working?)
        #           - For lattice surgery, detect the patch size and how many can be placed on the backend

        

        needs_partitioning = True

        num_qubits_chiplet = self.backend.get_chip_size()
        
        num_qubits_circuit = dag.num_qubits()

        # Perform partitioning, if circuit does not fit on on chiplet
        if num_qubits_chiplet < num_qubits_circuit:
            if self._calculate_partitions_method == "full":
                # Fill chiplet as much as possible
                k = int(np.ceil(num_qubits_circuit / num_qubits_chiplet))
                print(k)
            elif self._calculate_partitions_method == "patch-aware":
                # Approach to keep patches together:
                #   - Try to find patches in the circuit
                #   - How many patches can be place on a single chiplet
                #   - Distribute the patches to all chiplets:
                #       - Simply distributed patches if more chiplets than patches
                #       - If more patches than chiplets, try to have as many as possible good patches, and some bad ones.
                #         TODO: Find out a better way how to handle this
                
                pass
            else:
                pass
        else:
            k = 1


        # Set size of each partition as number of qubits on a chiplet
        partition_sizes = [num_qubits_chiplet for c in range(k)]

        return k, partition_sizes

    def cost_analysis(self):
        """Calculate statistics of partitioned graph
        """

        # Output metrics
        pass