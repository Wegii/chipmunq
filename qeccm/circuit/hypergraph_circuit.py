from qiskit import QuantumCircuit

# Represent circuit as hypergraph

# TODO: Add function for partitioning


# visualize circuit
# Have a look at the https://github.com/felix-burt/DISQCO regarding visualization


class HypergraphCircuit():
    
    def __init__(self, circuit: QuantumCircuit):
        

        self.circuit = circuit
        # Hypergraph circuit
        self.hgc = None


    def partition_circuit(self):
        pass