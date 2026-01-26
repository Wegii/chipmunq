from scalability_exps.experiment_scalability import *
from scalability_exps.experiment_statistics import *
from scalability_exps.experiment_inter_chiplet import *
from scalability_exps.experiment_defective_qubits import *
from scalability_exps.experiment_surgery_scalability import *

from qec_exps.experiment_distributed_lattice_surgery import *
from qec_exps.experiment_limited_inter_chiplet_connectivity import *
from qec_exps.experiment_noise_aware_routing import *


# Experiments for evaluating qecc-synth and MECH
run_runtime_scaling()




# How does the performance of the proposed implementation scale as circuit complexity increases?
run_exp_scalability()


# How do circuit depth and gate overhead scale as circuit size increases?
run_exp_statistics()


# How does the number of inter-chiplet connections influence circuit routing?
# How does the error of inter-chiplet connections influence circuit routing?
run_exp_inter_chiplet()



# How do defective qubits affect the resulting circuit?
run_exp_defective()


# How long does lattice surgery (with and without mapping+routing) take for different surgery libraries
perform_surgery_comparison()


# How does the logical error rate of lattice surgery operations change when distributed to multiple chiplets?
run_exp_distributed_lattice_surgery()


# How is the logical error rate influenced by a limited number of inter-chiplet connections with varying error rates?
run_exp_distributed_inter_chiplet()


# How is the logical error rate influenced by focusing on different metrics during routing 
perform_noise_aware_routing()