from qec_exps.experiment_distributed_lattice_surgery import *
from qec_exps.experiment_limited_inter_chiplet_connectivity import *
from qec_exps.experiment_noise_aware_routing import *
from related_work_exps.experiment_runtime_mono import *
from related_work_exps.experiment_statistics import *
from scalability_exps.experiment_defective_qubits import *
from scalability_exps.experiment_inter_chiplet import *
from scalability_exps.experiment_scalability import *
from scalability_exps.experiment_statistics import *


# Set to True if the results should be reproduced. False for utilizing pre-computed results and
# generate plots only
reproduce_results = True


# Experiments for evaluating runtime of qecc-synth and MECH
run_runtime_scaling(reproduce=reproduce_results)

# Experiments for evaluating two-qubit gate utilization of qecc-synth and MECH
run_statistics(reproduce=reproduce_results)

# How does the performance of the proposed implementation scale as circuit complexity increases?
run_exp_scalability(reproduce=reproduce_results)

# How do circuit depth and gate overhead scale as circuit size increases?
run_exp_statistics(reproduce=reproduce_results)

# How does the number and noise level of inter-chiplet connections influence circuit routing?
run_exp_inter_chiplet(reproduce=reproduce_results)

# How do defective qubits affect the resulting circuit?
run_exp_defective(reproduce=reproduce_results)


# How does the logical error rate of lattice surgery operations change when distributed to multiple chiplets?
run_exp_distributed_lattice_surgery(reproduce=reproduce_results)

# How is the logical error rate influenced by a limited number of inter-chiplet connections with varying error rates?
run_exp_distributed_inter_chiplet(reproduce=reproduce_results)

# How is the logical error rate influenced by focusing on different metrics during routing
run_noise_aware_routing(reproduce=reproduce_results)
# Perfor noise_aware_routing by performing random sweep over a wide range of hyperparameters. Excluded
# since this can take up to a day for finishing all runs.
# perform_noise_aware_routing_sweep(reproduce = reproduce_results)
