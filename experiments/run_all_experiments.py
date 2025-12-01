from scalability_exps.experiment_scalability import *
from scalability_exps.experiment_statistics import *
from scalability_exps.experiment_inter_chiplet import *
from scalability_exps.experiment_defective_qubits import *

from qec_exps.experiment_distributed_lattice_surgery import *
from qec_exps.experiment_limited_inter_chiplet_connectivity import *


run_exp_scalability()

run_exp_statistics()

run_exp_inter_chiplet()

run_exp_defective()



run_exp_distributed_lattice_surgery()

run_exp_distributed_inter_chiplet()