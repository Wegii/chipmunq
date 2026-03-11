from qec_exps.experiment_distributed_lattice_surgery import *
from qec_exps.experiment_limited_inter_chiplet_connectivity import *
from qec_exps.experiment_noise_aware_routing import *
from related_work_exps.experiment_runtime_mono import *
from related_work_exps.experiment_statistics import *
from scalability_exps.experiment_defective_qubits import *
from scalability_exps.experiment_inter_chiplet import *
from scalability_exps.experiment_scalability import *
from scalability_exps.experiment_statistics import *

import time


def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"


reproduce_results = True

functions = [
    ("run_runtime_scaling", lambda: run_runtime_scaling(reproduce=reproduce_results)),
    ("run_statistics", lambda: run_statistics(reproduce=reproduce_results)),
    ("run_exp_scalability", lambda: run_exp_scalability(reproduce=reproduce_results)),
    ("run_exp_statistics", lambda: run_exp_statistics(reproduce=reproduce_results)),
    ("run_exp_inter_chiplet", lambda: run_exp_inter_chiplet(reproduce=reproduce_results)),
    ("run_exp_defective", lambda: run_exp_defective(reproduce=reproduce_results)),
    ("run_exp_distributed_lattice_surgery", lambda: run_exp_distributed_lattice_surgery(reproduce=reproduce_results)),
    ("run_exp_distributed_inter_chiplet", lambda: run_exp_distributed_inter_chiplet(reproduce=reproduce_results)),
    ("run_noise_aware_routing", lambda: run_noise_aware_routing(reproduce=reproduce_results)),
]

results = []
for name, fn in functions:
    start = time.time()
    fn()
    duration = time.time() - start
    results.append(f"{name}: {format_duration(duration)}")

with open("experiments/execution_times.txt", "w") as f:
    f.write("\n".join(results))
