mkdir -p experiments/evaluation/timing

# Experiments for evaluating runtime of qecc-synth and MECH
time python3 experiments/related_work_exps/experiment_runtime_mono.py > experiments/evaluation/timing/rw_runtime.txt 2>&1
# Experiments for evaluating two-qubit gate utilization of qecc-synth and MECH
time python3 experiments/related_work_exps/experiment_statistics.py > experiments/evaluation/timing/rw_statistics.txt 2>&1

# How does the performance of the proposed implementation scale as circuit complexity increases?
time python3 experiments/scalability_exps/experiment_scalability.py > experiments/evaluation/timing/s_runtime.txt 2>&1
# How do circuit depth and gate overhead scale as circuit size increases?
time python3 experiments/scalability_exps/experiment_statistics.py > experiments/evaluation/timing/s_statistics.txt 2>&1
# How does the number and noise level of inter-chiplet connections influence circuit routing?
time python3 experiments/scalability_exps/experiment_inter_chiplet.py > experiments/evaluation/timing/s_inter_c.txt 2>&1
# How do defective qubits affect the resulting circuit?
time python3 experiments/scalability_exps/experiment_defective_qubits.py > experiments/evaluation/timing/s_defect.txt 2>&1

# How does the logical error rate of lattice surgery operations change when distributed to multiple chiplets?
time python3 experiments/qec_exps/experiment_distributed_lattice_surgery.py > experiments/evaluation/timing/qec_dist.txt 2>&1
# How is the logical error rate influenced by a limited number of inter-chiplet connections with varying error rates?
time python3 experiments/qec_exps/experiment_limited_inter_chiplet_connectivity.py > experiments/evaluation/timing/qec_inter_c.txt 2>&1
# How is the logical error rate influenced by focusing on different metrics during routing
time python3 experiments/qec_exps/experiment_noise_aware_routing.py > experiments/evaluation/timing/qec_routing.txt 2>&1