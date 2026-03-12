mkdir -p experiments/evaluation/timing

# Experiments for evaluating runtime of qecc-synth and MECH
python3 experiments/related_work_exps/experiment_runtime_mono.py
# Experiments for evaluating two-qubit gate utilization of qecc-synth and MECH
python3 experiments/related_work_exps/experiment_statistics.py

# How does the performance of the proposed implementation scale as circuit complexity increases?
python3 experiments/scalability_exps/experiment_scalability.py
# How do circuit depth and gate overhead scale as circuit size increases?
python3 experiments/scalability_exps/experiment_statistics.py
# How does the number and noise level of inter-chiplet connections influence circuit routing?
python3 experiments/scalability_exps/experiment_inter_chiplet.py
# How do defective qubits affect the resulting circuit?
python3 experiments/scalability_exps/experiment_defective_qubits.py

# How does the logical error rate of lattice surgery operations change when distributed to multiple chiplets?
python3 experiments/qec_exps/experiment_distributed_lattice_surgery.py
# How is the logical error rate influenced by a limited number of inter-chiplet connections with varying error rates?
python3 experiments/qec_exps/experiment_limited_inter_chiplet_connectivity.py
# How is the logical error rate influenced by focusing on different metrics during routing
python3 experiments/qec_exps/experiment_noise_aware_routing.py