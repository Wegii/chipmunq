# ecc_mapping
Mapping QECC to distributed systems




## Implementation


### Rust
- Graph Library: e.g. use qiskit rustworkx [rustworkx](https://github.com/Qiskit/rustworkx)
- Circuit representaiton: [qiskit](https://github.com/Qiskit/qiskit)
- Simulator: [Stim](https://github.com/quantumlib/Stim/tree/main?tab=readme-ov-file)
- Benchmarks: [eccentric_bench](https://github.com/aswierkowska/eccentric_bench)




## Comparison with other mapping algorithms

### SABRE
- Publication: https://arxiv.org/abs/2409.08368
- Utilizes the default (Light-) SABRE algorithm implemented in qiskit


### MECH
- Publication: https://arxiv.org/html/2305.05149v4


### QECC-Synth
- Publication: https://arxiv.org/abs/2308.06428
- works for given architecture
- does not work with the coupling graph generated from MECH -> now it works: CG needs to be a np.array


### Lattice Surgery Compilation Beyond the Surface Code
- Pre-print: https://arxiv.org/abs/2504.10591
- See [implementation](https://github.com/munich-quantum-toolkit/qecc/tree/ls-compilation/scripts/co3)




# Installation

## Dependencies

Qiskit


Install KaHyPar
- See[KaHyPar](https://github.com/kahypar/kahypar)
- See [Kaminpar](https://github.com/KaHIP/KaMinPar)

## Build from source
