> [!WARNING]
> This code is still under development and features may change without prior warning.

# ecc_mapping
Mapping QECC to distributed systems


## Tasks



## Implementation


### Main libraries
- Circuit representaiton: [qiskit](https://github.com/Qiskit/qiskit)
- Simulator: [Stim](https://github.com/quantumlib/Stim/tree/main?tab=readme-ov-file)
- Benchmarks: [eccentric_bench](https://github.com/aswierkowska/eccentric_bench)

Also implemented custom transpiler passes in qiskit (see glue/qiskit)
It is necessary to build all rust crates in install qiskit locally. See dev/build_rust.sh



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
The mapping-and-routing is implemented for seamless integration into the transpilation pipeline of qiskit

- qiskit


### Constructing circuits and experiments
For running the experiments the following additional libraries are necessary:

memory experiments:<br>
-[eccentric_bench](https://github.com/aswierkowska/eccentric_bench/): Follow installation instructions in repo

stability: <br>
- TODO

logical circuit: <br>
- [Topologiq](https://github.com/jbolns/topologiq): `pip install git+https://github.com/jbolns/topologiq.git`
- [TQEC](https://github.com/tqec/tqec): `pip install git+https://github.com/tqec/tqec.git`


### Mapping

- [KaHyPar](https://github.com/kahypar/kahypar)

- hypernetx (only for visualizing)

- networkx

- Potential replacement for KaHyPar: [Kaminpar](https://github.com/KaHIP/KaMinPar)
