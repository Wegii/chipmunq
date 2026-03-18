# <div align="center">Chipmunq 🐿</div>
<div align="center">Peter Wegmann, Aleksandra Świerkowska, Emmanouil Giortamis, and Pramod Bhatotia
<br><br></div>

</div> 
<div align="center">
![arXiv]([https://img.shields.io/badge/arXiv-2301.12345-red?style=flat-square](https://img.shields.io/badge/arXiv-2603.16389-red?style=flat-square))
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18985440.svg)](https://doi.org/10.5281/zenodo.18985440)
</div>

> Chipmunq is a compiler that enables mapping and routing FT circuits onto both monolithic and chiplet-based architectures in a scalable, QEC-patch-preserving, and architecture-aware manner, aligning with the evolving demands of near- and long-term quantum modular hardware architectures.


## Key Features of Chipmunq
- **The Chipmunq compiler:** The first scalable, hardware-aware mapping and routing [`framework`](qeccm/) that bridges the gap between topological QEC codes and heterogeneous chiplet-based architectures.
  
- **Noise-aware heuristics:**  A novel inter-chiplet [`routing strategy`](qeccm/src/router.py) that incorporates link fidelity and congestion awareness, successfully suppressing logical error rates by up to two orders of magnitude in heterogeneous noise environments while maintaining QEC effectiveness below correction thresholds.

- **Scalability benchmarking:** An [`evaluation`](experiments/) demonstrating that Chipmunq delivers 13.5$\times$ speedup on average in compilation time and an average 91.4\% reduction in SWAP overhead compared to LightSABRE.


## How do I use Chipmunq?
See the utilization in [`experiments`](experiments/) and the existing [`compilation functions`](experiments/exp_utils/transpilation_utils.py#L15C5-L15C37).


## Installation
Chipmunq can easily be installed by running the [`install.sh`](install.sh) script.


## Citation
```
@misc{wegmann2026chipmunqfaulttolerantcompilerchiplet,
      title={Chipmunq: A Fault-Tolerant Compiler for Chiplet Quantum Architectures }, 
      author={Peter Wegmann and Aleksandra Świerkowska and Emmanouil Giortamis and Pramod Bhatotia},
      year={2026},
      eprint={2603.16389},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2603.16389}, 
}
```
