#!/bin/bash
cd glue/qiskit

# Build rust dependencies
python3 setup.py build_rust --release --inplace

# Rebuild qiskit with new rust crates
pip3 install .