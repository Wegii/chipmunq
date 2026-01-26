from __future__ import annotations

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "."))
sys.path.append(os.path.join(os.getcwd(), "glue/eccentric_bench/"))

# MECH
sys.path.append(os.path.join(os.getcwd(), "./external/baseline/MECH"))
from external.baseline.MECH.Circuit import *
from external.baseline.MECH.Chiplet import *
from external.baseline.MECH.HighwayOccupancy import *
from external.baseline.MECH.Router import *
from external.baseline.MECH.MECHBenchmarks import *
from external.baseline.MECH.transpile_mech import transpile_circuit_MECH
import networkx as nx
from networkx.classes import Graph

# QECC-Synth
sys.path.append(os.path.join(os.getcwd(), "./external/baseline/QECC_Synth/SurfStitch/MyCode/src"))
from external.baseline.QECC_Synth.SurfStitch.MyCode.src.transpile_qeccsynth import transpile_circuit_QECCSynth


def run_runtime_scaling():
    pass


if __name__ == "__main__":
    run_runtime_scaling()
    