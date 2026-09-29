"""
Qiskit wrapper around OLSQ2 (https://github.com/WanHsuanLin/OLSQ2).

    routed = olsq2_transpilation(circuit, backend)

`backend` can be a Qiskit BackendV2 (e.g. GenericBackendV2, FakeSherbrooke),
a qiskit.transpiler.CouplingMap, or a plain edge list [(0, 1), (1, 2), ...].

Setup:
    git clone https://github.com/WanHsuanLin/OLSQ2.git
    pip install z3-solver python-sat qiskit
    export PYTHONPATH=$PYTHONPATH:/path/to/OLSQ2      # so that `import olsq` works
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.getcwd(), "."))


import contextlib
import io
import os
import sys
import tempfile
import types

from qiskit import QuantumCircuit
from qiskit.circuit.library import SwapGate
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import Unroll3qOrMore, RemoveBarriers, SabreLayout


# --- Compatibility shim -------------------------------------------------------
# OLSQ2's olsq/run_h_compiler.py imports `Unroller`, which was removed in Qiskit 1.0.
# OLSQ2 always calls run_sabre() for an upper bound, so we register a drop-in
# replacement that works with current Qiskit before importing olsq.
def _run_sabre(circuit_info, coupling, count_physical_qubit):
    qc = QuantumCircuit(count_physical_qubit)
    for g in circuit_info:
        if len(g) == 2:
            qc.cx(g[0], g[1])
        elif len(g) == 1:
            qc.h(g[0])
        else:
            raise TypeError("Currently only support one and two-qubit gate.")
    edges = list(coupling) + [(b, a) for a, b in coupling]
    cmap = CouplingMap(couplinglist=edges)
    routed = PassManager(SabreLayout(cmap, seed=0)).run(qc)  # layout + routing
    return routed.count_ops().get("swap", 0), routed.depth()


if "olsq.run_h_compiler" not in sys.modules:
    _shim = types.ModuleType("olsq.run_h_compiler")
    _shim.run_sabre = _run_sabre
    sys.modules["olsq.run_h_compiler"] = _shim

from olsq import OLSQ  # noqa: E402
from olsq.device import qcdevice  # noqa: E402
# ------------------------------------------------------------------------------


def _edges_from_backend(backend):
    """Return (num_physical_qubits, undirected edge list) for various backend types."""
    if isinstance(backend, CouplingMap):
        cmap = backend
    elif isinstance(backend, (list, tuple)):
        cmap = CouplingMap(couplinglist=[tuple(e) for e in backend])
    elif getattr(backend, "coupling_map", None) is not None:
        cmap = backend.coupling_map
        if not isinstance(cmap, CouplingMap):  # BackendV1-style list
            cmap = CouplingMap(couplinglist=cmap)
    elif getattr(backend, "target", None) is not None:
        cmap = backend.target.build_coupling_map()
    else:
        raise TypeError("backend must be a Qiskit backend, a CouplingMap, or an edge list")

    edges = sorted({tuple(sorted((int(a), int(b)))) for a, b in cmap.get_edges() if a != b})
    return cmap.size(), edges


def olsq2_transpilation(
    circuit: QuantumCircuit,
    backend,
    objective: str = "swap",        # "swap" or "depth"
    mode: str = "transition",       # "transition" (TB-OLSQ2, fast) or "normal" (exact time)
    use_sabre_bound: bool = True,   # SABRE upper bound to seed SWAP optimization
    swap_duration: int = 1,         # time units per SWAP (only 1 or 3 supported by OLSQ2)
    encoding: int = 1,              # pySAT cardinality encoding (see OLSQ2 README)
    verbose: bool = False,
) -> QuantumCircuit:
    """Map and route `circuit` onto `backend` with OLSQ2.

    Returns a QuantumCircuit on the backend's physical qubits with explicit SwapGates.
    circuit.metadata gets 'olsq2_initial_layout' and 'olsq2_final_layout'
    (lists: virtual qubit index -> physical qubit), plus 'olsq2_swap_count'
    and 'olsq2_depth'.
    """
    if objective not in ("swap", "depth"):
        raise ValueError("objective must be 'swap' or 'depth'")
    if swap_duration not in (1, 3):
        raise ValueError("OLSQ2 only supports swap_duration 1 or 3")

    n_phys, edges = _edges_from_backend(backend)
    if circuit.num_qubits > n_phys:
        raise ValueError(f"circuit has {circuit.num_qubits} qubits, backend only {n_phys}")

    # 1. OLSQ2 only handles 1- and 2-qubit operations; barriers carry no data dependency.
    circ = PassManager([Unroll3qOrMore(), RemoveBarriers()]).run(circuit)

    # 2. Build OLSQ IR directly (avoids OLSQ2's fragile QASM parser).
    #    Gate "names" are indices into `ops`, so we can rebuild exact Qiskit ops later.
    dag = circuit_to_dag(circ)
    qidx = {q: i for i, q in enumerate(circ.qubits)}
    ops, gates, names = [], [], []
    for node in dag.topological_op_nodes():
        if getattr(node.op, "condition", None) is not None:
            raise NotImplementedError("classically conditioned gates are not supported")
        qs = tuple(qidx[q] for q in node.qargs)
        if len(qs) == 0:
            continue
        ops.append((node.op, node.cargs))
        gates.append(qs)
        names.append(f"g{len(ops) - 1}")
    if not any(len(g) == 2 for g in gates):
        raise ValueError("circuit has no two-qubit gates; nothing to route")

    # 3. Solve. Run in a temp dir: OLSQ2 writes intermediate qasm files during SWAP optimization.
    solver = OLSQ(obj_is_swap=(objective == "swap"), mode=mode, encoding=encoding)
    solver.setdevice(qcdevice(name="dev", nqubits=n_phys, connection=edges,
                              swap_duration=swap_duration))
    solver.setprogram((circ.num_qubits, tuple(gates), tuple(names)), input_mode="IR")

    cwd = os.getcwd()
    sink = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with tempfile.TemporaryDirectory() as tmp, sink:
        os.chdir(tmp)
        try:
            result = solver.solve(use_sabre=use_sabre_bound and objective == "swap",
                                  output_mode="IR")
        finally:
            os.chdir(cwd)
    depth, sched_names, sched_qubits, final_map, init_map, _ = result

    # 4. Rebuild a Qiskit circuit. Within each time step OLSQ applies SWAPs after gates.
    out = QuantumCircuit(n_phys, name=f"{circuit.name}_olsq2")
    for creg in circ.cregs:
        out.add_register(creg)
    loose_clbits = [c for c in circ.clbits if c not in {b for r in circ.cregs for b in r}]
    if loose_clbits:
        out.add_bits(loose_clbits)

    n_swaps = 0
    for t in range(depth):
        swaps_here = []
        for name, qs in zip(sched_names[t], sched_qubits[t]):
            if name == "SWAP":
                swaps_here.append(("swap", qs))
            elif name == "cx":  # swap_duration=3: SWAP decomposed into 3 CX by OLSQ2
                swaps_here.append(("cx", qs))
            else:
                op, cargs = ops[int(name[1:])]
                out.append(op, [out.qubits[p] for p in qs], list(cargs))
        for kind, (a, b) in swaps_here:
            if kind == "swap":
                out.append(SwapGate(), [a, b])
                n_swaps += 1
            else:
                out.cx(a, b)
    if swap_duration == 3:
        n_swaps = out.count_ops().get("cx", 0) - sum(
            1 for op, _ in ops if op.name == "cx")
        n_swaps //= 3

    out.metadata = dict(circuit.metadata or {})
    out.metadata.update(
        olsq2_initial_layout=list(init_map),
        olsq2_final_layout=list(final_map),
        olsq2_swap_count=n_swaps,
        olsq2_depth=depth,
    )
    return out


if __name__ == "__main__":
    from qiskit.providers.fake_provider import GenericBackendV2

    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 0)
    qc.rz(0.3, 2)
    qc.measure(range(3), range(3))

    # 5-qubit line: 0-1-2-3-4 (no triangles, so at least one SWAP is needed)
    backend = GenericBackendV2(num_qubits=5, coupling_map=[[0, 1], [1, 2], [2, 3], [3, 4]], seed=0)

    routed = olsq2_transpilation(qc, backend)
    print(routed.draw(fold=-1))
    print(routed.metadata)