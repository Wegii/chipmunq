"""OLSQ2, SEQC and Murali et al. baselines for the related-work experiments.

Shared by run_statistics and run_runtime_scaling so that both use the same backend
construction, timeouts and plot styling.

Every method is called as ``fn(circuit, coupling_map) -> QuantumCircuit``, with the
coupling map that LightSABRE also receives (``generate_qiskit_backend_from_mech``). The
routed circuits keep explicit SWAP gates and the input gate set, so they can go through
``calc_circuit_qiskit_stats`` exactly like the LightSABRE output.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import queue
import sys
import time
import traceback
from pathlib import Path

sys.path.append(os.path.join(os.getcwd(), "."))

# --- Baseline wrappers ---------------------------------------------------------------
# Adjust these three imports if the wrapper files live elsewhere.
sys.path.append(os.path.join(os.getcwd(), "./external/baseline/qls"))
sys.path.append(os.path.join(os.getcwd(), "./external/baseline/qls/OLSQ2"))  # `import olsq`
sys.path.append(os.path.join(os.getcwd(), "./external/baseline/seqc"))  # `import olsq`
sys.path.append(os.path.join(os.getcwd(), "./external/baseline/noise_aware_mapping"))  # `import olsq`

from external.baseline.qls.qlsq2 import olsq2_transpilation  # noqa: E402
from external.baseline.seqc.seqc import SEQCCompiler  # noqa: E402
from external.baseline.noise_aware_mapping.murali import noise_adaptive_transpilation  # noqa: E402

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit import Parameter  # noqa: E402
from qiskit.circuit.library import SwapGate, get_standard_gate_name_mapping  # noqa: E402
from qiskit.transpiler import CouplingMap, Target  # noqa: E402

# Wall-clock limit per (method, distance) for *every* method: a run that takes longer is killed and
# marked TIMEOUT. Timed-out bars in the runtime plots are drawn at this height.
TIMEOUT_S = 30000.0

# Time a child process may spend starting up (spawn, imports, unpickling the arguments) before the
# transpilation clock starts. Not counted towards TIMEOUT_S.
STARTUP_TIMEOUT_S = 600.0

# R-SMT* (the exact MILP placement) builds pairs x n_phys^2 variables and does not fit in
# memory for surface-code circuits beyond the smallest distance. GreedyE* is the scalable
# heuristic from the same paper (Sec. 5.2).
MURALI_METHOD = "greedy_e"

SEED = 0

# Qiskit transpiler
import qiskit
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager, StagedPassManager
from qiskit.transpiler.passes import ApplyLayout, TrivialLayout, Unroll3qOrMore
from qiskit.transpiler.passes.layout.enlarge_with_ancilla import EnlargeWithAncilla
from qiskit.transpiler.passes.layout.full_ancilla_allocation import FullAncillaAllocation

from qeccm.backends.BackendChipletV2 import BackendChipletV2

# Custom implementation
from qeccm.src.mar import PartitionedMapRoutePlugin




def custom_partitioned_transpilation(
    circuit: QuantumCircuit, backend: BackendChipletV2, pre_defined_partitions: list = None
) -> qiskit.QuantumCircuit:
    """Transpile circuit to a chiplet backend using custom mapping and routing.

    :param circuit: _description_
    :type circuit: QuantumCircuit
    :param backend: _description_
    :type backend: BackendChipletV2
    :return: _description_
    :rtype: qiskit.QuantumCircuit
    """
    mar_pmsp = PartitionedMapRoutePlugin()
    # Pass to construct hypergraph from circuit
    init_pm = mar_pmsp._generate_initial_pass()
    # Pass to perform partition and mapping
    partitioning_pm = mar_pmsp._generate_layout_pass(backend, partitions=pre_defined_partitions)
    # Pass to perform routing
    routing_pm = mar_pmsp._generate_routing_pass(backend)
    # Construct pass manager with all passes
    staged_pm = StagedPassManager(
        stages=["init", "layout", "routing"], init=init_pm, layout=partitioning_pm, routing=routing_pm
    )
    # staged_pm = StagedPassManager(stages=["init", "layout"], init=init_pm, layout=partitioning_pm)
    # Run passes
    routed_circuit = staged_pm.run(circuit)

    return routed_circuit


def custom_cost_transpilation(
    circuit: QuantumCircuit,
    backend: BackendChipletV2,
    pre_defined_partitions: list = None,
    routing_alpha: float = 0.0,
    routing_beta: float = 0.0,
    patch_initialization: str = "",
) -> QuantumCircuit:

    # Initialize transpilation plugin in order to run the different passes
    mar_pmsp = PartitionedMapRoutePlugin()
    # Pass to construct hypergraph from circuit
    init_pm = mar_pmsp._generate_initial_pass()
    # Pass to perform partition and mapping
    partitioning_pm = mar_pmsp._generate_layout_pass(
        backend, partitions=pre_defined_partitions, patch_initialization=patch_initialization
    )
    # Perform routing utilizing cost routing
    routing_pm = mar_pmsp._generate_routing_pass(backend, routing_type="cost", alpha=routing_alpha, beta=routing_beta)
    # Construct pass manager with all passes
    staged_pm = StagedPassManager(
        stages=["init", "layout", "routing"], init=init_pm, layout=partitioning_pm, routing=routing_pm
    )

    # Run passes
    routed_circuit = staged_pm.run(circuit)

    return routed_circuit


def custom_accelerated_partitioned_transpilation(
    circuit: QuantumCircuit, backend: BackendChipletV2
) -> qiskit.QuantumCircuit:
    """Transpile circuit to a chiplet backend using custom mapping and accelerated routing.

    :param circuit: _description_
    :type circuit: QuantumCircuit
    :param backend: _description_
    :type backend: BackendChipletV2
    :return: _description_
    :rtype: qiskit.QuantumCircuit
    """
    # Custom implementation of basic routing

    from qeccm.src.router import AcceleratedBasicSwapRouter

    mar_pmsp = PartitionedMapRoutePlugin()
    # Pass to construct hypergraph from circuit
    init_pm = mar_pmsp._generate_initial_pass()
    # Pass to perform partition and mapping
    partitioning_pm = mar_pmsp._generate_layout_pass(backend)
    # Pass to perform routing
    # Pre-routing pass
    routing_pm = PassManager([EnlargeWithAncilla(), ApplyLayout(), AcceleratedBasicSwapRouter(backend)])

    # Construct pass manager with all passes
    staged_pm = StagedPassManager(
        stages=["init", "layout", "routing"], init=init_pm, layout=partitioning_pm, routing=routing_pm
    )
    # Run passes
    routed_circuit = staged_pm.run(circuit)

    return routed_circuit


def basicswap_transpilation(circuit: QuantumCircuit, backend: BackendChipletV2) -> qiskit.QuantumCircuit:
    """Transpile circuit to a chiplet backend using basic swap mapping and routing.

    :param circuit: _description_
    :type circuit: QuantumCircuit
    :param backend: _description_
    :type backend: BackendChipletV2
    :return: _description_
    :rtype: qiskit.QuantumCircuit
    """
    init_pm = PassManager([Unroll3qOrMore()])

    layout_pm = PassManager([TrivialLayout(backend.coupling_map), FullAncillaAllocation(backend.coupling_map)])

    routing_op = qiskit.transpiler.passes.BasicSwap(coupling_map=CouplingMap(backend.coupling_map))
    router_pm = PassManager([EnlargeWithAncilla(), ApplyLayout(), routing_op])
    staged_pm = StagedPassManager(
        stages=["init", "layout", "routing"], init=init_pm, layout=layout_pm, routing=router_pm
    )

    return staged_pm.run(circuit)


def sabre_transpilation(circuit: QuantumCircuit, backend: BackendChipletV2) -> qiskit.QuantumCircuit:
    """Transpile circuit to a chiplet backend using SABRE mapping and routing.

    :param circuit: _description_
    :type circuit: QuantumCircuit
    :param backend: _description_
    :type backend: BackendChipletV2
    :return: _description_
    :rtype: qiskit.QuantumCircuit
    """
    init_pm = PassManager([Unroll3qOrMore()])

    # TODO: Change this to SABRE layout
    layout_pm = PassManager([TrivialLayout(backend.coupling_map), FullAncillaAllocation(backend.coupling_map)])

    routing_op = qiskit.transpiler.passes.SabreSwap(
        coupling_map=CouplingMap(backend.coupling_map), heuristic="decay", seed=42
    )
    router_pm = PassManager([EnlargeWithAncilla(), ApplyLayout(), routing_op])
    staged_pm = StagedPassManager(
        stages=["init", "layout", "routing"], init=init_pm, layout=layout_pm, routing=router_pm
    )

    return staged_pm.run(circuit)








# --------------------------------------------------------------------------------------
# Backend helpers
# --------------------------------------------------------------------------------------


def as_coupling_map(cm) -> CouplingMap:
    """Accept a CouplingMap or an edge list (both forms are used for LightSABRE)."""
    if isinstance(cm, CouplingMap):
        return cm
    return CouplingMap(couplinglist=[tuple(e) for e in cm])


STIM_ANNOTATIONS = {"DETECTOR", "OBSERVABLE_INCLUDE", "TICK", "QUBIT_COORDS", "SHIFT_COORDS"}


def strip_annotations(circuit: QuantumCircuit) -> QuantumCircuit:
    """Copy of ``circuit`` without Stim annotation instructions."""
    out = circuit.copy_empty_like()
    for inst in circuit.data:
        if inst.operation.name.upper() not in STIM_ANNOTATIONS:
            out.append(inst)
    return out

def build_seqc_target(circuit: QuantumCircuit, cm, chiplets=None) -> Target:
    """Error-free Target on the coupling map, supporting the circuit's own gate set.

    SEQC needs a Target, not a coupling map. Every gate of the input circuit is native on
    every qubit/edge and SWAP is native on every edge, so SEQC's basis-translation stage is
    an identity and its output stays comparable (same gates + explicit SWAPs) to LightSABRE.

    ``chiplets``: optional partition of the physical qubits. Edges between chiplets then
    support only SWAP (SEQC's non-universal link model). ``None`` means one monolithic chip.
    """
    cmap = as_coupling_map(cm)
    n = cmap.size()
    edges = sorted({tuple(sorted(e)) for e in cmap.get_edges() if e[0] != e[1]})
    chip_of = {}
    if chiplets is not None:
        for c, qs in enumerate(chiplets):
            for q in qs:
                chip_of[q] = c
    intra = [e for e in edges if chip_of.get(e[0], 0) == chip_of.get(e[1], 0)]
    directed = lambda es: [(a, b) for a, b in es] + [(b, a) for a, b in es]  # noqa: E731

    std = get_standard_gate_name_mapping()
    target = Target(num_qubits=n)
    names = {n for n in circuit.count_ops() if n.upper() not in STIM_ANNOTATIONS}
    names -= {"barrier", "swap"}
    names |= {"measure", "reset"}
    for name in sorted(names):
        if name not in std:
            raise ValueError(f"SEQC target: gate '{name}' is not a standard Qiskit gate.")
        op = std[name]
        if op.params:  # parametrised gates need a symbolic parameter in the Target
            op = op.__class__(*[Parameter(f"{name}_{i}") for i in range(len(op.params))])
        if op.num_qubits == 1:
            target.add_instruction(op, {(q,): None for q in range(n)})
        elif op.num_qubits == 2:
            target.add_instruction(op, {e: None for e in directed(intra)})
        else:
            raise ValueError(f"SEQC target: {op.num_qubits}-qubit gate '{name}' is not supported.")
    target.add_instruction(SwapGate(), {e: None for e in directed(edges)})
    return target


# --------------------------------------------------------------------------------------
# Transpilation entry points
# --------------------------------------------------------------------------------------


def transpile_circuit_OLSQ2(circuit: QuantumCircuit, coupling_map) -> QuantumCircuit:
    return olsq2_transpilation(circuit, as_coupling_map(coupling_map), objective="swap",
                               mode="transition")


def transpile_circuit_SEQC(circuit: QuantumCircuit, coupling_map, chiplets=None) -> QuantumCircuit:
    circuit = strip_annotations(circuit)
    target = build_seqc_target(circuit, coupling_map, chiplets)

    if chiplets is None:
        chiplets = [list(range(target.num_qubits))]
    # n_jobs=1: SEQC parallelises per chiplet; the other baselines run single-threaded.
    # optimization_level=0: no gate cancellation, so the 2q-gate overhead is routing only.
    compiler = SEQCCompiler(target, chiplets=chiplets, optimization_level=0, seed=SEED, n_jobs=1)
    return compiler.run(circuit)


def transpile_circuit_Murali(circuit: QuantumCircuit, coupling_map) -> QuantumCircuit:
    """``coupling_map`` may be a full backend (noise-adaptive: it sees the inter-chiplet error rates) or a
    CouplingMap / edge list (noise-unaware fallback)."""
    if hasattr(coupling_map, "coupling_map"):  # a backend: pass it through unchanged
        return noise_adaptive_transpilation(circuit, coupling_map, method=MURALI_METHOD)
    return noise_adaptive_transpilation(circuit, as_coupling_map(coupling_map), method=MURALI_METHOD)


EXTRA_METHODS = {
    "olsq2": transpile_circuit_OLSQ2,
    "seqc": transpile_circuit_SEQC,
    "murali": transpile_circuit_Murali,
}


# --------------------------------------------------------------------------------------
# Timeout wrapper
# --------------------------------------------------------------------------------------


def _worker(q, fn, args, kwargs):
    # Own process group, so a timeout also kills solver subprocesses started by ``fn``
    try:
        os.setsid()
    except (AttributeError, OSError):
        pass
    q.put(("started", None, 0.0))  # imports and argument unpickling are done: start the clock
    start = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        q.put(("ok", result, time.perf_counter() - start))
    except BaseException:  # report any failure to the parent instead of hanging it
        q.put(("error", traceback.format_exc(), time.perf_counter() - start))


TIMEOUT = "T/O"  # stored instead of a result when a run hits TIMEOUT_S
FAILED = "fail"  # stored when a run raised an exception


def _kill(p) -> None:
    """Kill the child and everything it started."""
    import signal

    if p.is_alive():
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (AttributeError, OSError, ProcessLookupError):
            p.kill()
    p.join(10)


def _wait(q, p, limit):
    """Next message from the child, or None if ``limit`` seconds pass / the child dies silently."""
    deadline = time.perf_counter() + limit
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return None
        try:
            return q.get(timeout=min(1.0, remaining))
        except queue.Empty:
            if not p.is_alive():  # crashed (e.g. segfault / OOM kill) without reporting
                try:
                    return q.get(timeout=1.0)
                except queue.Empty:
                    return ("error", f"child process died (exit code {p.exitcode})", 0.0)


def run_with_timeout(fn, *args, timeout: float | None = None, **kwargs):
    """Run ``fn(*args, **kwargs)`` in a fresh process and kill it if it runs longer than ``timeout``
    seconds (default ``TIMEOUT_S``).

    Returns ``(result, runtime_s, status)`` with status ``"ok"``, ``TIMEOUT`` or ``FAILED``
    (``result`` is ``None`` unless ok; a failure's traceback is printed). The clock starts once the
    child has finished starting up (spawn, imports, argument unpickling), so the limit and the
    reported runtime cover only the call itself; a timed-out run reports exactly ``timeout``.
    The child runs in its own process group, which is killed as a whole on timeout.

    The child is spawned, not forked: once the parent has run Qiskit's SABRE, its Rust thread pool
    deadlocks a forked child in SabreLayout, which OLSQ2 (for its SWAP upper bound) and SEQC both
    call. ``fn`` must therefore be a module-level function and the arguments picklable.
    """
    timeout = TIMEOUT_S if timeout is None else timeout
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(q, fn, args, kwargs), daemon=True)
    p.start()
    name = getattr(fn, "__name__", str(fn))
    try:
        msg = _wait(q, p, STARTUP_TIMEOUT_S)
        if msg is None:
            print(f"[{name}] failed: process did not start within {STARTUP_TIMEOUT_S:.0f} s")
            return None, 0.0, FAILED
        if msg[0] == "error":  # died before (or while) reporting start-up
            print(f"[{name}] failed:\n{msg[1]}")
            return None, msg[2], FAILED
        msg = _wait(q, p, timeout)
        if msg is None:
            print(f"[{name}] killed after {timeout:.0f} s (timeout)")
            return None, timeout, TIMEOUT
        status, payload, runtime = msg
        if status == "error":
            print(f"[{name}] failed:\n{payload}")
            return None, runtime, FAILED
        return payload, runtime, "ok"
    finally:
        _kill(p)


# --------------------------------------------------------------------------------------
# Storage and plot styling
# --------------------------------------------------------------------------------------

# key, legend label, face colour, hatch. The first three match the existing figures.
METHOD_STYLES = [
    ("sabre", "LightSABRE", "lightcoral", "o"),
    ("mech", "MECH", "#A7D9ED", "//"),
    ("qeccsynth", "QECC-Synth", "#B2D8B2", "/"),
    ("olsq2", "OLSQ2", "#F3C98B", "\\\\"),
    ("seqc", "SEQC", "#C9B7E3", ".."),
    ("murali", "Murali et al.", "#D5D5D5", "--"),
]
ALL_METHODS = [m[0] for m in METHOD_STYLES]


def load_results(output_dir: Path, pattern: str, methods=ALL_METHODS) -> dict:
    """Load ``pattern.format(method)`` for every method that has a result file."""
    results = {}
    for m in methods:
        path = Path(output_dir) / pattern.format(m)
        if path.exists():
            with open(path, "rb") as f:
                results[m] = pickle.load(f)
        else:
            print(f"No results for {m} ({path}); leaving it out of the plot.")
    return results


def save_results(output_dir: Path, pattern: str, results: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for m, storage in results.items():
        with open(output_dir / pattern.format(m), "wb") as f:
            pickle.dump(storage, f)


def style_timeout_bar(bar, edge_color: str) -> None:
    """Hollow, dashed, cross-hatched bar marking a timeout (as for QECC-Synth before)."""
    bar.set_facecolor("white")
    bar.set_linestyle("--")
    bar.set_hatch("xxx")
    bar.set_linewidth(2)
    bar.set_edgecolor(edge_color)