"""Helpers for evaluating Chipmunq's mapping and routing stages independently.

Chipmunq's mapping is split into stages that can each be swapped out:

    patch IR ─► partitioning ─► patch contraction ─► sequencing ─► global mapping ─► defect-aware
                (patches)       (patch graph)        (BFS order)    (patch→chiplet)   placement

followed by (noise-aware) routing. Every function here takes or returns a *mapping*
``{virtual qubit index -> physical qubit}`` so that any mapping can be combined with any
router on the *same* backend object. That is what makes the comparisons controlled:
same circuit, same backend (defects, link noise), only one stage changes.

Mappings
    chipmunq_mapping            full Chipmunq mapping (optionally with an alternative mapper)
    DefectBlindMapper           placement ignores defects      (-> repair_defects)
    RandomSequenceMapper        random patch order instead of the dependency BFS
    permute_chiplets            random patch→chiplet assignment, intra-chiplet placement kept
    intra_chiplet_sabre_mapping patch contraction removed: qubit-level SABRE inside each chiplet
    kahypar_grid_partitions     patch IR removed: KaHyPar blocks placed as generic grid blocks
    sabre_mapping               global qubit-level SabreLayout (no Chipmunq mapping at all)

Routers (all from a fixed mapping)
    route_from_mapping(..., router="basic" | "focus" | "tradeoff" | "sabre")
"""

from __future__ import annotations

import contextlib
import copy
import io
import math
import random
import time
from collections import deque

import numpy as np
import stim
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, Layout, PassManager
from qiskit.transpiler.passes import ApplyLayout, SabreLayout, SabreSwap, SetLayout, Unroll3qOrMore
from qiskit.transpiler.passes.layout.enlarge_with_ancilla import EnlargeWithAncilla
from qiskit.transpiler.passes.layout.full_ancilla_allocation import FullAncillaAllocation

from glue.qiskit_qec.stim_tools import get_stim_circuits_with_detectors
from qeccm.backends.BackendChipletV2 import BackendChipletV2
from qeccm.circuit.hypergraph_circuit import HypergraphCircuit
from qeccm.src.mapper import TrivialMapper
from qeccm.src.partitioners import KaHyParPartitioning
from qeccm.src.router import CostRouter

ANNOTATIONS = {"DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS", "barrier"}

# --------------------------------------------------------------------------------------
# Router configurations. Same values as run_noise_aware_routing (ra = 3):
#   Basic    alpha = 0,                beta = 0
#   Focus    alpha = 3 * ra / p_inter,  beta = 1   (run name "cost_inter")
#   Tradeoff alpha = ra / p_inter,      beta = 1   (run name "cost_tradeoff")
# --------------------------------------------------------------------------------------
ROUTING_ALPHA = 3


def router_params(router: str, ps_inter: float, ra: float = ROUTING_ALPHA) -> tuple[float, float]:
    if router == "basic":
        return 0.0, 0.0
    if router == "focus":
        return 3 * ra / ps_inter, 1.0
    if router == "tradeoff":
        return ra / ps_inter, 1.0
    raise ValueError(router)


@contextlib.contextmanager
def quiet(enabled: bool = True):
    """Chipmunq's passes print a lot; silence them during timing."""
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# --------------------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------------------
def symmetric_remote_noise(backend: BackendChipletV2) -> dict:
    """Link-noise map with both orientations.

    ``backend.inter_chiplet_connections`` stores every link once, e.g. (11, 72), and the
    noise model checks ``tuple(pair) in remote``. A gate applied as (72, 11) is therefore
    treated as an *intra*-chiplet gate. Routers pick the orientation arbitrarily (Chipmunq
    along its path direction, SABRE per SWAP), so without symmetrising, the amount of link
    noise a circuit receives depends on the router in a way unrelated to its quality.
    """
    remote = dict(backend.inter_chiplet_connections)
    for (a, b), p in list(remote.items()):
        remote.setdefault((b, a), p)
    return remote


# --------------------------------------------------------------------------------------
# Backend helpers
# --------------------------------------------------------------------------------------
def valid_qubits(backend: BackendChipletV2) -> list[int]:
    """Physical qubits with at least one working coupler (defective qubits excluded)."""
    g = backend.defective_coupling_map.graph
    return [q for q in range(backend.defective_coupling_map.size()) if g.in_degree(q) + g.out_degree(q) > 0]


def reduced_coupling_map(backend: BackendChipletV2, qubits: list[int] | None = None):
    """Defect-free coupling map restricted to ``qubits``; returns (cmap, local->global)."""
    qubits = valid_qubits(backend) if qubits is None else qubits
    return backend.defective_coupling_map.reduce(qubits, check_if_connected=False), list(qubits)


def local_positions(backend: BackendChipletV2) -> dict[int, tuple[int, int]]:
    """physical qubit -> (chiplet, index within chiplet)."""
    pos = {}
    for c in range(backend.get_num_chips()):
        for i, p in enumerate(backend.get_chiplet_at(c)):
            pos[int(p)] = (c, i)
    return pos


# --------------------------------------------------------------------------------------
# Mappings
# --------------------------------------------------------------------------------------
def _layout_to_dict(circuit: QuantumCircuit, layout: Layout) -> dict[int, int]:
    idx = {q: i for i, q in enumerate(circuit.qubits)}
    return {idx[v]: int(p) for v, p in layout.get_virtual_bits().items() if v in idx}


def chipmunq_mapping(circuit: QuantumCircuit, backend: BackendChipletV2, partitions: list,
                     patch_initialization: str = "", mapper_cls=TrivialMapper, mapper_kwargs=None,
                     verbose: bool = False) -> tuple[dict[int, int], float]:
    """Run Chipmunq's init + layout stages only. Returns (mapping, seconds)."""
    mapper_kwargs = mapper_kwargs or {}
    pm = PassManager([
        Unroll3qOrMore(),
        HypergraphCircuit(),
        KaHyParPartitioning(backend, partitions),
        mapper_cls(backend, patch_initialization=patch_initialization, **mapper_kwargs),
    ])
    t0 = time.perf_counter()
    with quiet(not verbose):
        pm.run(circuit)
    elapsed = time.perf_counter() - t0
    return _layout_to_dict(circuit, pm.property_set["layout"]), elapsed


class DefectBlindMapper(TrivialMapper):
    """Chipmunq mapper that does not see the defects (no no-placement zones)."""

    def run(self, dag):
        saved = self.backend.chiplet_to_defective_qubits
        self.backend.chiplet_to_defective_qubits = {c: [] for c in saved}
        try:
            return super().run(dag)
        finally:
            self.backend.chiplet_to_defective_qubits = saved


class _ChainGraph:
    """Stand-in for the contracted patch graph whose BFS visits ``order`` exactly."""

    def __init__(self, order):
        self._order = list(order)
        self._next = {a: [b] for a, b in zip(self._order, self._order[1:])}

    def nodes(self):
        return list(self._order)

    def neighbors(self, n):
        return iter(self._next.get(n, []))


class RandomSequenceMapper(TrivialMapper):
    """Chipmunq mapper with the dependency-BFS sequencing replaced by a random order.

    Global mapping and placement are unchanged: each patch is still placed relative to its
    predecessor in the sequence, but that predecessor is random instead of a neighbour.
    """

    def __init__(self, backend, patch_initialization: str = "", seed: int = 0):
        super().__init__(backend, patch_initialization=patch_initialization)
        self._rng = random.Random(seed)

    def assign_partition_to_qpu(self, partitioned_hg, dag):
        order = list(partitioned_hg._collapsed_phg.nodes())
        self._rng.shuffle(order)
        proxy = copy.copy(partitioned_hg)
        proxy._collapsed_phg = _ChainGraph(order)
        return super().assign_partition_to_qpu(proxy, dag)


def repair_defects(mapping: dict[int, int], backend: BackendChipletV2) -> tuple[dict[int, int], int]:
    """Move every virtual qubit sitting on a defective qubit to the nearest free working
    qubit (hop distance on the defect-free lattice, same chiplet preferred).
    Returns (mapping, number of moved qubits)."""
    defective = set(int(q) for q in backend.all_defective_qubits)
    ok = set(valid_qubits(backend))
    used = set(mapping.values())
    full = backend.coupling_map
    moved = 0
    out = dict(mapping)
    for v in sorted(out):
        p = out[v]
        if p not in defective and p in ok:
            continue
        # BFS on the full lattice (defective qubits still conduct the search, not qubits)
        seen, dq, found = {p}, deque([(p, 0)]), []
        best_d = None
        while dq:
            q, d = dq.popleft()
            if best_d is not None and d > best_d:
                break
            if q in ok and q not in used:
                found.append(q)
                best_d = d
                continue
            for nb in full.neighbors(q):
                if nb not in seen:
                    seen.add(nb)
                    dq.append((nb, d + 1))
        if not found:
            raise RuntimeError("repair_defects: no free working qubit left")
        same = [q for q in found if backend.get_chiplet_of_node(q) == backend.get_chiplet_of_node(p)]
        q = min(same or found)
        used.discard(p)
        used.add(q)
        out[v] = q
        moved += 1
    return out, moved


def permute_chiplets(mapping: dict[int, int], backend: BackendChipletV2, seed: int) -> dict[int, int]:
    """Random patch→chiplet assignment: move whole chiplet contents to a random chiplet,
    keeping each qubit's position inside its chiplet (all chiplets have the same shape)."""
    rng = np.random.default_rng(seed)
    nc = backend.get_num_chips()
    perm = rng.permutation(nc)
    pos = local_positions(backend)
    return {v: int(backend.get_chiplet_at(int(perm[pos[p][0]]))[pos[p][1]]) for v, p in mapping.items()}


def _two_qubit_pairs(circuit: QuantumCircuit):
    idx = {q: i for i, q in enumerate(circuit.qubits)}
    for ins in circuit.data:
        if ins.operation.num_qubits == 2 and ins.operation.name not in ANNOTATIONS:
            yield idx[ins.qubits[0]], idx[ins.qubits[1]]


def intra_chiplet_sabre_mapping(mapping: dict[int, int], circuit: QuantumCircuit, backend: BackendChipletV2,
                                seed: int = 0) -> dict[int, int]:
    """Keep Chipmunq's patch→chiplet assignment but place the qubits of each chiplet with
    qubit-level SabreLayout on that chiplet's working qubits (no patch geometry)."""
    by_chip: dict[int, list[int]] = {}
    for v, p in mapping.items():
        by_chip.setdefault(backend.get_chiplet_of_node(p), []).append(v)
    pairs = list(_two_qubit_pairs(circuit))
    ok = set(valid_qubits(backend))
    out = {}
    for c, vqs in by_chip.items():
        phys = [int(p) for p in backend.get_chiplet_at(c) if int(p) in ok]
        cmap, local_to_global = reduced_coupling_map(backend, phys)
        # Defects can split a chiplet: place on its largest connected working component
        comp_global = _component_globals(cmap, local_to_global, len(vqs))
        comp_cmap, comp_l2g = reduced_coupling_map(backend, comp_global)
        vidx = {v: i for i, v in enumerate(vqs)}
        sub = QuantumCircuit(len(vqs))
        for a, b in pairs:
            if a in vidx and b in vidx:
                sub.cx(vidx[a], vidx[b])
        if sub.size() == 0:
            chosen = list(range(len(vqs)))
        else:
            pm = PassManager([SabreLayout(comp_cmap, seed=seed, max_iterations=4, skip_routing=True)])
            pm.run(sub)
            lay = pm.property_set["layout"]
            chosen = [lay[sub.qubits[i]] for i in range(len(vqs))]
        for v, l in zip(vqs, chosen):
            out[v] = comp_l2g[l]
    return out


def _component_globals(cmap: CouplingMap, local_to_global: list[int], need: int) -> list[int]:
    """Global qubit ids of the largest connected component of ``cmap``."""
    import rustworkx as rx
    g = cmap.graph.to_undirected()
    comps = sorted(rx.connected_components(g), key=len, reverse=True)
    if len(comps[0]) < need:
        raise RuntimeError("largest working component too small")
    return sorted(local_to_global[i] for i in comps[0])


def sabre_mapping(circuit: QuantumCircuit, backend: BackendChipletV2, seed: int = 0) -> tuple[dict[int, int], float]:
    """Global qubit-level SabreLayout on the working qubits (LightSABRE's layout)."""
    cmap, l2g = reduced_coupling_map(backend)
    pm = PassManager([Unroll3qOrMore(), SabreLayout(cmap, seed=seed, max_iterations=4, skip_routing=True)])
    t0 = time.perf_counter()
    pm.run(circuit)
    elapsed = time.perf_counter() - t0
    lay = pm.property_set["layout"]
    idx = {q: i for i, q in enumerate(circuit.qubits)}
    return {idx[v]: l2g[p] for v, p in lay.get_virtual_bits().items() if v in idx}, elapsed


def kahypar_grid_partitions(circuit: QuantumCircuit, partitions: list, seed: int = 42) -> list:
    """Replace the patch IR by KaHyPar blocks of the qubit interaction graph.

    k and the block sizes are taken from the patch IR (so the comparison changes *which*
    qubits are grouped, not how many groups there are). Each block becomes a generic
    ``grid`` partition with its patch's bounding box (enlarged if the block is bigger),
    so contraction, sequencing, global mapping and defect-aware placement still run.
    """
    import kahypar

    n = circuit.num_qubits
    weights: dict[tuple[int, int], int] = {}
    for a, b in _two_qubit_pairs(circuit):
        key = (min(a, b), max(a, b))
        weights[key] = weights.get(key, 0) + 1
    edges = sorted(weights)
    index_vector, edge_vector = [0], []
    for a, b in edges:
        edge_vector += [a, b]
        index_vector.append(len(edge_vector))
    k = len(partitions)
    ctx = kahypar.Context()
    ctx.loadINIconfiguration("qeccm/src/kahypar_config.ini")
    ctx.setK(k)
    sizes = [len(p["indices"]) for p in partitions]
    slack = max(1, int(0.03 * n))
    ctx.setCustomTargetBlockWeights([s + slack for s in sizes])
    ctx.suppressOutput(True)
    ctx.setSeed(seed)
    hg = kahypar.Hypergraph(n, len(edges), index_vector, edge_vector, k, [weights[e] for e in edges], [1] * n)
    kahypar.partition(hg, ctx)
    blocks = [[] for _ in range(k)]
    for v in range(n):
        blocks[hg.blockID(v)].append(v)
    # Pair blocks with patch boxes by size (largest block -> largest patch)
    order_b = sorted(range(k), key=lambda i: -len(blocks[i]))
    order_p = sorted(range(k), key=lambda i: -sizes[i])
    out = [None] * k
    for bi, pi in zip(order_b, order_p):
        w, h = partitions[pi]["width"], partitions[pi]["height"]
        while w * h < len(blocks[bi]):
            h += 1
        out[pi] = {"indices": sorted(blocks[bi]), "width": w, "height": h, "type": "grid"}
    return out


# --------------------------------------------------------------------------------------
# Routing from a fixed mapping
# --------------------------------------------------------------------------------------
def _placement_passes(circuit: QuantumCircuit, backend: BackendChipletV2, mapping: dict[int, int]):
    lay = Layout({circuit.qubits[v]: p for v, p in mapping.items()})
    return [Unroll3qOrMore(), SetLayout(lay), FullAncillaAllocation(backend.coupling_map), EnlargeWithAncilla(),
            ApplyLayout()]


_SABRE_WARM = False


def _warm_up_sabre() -> None:
    """The first SabreSwap call of a process pays a one-time start-up cost (~50-100 ms);
    run a tiny instance first so it is not charged to the timed budget."""
    global _SABRE_WARM
    if not _SABRE_WARM:
        qc = QuantumCircuit(4)
        qc.cx(0, 3)
        PassManager([SabreSwap(CouplingMap.from_line(4), seed=0, trials=1)]).run(qc)
        _SABRE_WARM = True


def murali_noise(backend: BackendChipletV2) -> tuple[CouplingMap, dict]:
    """Working couplers and their 2q errors as Murali et al.'s router needs them: intra-chiplet
    errors from the (defective) target, inter-chiplet links from the per-link noise that the
    simulation applies. Without this the router would be "noise-aware" of the wrong noise."""
    cmap = backend.defective_coupling_map
    remote = symmetric_remote_noise(backend)
    target = backend._defective_target
    errs = {}
    for a, b in cmap.get_edges():
        e = (min(a, b), max(a, b))
        if e in errs:
            continue
        if (a, b) in remote:
            errs[e] = remote[(a, b)]
            continue
        best = None
        for name in ("cz", "cx", "ecr"):
            if name in target.operation_names:
                for q in ((a, b), (b, a)):
                    props = target[name].get(q)
                    if props is not None and props.error is not None:
                        best = props.error if best is None else min(best, props.error)
        if best is not None:
            errs[e] = best
    return cmap, errs


def seqc_target(backend: BackendChipletV2, circuit: QuantumCircuit):
    """SEQC's device model on this backend: the circuit's own gates are native (so SEQC's basis
    translation is an identity and the output stays convertible to Stim), 2q gates only on
    working intra-chiplet couplers with their calibrated error, and inter-chiplet links that
    support only SWAP with the per-link noise the simulation applies. This is what makes
    SEQC's link selection and intra-chiplet routing noise-aware. Links are told apart from
    intra-chiplet couplers through the explicit chiplet partition, not the gate set.""" 
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import SwapGate, get_standard_gate_name_mapping
    from qiskit.transpiler import InstructionProperties, Target

    cmap, errs = murali_noise(backend)  # working couplers + errors (links: per-link noise)
    n = backend.num_qubits_total
    remote = symmetric_remote_noise(backend)
    std = get_standard_gate_name_mapping()
    t = Target(num_qubits=n)
    names = {i.operation.name for i in circuit.data} - ANNOTATIONS - {"swap"}
    names |= {"measure", "reset"}
    intra = [(a, b) for a, b in cmap.get_edges() if (a, b) not in remote]
    links = [(a, b) for a, b in cmap.get_edges() if (a, b) in remote]
    for name in sorted(names):
        op = std[name]
        if op.params:
            op = op.__class__(*[Parameter(f"{name}_{i}") for i in range(len(op.params))])
        if op.num_qubits == 1:
            t.add_instruction(op, {(q,): None for q in range(n)})
        else:
            t.add_instruction(op, {e: InstructionProperties(error=errs[(min(e), max(e))]) for e in intra})
    # SWAP is native everywhere, as for the other routers (the noise model charges a SWAP as one
    # 2q error). Otherwise SEQC's intra-chiplet SWAPs would become 3 CX and pay 3x the noise.
    t.add_instruction(SwapGate(), {**{e: InstructionProperties(error=errs[(min(e), max(e))]) for e in intra},
                                   **{e: InstructionProperties(error=remote[e]) for e in links}})
    return t


def split_annotations(circuit: QuantumCircuit):
    """Remove Stim annotations and express every DETECTOR / OBSERVABLE_INCLUDE by the clbits of
    the measurements it refers to (its rec[-k] targets are relative to the measurement order,
    which SEQC changes). Returns (circuit without annotations, [(name, args, [clbit index])])."""
    core = circuit.copy_empty_like()
    meas_clbits, anns = [], []
    for ins in circuit.data:
        name = ins.operation.name
        if name == "measure":
            meas_clbits.append(circuit.find_bit(ins.clbits[0]).index)
        if name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            prm = ins.operation.params[0]
            anns.append((name, list(prm["coords"]), [meas_clbits[len(meas_clbits) + r] for r in prm["rec_indices"]]))
        elif name not in ANNOTATIONS:
            core.append(ins)
    if len(set(meas_clbits)) != len(meas_clbits):
        raise ValueError("re-attaching detectors needs one clbit per measurement")
    return core, anns


def stim_with_annotations(routed: QuantumCircuit, anns) -> stim.Circuit:
    """Convert a routed circuit without annotations to Stim and re-attach the detectors and
    observables at the end, pointing at the measurements (by clbit) they referred to."""
    base = get_stim_circuits_with_detectors(routed)[0][0]
    # The Stim conversion may reorder operations on different qubits, never on the same one:
    # identify each Stim measurement as the k-th measurement of its qubit.
    per_qubit: dict[int, list[int]] = {}
    for ins in routed.data:
        if ins.operation.name == "measure":
            per_qubit.setdefault(routed.find_bit(ins.qubits[0]).index, []).append(routed.find_bit(ins.clbits[0]).index)
    seen: dict[int, int] = {}
    record = []  # stim measurement record index -> clbit
    for inst in base.flattened():
        if inst.name in ("M", "MR", "MX", "MY", "MZ"):
            for tgt in inst.targets_copy():
                q = tgt.value
                record.append(per_qubit[q][seen.get(q, 0)])
                seen[q] = seen.get(q, 0) + 1
    pos = {c: i for i, c in enumerate(record)}
    total = len(record)
    out = base.copy()
    for name, args, clbits in anns:
        out.append(name, [stim.target_rec(pos[c] - total) for c in clbits], args)
    return out


def seqc_route(circuit: QuantumCircuit, backend: BackendChipletV2, mapping: dict[int, int], seed: int = 0):
    """SEQC routing from ``mapping``. Returns (routed circuit without annotations, Stim circuit)."""
    from external.baseline.seqc.seqc import SEQCCompiler

    core, anns = split_annotations(circuit)
    target = seqc_target(backend, core)
    chiplets = [[int(q) for q in backend.get_chiplet_at(c)] for c in range(backend.get_num_chips())]
    comp = SEQCCompiler(target, chiplets=chiplets, optimization_level=0, seed=seed, n_jobs=1)
    with quiet():
        out = comp.run(core, initial_layout=[mapping[i] for i in range(circuit.num_qubits)])
    return out, stim_with_annotations(out, anns)


def _n_swaps(qc: QuantumCircuit) -> int:
    return qc.count_ops().get("swap", 0)


def route_from_mapping(circuit: QuantumCircuit, backend: BackendChipletV2, mapping: dict[int, int], router: str,
                       ps_inter: float, *, ra: float = ROUTING_ALPHA, seed: int = 0, budget_s: float | None = None,
                       sabre_trials: int | None = None, timing_repeats: int = 3,
                       alpha: float | None = None, beta: float | None = None,
                       verbose: bool = False) -> tuple[QuantumCircuit, dict]:
    """Route ``circuit`` from ``mapping`` on ``backend``.

    router: "basic" | "focus" | "tradeoff" -> Chipmunq's CostRouter with the Fig. 9 params
                                              (``alpha``/``beta`` override them if given)
            "sabre"                        -> LightSABRE SabreSwap on the defect-free coupling map
            "murali"                       -> Murali et al.'s noise-adaptive routing (most reliable
                                              path, swap-and-return) from the same placement
            "seqc"                         -> SEQC's inter-chiplet routing (Alg. 2), noise-weighted link
                                              assignment and intra-chiplet routing, from the same
                                              placement. info["stim"] holds the Stim circuit with the
                                              detectors re-attached (see ``seqc_route``).

    SABRE with ``budget_s``: independent single-trial SabreSwap restarts (fresh seeds),
    keeping the result with the fewest SWAPs (SABRE's own trial criterion, so SABRE is not
    given noise information). A restart is only started if, at the mean restart time so far,
    it is expected to finish within the budget; at least one restart always runs.
    Without a budget, one SabreSwap call with ``sabre_trials`` trials (Qiskit default: #CPUs).

    Chipmunq routers are deterministic; they are run ``timing_repeats`` times and the median
    time is reported, so a single slow run cannot inflate the SABRE budget.

    Returns (routed circuit, info) with info["routing_s"] including the layout application.
    """
    info = {"router": router}
    if router in ("basic", "focus", "tradeoff"):
        a0, b0 = router_params(router, ps_inter, ra)
        alpha = a0 if alpha is None else alpha
        beta = b0 if beta is None else beta
        times = []
        for _ in range(max(1, timing_repeats)):  # deterministic: repeat only to get a stable time
            pm = PassManager(_placement_passes(circuit, backend, mapping) + [CostRouter(backend, alpha=alpha, beta=beta)])
            t0 = time.perf_counter()
            with quiet(not verbose):
                out = pm.run(circuit)
            times.append(time.perf_counter() - t0)
        info.update(routing_s=float(np.median(times)), trials=1, alpha=alpha, beta=beta)
        return out, info
    if router == "murali":
        from external.baseline.noise_aware_mapping.murali import noise_adaptive_transpilation

        cmap, cx_errors = murali_noise(backend)
        layout = [mapping[i] for i in range(circuit.num_qubits)]
        t0 = time.perf_counter()
        out = noise_adaptive_transpilation(circuit, cmap, method="greedy_e", cx_errors=cx_errors,
                                           initial_layout=layout)
        info.update(routing_s=time.perf_counter() - t0, trials=1)
        return out, info
    if router == "seqc":
        t0 = time.perf_counter()
        out, stim_c = seqc_route(circuit, backend, mapping, seed=seed)
        info.update(routing_s=time.perf_counter() - t0, trials=1, stim=stim_c)
        return out, info
    if router != "sabre":
        raise ValueError(router)

    _warm_up_sabre()
    t0 = time.perf_counter()
    placed = PassManager(_placement_passes(circuit, backend, mapping)).run(circuit)
    t_place = time.perf_counter() - t0
    cmap = backend.defective_coupling_map
    best, trials = None, 0
    t0 = time.perf_counter()
    if budget_s is None:
        best = PassManager([SabreSwap(cmap, heuristic="decay", seed=seed, trials=sabre_trials)]).run(placed)
        trials = sabre_trials if sabre_trials else float("nan")  # nan = Qiskit's default trial count
    else:
        while True:
            cand = PassManager([SabreSwap(cmap, heuristic="decay", seed=seed + trials, trials=1)]).run(placed)
            trials += 1
            if best is None or _n_swaps(cand) < _n_swaps(best):
                best = cand
            spent = t_place + (time.perf_counter() - t0)
            mean_restart = (spent - t_place) / trials
            if spent + mean_restart > budget_s:  # the next restart would not fit
                break
    info.update(routing_s=t_place + time.perf_counter() - t0, trials=trials, budget_s=budget_s)
    return best, info


def lightsabre(circuit: QuantumCircuit, backend: BackendChipletV2, seed: int = 0) -> tuple[QuantumCircuit, dict]:
    """Stock LightSABRE (SabreLayout + SabreSwap) on the working qubits."""
    mapping, t_map = sabre_mapping(circuit, backend, seed=seed)
    out, info = route_from_mapping(circuit, backend, mapping, "sabre", 0.0, seed=seed)
    info["mapping_s"] = t_map
    return out, info


# --------------------------------------------------------------------------------------
# Verification and metrics
# --------------------------------------------------------------------------------------
def to_stim(routed: QuantumCircuit) -> stim.Circuit:
    return get_stim_circuits_with_detectors(routed)[0][0]


def check_routed(routed: QuantumCircuit, reference: stim.Circuit, backend: BackendChipletV2, shots: int = 256,
                 stim_circuit: stim.Circuit | None = None) -> stim.Circuit:
    """Convert to Stim and check the routed circuit is still a valid QEC experiment:
    same detectors/observables, all deterministic without noise, every 2q gate on a
    working coupler. Raises on failure; returns the Stim circuit. Pass ``stim_circuit`` if the
    router already produced it (SEQC)."""
    stim_c = to_stim(routed) if stim_circuit is None else stim_circuit
    if (stim_c.num_detectors, stim_c.num_observables) != (reference.num_detectors, reference.num_observables):
        raise RuntimeError("detector/observable count changed by routing")
    det, obs = stim_c.compile_detector_sampler(seed=0).sample(shots, separate_observables=True)
    if det.any() or obs.any():
        raise RuntimeError("routing produced non-deterministic detectors/observables")
    cmap = backend.defective_coupling_map
    edges = set(cmap.get_edges())
    for ins in routed.data:
        if ins.operation.num_qubits == 2 and ins.operation.name not in ANNOTATIONS:
            a, b = (routed.find_bit(q).index for q in ins.qubits)
            if (a, b) not in edges and (b, a) not in edges:
                raise RuntimeError(f"2q gate on missing/defective coupler ({a},{b})")
    return stim_c


def overhead_stats(routed: QuantumCircuit, original: QuantumCircuit, backend: BackendChipletV2) -> dict:
    """Gate/depth overhead relative to the unrouted circuit (SWAP = 3 two-qubit gates)."""
    def gate_filter(ins):
        return ins.operation.name not in ANNOTATIONS

    def two_q_filter(ins):
        return ins.operation.num_qubits == 2 and ins.operation.name not in ANNOTATIONS

    def n2q(qc):
        return sum(1 for ins in qc.data if two_q_filter(ins))

    remote = set(backend.inter_chiplet_connections)
    swaps_intra = swaps_inter = gates_inter = 0
    for ins in routed.data:
        if not two_q_filter(ins):
            continue
        a, b = (routed.find_bit(q).index for q in ins.qubits)
        inter = (a, b) in remote or (b, a) in remote
        if ins.operation.name == "swap":
            swaps_inter += inter
            swaps_intra += not inter
        else:
            gates_inter += inter
    decomposed = routed.decompose("swap")
    return {
        "swaps": swaps_intra + swaps_inter,
        "swaps_intra": swaps_intra,
        "swaps_inter": swaps_inter,
        "inter_chiplet_2q": 3 * swaps_inter + gates_inter,
        "2q_overhead": (n2q(routed) + 2 * (swaps_intra + swaps_inter)) - n2q(original),
        "depth_overhead": routed.depth(gate_filter) - original.depth(gate_filter),
        "2q_depth_overhead": decomposed.depth(two_q_filter) - original.depth(two_q_filter),
    }