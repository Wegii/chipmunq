"""
seqc.py -- A Qiskit implementation of SEQC, the Stratify-Elaborate Quantum Compiler.

    M. J. Jeng, N. V. Maruszewski, C. Selna, M. Gavrincea, K. N. Smith, N. Hardavellas,
    "Modular Compilation for Quantum Chiplet Architectures", arXiv:2501.08478 (v3, 2025).

Pipeline (Fig. 2 of the paper)
------------------------------
Stratification (topology only, run once per circuit/architecture, cacheable):
    1. Qubit-to-subcircuit mapping       -- simulated annealing (Algorithm 1)
    2. Subcircuit-to-chiplet mapping     -- seeded from (1), refined by a small QAP anneal
                                            and randomized permutation trials, each scored by (3)
    3. Inter-chiplet routing             -- chiplet-level SABRE variant with
                                            symbiotic / commensalistic / parasitic SWAP tiers (Alg. 2)
Elaboration (recurring, uses up-to-date calibration, parallel per chiplet):
    4. Intra-chiplet layout              -- SabreLayout per chiplet            [parallel]
    5. Inter-chiplet SWAP lowering       -- greedy nearest-link assignment     [serial]
    6. Intra-chiplet routing             -- SabreSwap with pinned link ports   [parallel]
    7. Basis translation + optimization  -- per chiplet, disjoint gate sets    [parallel]
    8. Stitching                         -- merge chiplet circuits + inter-chiplet SWAPs

Also included:
    * PeepholeCorrection  -- the Sec. 3.1 pass that makes stock Qiskit chiplet-aware (PA-Qiskit)
    * make_chiplet_backend -- the mock heavy-hex-chiplet grid backend of Sec. 4.2 / Table 1
    * circuit_metrics      -- inter-chiplet gate count, depth, ESP, estimated duration

Usage
-----
    from seqc import SEQCCompiler, make_chiplet_backend
    backend = make_chiplet_backend(2, 2)          # 4 chiplets x 10 qubits
    compiler = SEQCCompiler(backend)
    exe = compiler.run(circuit)                   # executable QuantumCircuit

    # or, reusing stratification across calibration cycles:
    strat = compiler.stratify(circuit)
    exe = compiler.elaborate(strat)               # re-run only this before each execution
"""
from __future__ import annotations

import math
import os
import random
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import rustworkx as rx

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import AncillaRegister, Barrier, ControlFlowOp, Gate, Parameter
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as _SEL
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.circuit.library import CZGate, Measure, Reset, RZGate, SwapGate, SXGate, XGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import (
    CouplingMap,
    InstructionProperties,
    Layout,
    PassManager,
    QubitProperties,
    Target,
    TranspileLayout,
)
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import (
    BasisTranslator,
    CommutativeCancellation,
    ConsolidateBlocks,
    Decompose,
    GateDirection,
    HighLevelSynthesis,
    Optimize1qGatesDecomposition,
    RemoveBarriers,
    SabreLayout,
    Unroll3qOrMore,
    UnitarySynthesis,
)

_PORT_LABEL = "__seqc_xswap_"

__all__ = [
    "ChipletArchitecture",
    "StratifiedCircuit",
    "SEQCCompiler",
    "seqc_transpile",
    "PeepholeCorrection",
    "pa_qiskit_transpile",
    "make_chiplet_backend",
    "circuit_metrics",
]


# =====================================================================================
# Chiplet architecture model
# =====================================================================================
def _as_target(backend_or_target) -> Target:
    if isinstance(backend_or_target, Target):
        return backend_or_target
    return backend_or_target.target


def _two_qubit_gate_names(target: Target, a: int, b: int) -> set[str]:
    names = set()
    for name in target.operation_names:
        if name in ("barrier", "delay"):
            continue
        op = target.operation_from_name(name)
        nq = getattr(op, "num_qubits", None)
        if isinstance(op, type) or nq == 2:  # variadic ops (class) or real 2q ops
            if target.instruction_supported(name, (a, b)) or target.instruction_supported(name, (b, a)):
                names.add(name)
    return names


@dataclass
class ChipletArchitecture:
    """Static (topology-only) description of a modular device."""

    num_qubits: int
    chiplets: list[list[int]]                      # physical qubits in each chiplet
    chip_of: list[int]                             # physical qubit -> chiplet index
    local_index: list[int]                         # physical qubit -> index inside its chiplet
    intra_edges: list[list[tuple[int, int]]]       # per-chiplet undirected edges (local indices)
    links: dict[tuple[int, int], list[tuple[int, int, float]]]  # (A,B), A<B -> [(a_phys, b_phys, err)]
    chip_adj: list[list[int]]                      # chiplet adjacency
    chip_dist: np.ndarray                          # all-pairs chiplet hop distance
    local_dist: list[np.ndarray]                   # per-chiplet hop-distance matrices

    @property
    def num_chiplets(self) -> int:
        return len(self.chiplets)

    def is_inter_chip(self, a: int, b: int) -> bool:
        return self.chip_of[a] != self.chip_of[b]

    @classmethod
    def from_target(
        cls,
        backend_or_target,
        chiplets: Sequence[Iterable[int]] | None = None,
        inter_link_gates: Iterable[str] = ("swap",),
    ) -> "ChipletArchitecture":
        """Build the chiplet model.

        If ``chiplets`` is None, inter-chiplet links are detected as edges whose only
        2q instructions are in ``inter_link_gates`` (the paper's non-universal links), and
        chiplets are the connected components of the remaining (intra-chiplet) graph.
        """
        target = _as_target(backend_or_target)
        n = target.num_qubits
        cmap = target.build_coupling_map()
        if cmap is None:
            raise ValueError("Target has no coupling map (all-to-all); cannot infer chiplets.")
        edges = sorted({tuple(sorted(e)) for e in cmap.get_edges()})
        inter_link_gates = set(inter_link_gates)

        if chiplets is None:
            g = rx.PyGraph()
            g.add_nodes_from(range(n))
            for a, b in edges:
                names = _two_qubit_gate_names(target, a, b)
                if not (names and names <= inter_link_gates):
                    g.add_edge(a, b, None)
            comps = [sorted(c) for c in rx.connected_components(g)]
            chiplets = sorted(comps, key=lambda c: c[0])
            if len(chiplets) == 1:
                raise ValueError(
                    "Could not detect chiplets: no edges restricted to inter-chiplet gates "
                    f"{sorted(inter_link_gates)}. Pass `chiplets=` explicitly."
                )
        else:
            chiplets = [sorted(c) for c in chiplets]
            if sorted(q for c in chiplets for q in c) != list(range(n)):
                raise ValueError("`chiplets` must partition all physical qubits.")

        chip_of = [0] * n
        local_index = [0] * n
        for ci, qs in enumerate(chiplets):
            for li, q in enumerate(qs):
                chip_of[q], local_index[q] = ci, li

        nc = len(chiplets)
        intra_edges: list[list[tuple[int, int]]] = [[] for _ in range(nc)]
        links: dict[tuple[int, int], list[tuple[int, int, float]]] = {}
        for a, b in edges:
            ca, cb = chip_of[a], chip_of[b]
            if ca == cb:
                intra_edges[ca].append((local_index[a], local_index[b]))
            else:
                if ca > cb:
                    a, b, ca, cb = b, a, cb, ca
                err = _edge_error(target, a, b)
                links.setdefault((ca, cb), []).append((a, b, err))

        local_dist = []
        for ci, qs in enumerate(chiplets):
            if len(qs) == 1:
                local_dist.append(np.zeros((1, 1)))
                continue
            cm = CouplingMap(_bidir(intra_edges[ci]))
            if not cm.is_connected() or cm.size() != len(qs):
                raise ValueError(f"Chiplet {ci} is not internally connected.")
            local_dist.append(np.asarray(cm.distance_matrix))

        cg = rx.PyGraph()
        cg.add_nodes_from(range(nc))
        for (ca, cb) in links:
            cg.add_edge(ca, cb, None)
        chip_adj = [sorted(cg.neighbors(c)) for c in range(nc)]
        chip_dist = np.asarray(rx.distance_matrix(cg, null_value=np.inf))
        if nc > 1 and np.isinf(chip_dist).any():
            raise ValueError("Chiplet graph is disconnected.")
        return cls(n, chiplets, chip_of, local_index, intra_edges, links, chip_adj,
                   chip_dist.astype(int), local_dist)


def _edge_error(target: Target, a: int, b: int) -> float:
    best = None
    for name in _two_qubit_gate_names(target, a, b):
        for qargs in ((a, b), (b, a)):
            try:
                props = target[name].get(qargs)
            except KeyError:
                props = None
            if props is not None and props.error is not None:
                best = props.error if best is None else min(best, props.error)
    return 0.0 if best is None else best


def _bidir(edges):
    out = []
    for a, b in edges:
        out += [(a, b), (b, a)]
    return out


def _sub_target(target: Target, phys: Sequence[int]) -> Target:
    """Restrict a Target to one chiplet (re-indexed to local qubits). Inter-chiplet
    instructions are dropped, so the chiplet only ever sees its intra-chiplet gate set --
    the 'disjoint gate sets' guarantee of Sec. 3.2.2."""
    idx = {p: i for i, p in enumerate(phys)}
    qp = None
    if target.qubit_properties is not None:
        qp = [target.qubit_properties[p] for p in phys]
    t = Target(num_qubits=len(phys), dt=target.dt, qubit_properties=qp)
    for name in target.operation_names:
        op = target.operation_from_name(name)
        qmap = target[name]
        if None in qmap:  # global / ideal instruction
            t.add_instruction(op, name=name)
            continue
        props = {
            tuple(idx[q] for q in qargs): p
            for qargs, p in qmap.items()
            if all(q in idx for q in qargs)
        }
        if props:
            t.add_instruction(op, props, name=name)
    return t


# =====================================================================================
# Mock backend (Sec. 4.2, Table 1, Fig. 6)
# =====================================================================================
class ChipletBackend(BackendV2):
    """Minimal BackendV2 wrapper around a chiplet Target (compile-only)."""

    def __init__(self, target: Target, name: str = "chiplet_backend"):
        super().__init__(name=name)
        self._target = target

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    @classmethod
    def _default_options(cls):
        return Options()

    def run(self, run_input, **options):
        raise NotImplementedError("This mock backend is for compilation only.")


# 10-qubit heavy-hex chiplet cell (Fig. 6). Rows 0-3 and 5-8, bridge 4 (col 0) joins the rows,
# qubit 9 hangs off 7 (col 2) and bridges to qubit 2 of the chiplet above. Tiling the cells on a
# grid reproduces a global heavy-hex lattice.
_CELL_EDGES = [(0, 1), (1, 2), (2, 3), (5, 6), (6, 7), (7, 8), (0, 4), (4, 5), (7, 9)]


def make_chiplet_backend(rows: int, cols: int, universal_links: bool = False,
                         name: str | None = None) -> ChipletBackend:
    """Grid of heavy-hex chiplets with the Table 1 specs.

    universal_links=False : inter-chiplet links support only SWAP (702.4 ns, 10.23% error).
    universal_links=True  : inter-chiplet links support CZ (126 ns, 2.42% error) -- the
                            chiplet-unaware Qiskit baseline setup of the paper.
    """
    nc = rows * cols
    n = 10 * nc
    g = lambda i, j, l: 10 * (i * cols + j) + l  # noqa: E731
    intra, inter = [], []
    for i in range(rows):
        for j in range(cols):
            intra += [(g(i, j, a), g(i, j, b)) for a, b in _CELL_EDGES]
            if j + 1 < cols:
                inter += [(g(i, j, 3), g(i, j + 1, 0)), (g(i, j, 8), g(i, j + 1, 5))]
            if i + 1 < rows:
                inter += [(g(i, j, 9), g(i + 1, j, 2))]

    t = Target(num_qubits=n, dt=None,
               qubit_properties=[QubitProperties(t1=20e-6, t2=30e-6, frequency=6e9)] * n)
    one = lambda d, e: {(q,): InstructionProperties(d, e) for q in range(n)}  # noqa: E731
    t.add_instruction(XGate(), one(25e-9, 1.09e-3))
    t.add_instruction(SXGate(), one(25e-9, 1.09e-3))
    t.add_instruction(RZGate(Parameter("theta")), one(0.0, 0.0))
    t.add_instruction(Measure(), one(500e-9, 1.96e-3))
    t.add_instruction(Reset(), one(500e-9, 1.86e-3))
    cz = {}
    for a, b in intra:
        cz[(a, b)] = cz[(b, a)] = InstructionProperties(34e-9, 6.05e-3)
    if universal_links:
        for a, b in inter:
            cz[(a, b)] = cz[(b, a)] = InstructionProperties(126e-9, 2.42e-2)
    t.add_instruction(CZGate(), cz)
    if not universal_links:
        sw = {}
        for a, b in inter:
            sw[(a, b)] = sw[(b, a)] = InstructionProperties(702.4e-9, 0.1023)
        t.add_instruction(SwapGate(), sw)
    return ChipletBackend(t, name or f"chiplet_{rows}x{cols}{'_universal' if universal_links else ''}")


# =====================================================================================
# Pre-processing
# =====================================================================================
def _preprocess(circuit: QuantumCircuit):
    """Flatten to <=2q gates, drop barriers, turn user SWAPs into CXs (so every SWAP seen
    later was inserted by a router), and return a flat op list over integer indices."""
    for inst in circuit.data:
        op = inst.operation
        if isinstance(op, ControlFlowOp) or getattr(op, "condition", None) is not None:
            raise NotImplementedError("SEQC: classical control flow is not supported.")
    std = list(get_standard_gate_name_mapping()) + ["unitary"]
    pm = PassManager([RemoveBarriers(), HighLevelSynthesis(basis_gates=std), Unroll3qOrMore(),
                      Decompose(gates_to_decompose=["swap"])])
    flat = pm.run(circuit)
    qi = {q: i for i, q in enumerate(flat.qubits)}
    ci = {c: i for i, c in enumerate(flat.clbits)}
    ops = []
    for inst in flat.data:
        if not inst.qubits:
            continue
        ops.append((inst.operation, tuple(qi[q] for q in inst.qubits), tuple(ci[c] for c in inst.clbits)))
    return ops, flat.global_phase


# =====================================================================================
# Stratification, step 1: qubit-to-subcircuit mapping (Algorithm 1)
# =====================================================================================
def _interaction_graph(ops, n):
    adj = [Counter() for _ in range(n)]
    for _, qs, _ in ops:
        if len(qs) == 2:
            a, b = qs
            adj[a][b] += 1
            adj[b][a] += 1
    return [list(a.items()) for a in adj]


def _greedy_initial_assignment(adj, capacities, chip_adj):
    """BFS over the interaction graph, filling chiplets in chiplet-BFS order."""
    n = len(adj)
    deg = [sum(w for _, w in a) for a in adj]
    seen, order = [False] * n, []
    for s in sorted(range(n), key=lambda q: -deg[q]):
        if seen[s]:
            continue
        seen[s] = True
        dq = deque([s])
        while dq:
            q = dq.popleft()
            order.append(q)
            for p, _ in sorted(adj[q], key=lambda x: -x[1]):
                if not seen[p]:
                    seen[p] = True
                    dq.append(p)
    # chip order: BFS from chip 0
    chip_order, cseen, dq = [], {0}, deque([0])
    while dq:
        c = dq.popleft()
        chip_order.append(c)
        for d in chip_adj[c]:
            if d not in cseen:
                cseen.add(d)
                dq.append(d)
    chip_order += [c for c in range(len(capacities)) if c not in cseen]
    assign = [0] * n
    it = iter(order)
    for c in chip_order:
        for _ in range(capacities[c]):
            assign[next(it)] = c
    return assign


def _swap_delta(adj, assign, q0, q1, dist):
    s0, s1 = assign[q0], assign[q1]
    d = 0
    for p, w in adj[q0]:
        if p != q1:
            cp = assign[p]
            d += w * ((dist[s1][cp] if dist is not None else (cp != s1))
                      - (dist[s0][cp] if dist is not None else (cp != s0)))
    for p, w in adj[q1]:
        if p != q0:
            cp = assign[p]
            d += w * ((dist[s0][cp] if dist is not None else (cp != s0))
                      - (dist[s1][cp] if dist is not None else (cp != s1)))
    return d


def _cost(adj, assign, dist):
    c = 0
    for q, nb in enumerate(adj):
        for p, w in nb:
            if p > q:
                c += w * (dist[assign[q]][assign[p]] if dist is not None else assign[q] != assign[p])
    return c


def _sa_worker(args):
    """One simulated-annealing trial of Algorithm 1 (swap-based kicks between chiplets)."""
    adj, init, T0, alpha, beta, refine_iters, dist, seed = args
    rng = random.Random(seed)
    n = len(init)
    L = list(init)
    cur = _cost(adj, L, dist)
    best, best_cost = list(L), cur
    if len(set(L)) < 2:
        return best, best_cost

    def random_cross_pair():
        q0 = rng.randrange(n)
        while True:
            q1 = rng.randrange(n)
            if L[q1] != L[q0]:
                return q0, q1

    T = T0
    while T > 1:
        D = max(1, math.ceil(T * alpha))
        applied, delta = [], 0
        for _ in range(D):                      # Kick(): D random cross-chiplet swaps
            q0, q1 = random_cross_pair()
            delta += _swap_delta(adj, L, q0, q1, dist)
            L[q0], L[q1] = L[q1], L[q0]
            applied.append((q0, q1))
        # Metropolis acceptance (the paper's Alg. 1, line 28, with the usual sign convention)
        if delta < 0 or rng.random() < math.exp(-delta / T):
            cur += delta
            if cur < best_cost:
                best, best_cost = list(L), cur
        else:
            for q0, q1 in reversed(applied):
                L[q0], L[q1] = L[q1], L[q0]
        T *= beta

    # zero-temperature tail (greedy descent from the best solution)
    L, cur = best, best_cost
    for _ in range(refine_iters):
        q0, q1 = random_cross_pair()
        d = _swap_delta(adj, L, q0, q1, dist)
        if d < 0:
            L[q0], L[q1] = L[q1], L[q0]
            cur += d
    return L, cur


# =====================================================================================
# Stratification, step 2: subcircuit-to-chiplet mapping
# =====================================================================================
def _subcircuit_to_chip_qap(assign, adj, arch: ChipletArchitecture, seed, iters=4000):
    """Permute subcircuit labels among equal-capacity chiplets to minimize
    sum_{s,t} W[s,t] * chip_dist[pi(s), pi(t)] (small QAP by annealing)."""
    nc = arch.num_chiplets
    W = np.zeros((nc, nc))
    for q, nb in enumerate(adj):
        for p, w in nb:
            if p > q and assign[p] != assign[q]:
                W[assign[q], assign[p]] += w
                W[assign[p], assign[q]] += w
    cap = [len(c) for c in arch.chiplets]
    E = arch.chip_dist
    pi = list(range(nc))
    cost = lambda pi: float((W * E[np.ix_(pi, pi)]).sum()) / 2  # noqa: E731
    cur = cost(pi)
    best, best_c = list(pi), cur
    rng = random.Random(seed)
    groups = [s for s in range(nc) if sum(1 for t in range(nc) if cap[t] == cap[s]) > 1]
    if len(groups) < 2:
        return best
    T = max(1.0, W.max())
    for _ in range(iters):
        s, t = rng.sample(groups, 2)
        if cap[s] != cap[t]:
            continue
        pi[s], pi[t] = pi[t], pi[s]
        c = cost(pi)
        if c <= cur or rng.random() < math.exp(-(c - cur) / T):
            cur = c
            if c < best_c:
                best, best_c = list(pi), c
        else:
            pi[s], pi[t] = pi[t], pi[s]
        T *= 0.999
    return best


def _random_perm_same_capacity(arch, rng):
    cap = [len(c) for c in arch.chiplets]
    pi = list(range(arch.num_chiplets))
    by_cap = {}
    for c in range(arch.num_chiplets):
        by_cap.setdefault(cap[c], []).append(c)
    for group in by_cap.values():
        shuffled = group[:]
        rng.shuffle(shuffled)
        for a, b in zip(group, shuffled):
            pi[a] = b
    return pi


# =====================================================================================
# Stratification, step 3: inter-chiplet routing (Algorithm 2)
# =====================================================================================
def _inter_chiplet_route(args):
    ops, assign0, chip_adj, E, seed = args
    rng = random.Random(seed)
    N = len(assign0)
    chip = list(assign0)
    nc = len(chip_adj)
    members = [set() for _ in range(nc)]
    for q, c in enumerate(chip):
        members[c].add(q)

    nops = len(ops)
    preds = [0] * nops
    succ = [[] for _ in range(nops)]
    last_q, last_c = {}, {}
    for i, (_, qs, cs) in enumerate(ops):
        ps = {last_q[q] for q in qs if q in last_q} | {last_c[c] for c in cs if c in last_c}
        preds[i] = len(ps)
        for p in ps:
            succ[p].append(i)
        for q in qs:
            last_q[q] = i
        for c in cs:
            last_c[c] = i

    rem = [Counter() for _ in range(N)]  # unexecuted 2q-gate partners
    for _, qs, _ in ops:
        if len(qs) == 2:
            a, b = qs
            rem[a][b] += 1
            rem[b][a] += 1

    front: set[int] = set()
    fq: dict[int, int] = {}  # qubit -> front op
    schedule = []

    def add_front(i):
        front.add(i)
        for q in ops[i][1]:
            fq[q] = i

    def step(cands):  # advances the wavefront (Alg. 2 Step)
        work = deque(c for c in cands if c is not None)
        executed = 0
        while work:
            i = work.popleft()
            if i not in front:
                continue
            qs = ops[i][1]
            if len(qs) == 2 and chip[qs[0]] != chip[qs[1]]:
                continue
            front.discard(i)
            for q in qs:
                if fq.get(q) == i:
                    del fq[q]
            if len(qs) == 2:
                a, b = qs
                rem[a][b] -= 1
                rem[b][a] -= 1
                if not rem[a][b]:
                    del rem[a][b], rem[b][a]
            schedule.append(("op", i, chip[qs[0]]))
            executed += 1
            for s in succ[i]:
                preds[s] -= 1
                if preds[s] == 0:
                    add_front(s)
                    work.append(s)
        return executed

    for i in range(nops):
        if preds[i] == 0:
            add_front(i)
    step(list(front))

    def gate_dist(g, newchip):
        a, b = ops[g][1]
        return E[newchip(a)][newchip(b)]

    def evaluate(x, w):
        """Score SWAP moving x: chip[x] -> chip[w] and w the other way."""
        A, B = chip[x], chip[w]
        newchip = lambda q: B if q == x else (A if q == w else chip[q])  # noqa: E731
        oldchip = lambda q: chip[q]  # noqa: E731
        improved, harmed = set(), set()
        for g in {fq.get(x), fq.get(w)} - {None}:
            d0, d1 = gate_dist(g, oldchip), gate_dist(g, newchip)
            if d1 < d0:
                improved.add(g)
            elif d1 > d0:
                harmed.add(g)
        # Heuristic(): change in #remaining 2q gates that straddle chiplets (+ distance tiebreak)
        dcross = ddist = 0
        for q, cq_old, cq_new, other in ((x, A, B, w), (w, B, A, x)):
            for p, cnt in rem[q].items():
                if p == other:
                    continue
                cp = chip[p]
                dcross += cnt * ((cp != cq_new) - (cp != cq_old))
                ddist += cnt * (E[cq_new][cp] - E[cq_old][cp])
        idle = sum(rem[w].values())
        return improved, harmed, (dcross, ddist, idle, rng.random())

    last_pair = None
    stall, stall_limit = 0, 4 * nc + 16
    forced_gate = None
    n_swaps = 0
    while front:
        blocked = sorted(front)
        if forced_gate is not None and forced_gate not in front:
            forced_gate = None
        gates = [forced_gate] if forced_gate is not None else blocked
        tiers = {0: [], 1: [], 2: []}  # symbiotic, commensalistic, parasitic
        seen = set()
        for g in gates:
            u, v = ops[g][1]
            for x, tgt in ((u, chip[v]), (v, chip[u])):
                A = chip[x]
                for Nb in chip_adj[A]:
                    if E[Nb][tgt] >= E[A][tgt]:
                        continue
                    for w in members[Nb]:
                        key = (min(x, w), max(x, w))
                        if key in seen or (key == last_pair and forced_gate is None):
                            continue
                        seen.add(key)
                        improved, harmed, h = evaluate(x, w)
                        if g not in improved:
                            continue
                        if forced_gate is not None:
                            tiers[0].append((h, x, w))
                        elif harmed:
                            tiers[2].append((h, x, w))
                        elif len(improved) >= 2:
                            tiers[0].append((h, x, w))
                        else:
                            tiers[1].append((h, x, w))
        pool = tiers[0] or tiers[1] or tiers[2]
        if not pool:  # only the anti-oscillation filter could cause this
            last_pair = None
            forced_gate = blocked[0]
            continue
        _, x, w = min(pool)
        A, B = chip[x], chip[w]
        chip[x], chip[w] = B, A
        members[A].remove(x); members[A].add(w)   # noqa: E702
        members[B].remove(w); members[B].add(x)   # noqa: E702
        schedule.append(("x", x, w, A, B))
        n_swaps += 1
        last_pair = (min(x, w), max(x, w))
        if step([fq.get(x), fq.get(w)]):
            stall = 0
        else:
            stall += 1
            if stall > stall_limit and forced_gate is None:
                forced_gate = blocked[0]  # release valve: route oldest gate greedily
    return schedule, n_swaps, chip


# =====================================================================================
# Stratified circuit
# =====================================================================================
@dataclass
class StratifiedCircuit:
    source: QuantumCircuit
    ops: list
    global_phase: float
    num_logical: int                 # qubits in source circuit (rest are idle ancillas)
    num_total: int                   # == device qubits
    initial_assign: list[int]        # logical -> chiplet
    schedule: list
    num_inter_chiplet_swaps: int
    # derived per-chiplet view
    chip_events: list[list[tuple]]   # per chiplet: ("op", op, slots, clbits) | ("port", e, slot)
    xswaps: list[tuple[int, int, int, int]]  # event e -> (A, slotA, B, slotB)
    initial_slots: list[list[int]]   # chiplet -> slot -> logical at start
    final_slots: list[list[int]]     # chiplet -> slot -> logical at end


def _build_chip_events(ops, schedule, assign0, arch):
    nc = arch.num_chiplets
    slots = [[] for _ in range(nc)]
    for q, c in enumerate(assign0):
        slots[c].append(q)
    init_slots = [list(s) for s in slots]
    slot_of = {q: (c, i) for c in range(nc) for i, q in enumerate(slots[c])}
    events = [[] for _ in range(nc)]
    xswaps = []
    for item in schedule:
        if item[0] == "op":
            _, idx, c = item
            op, qs, cs = ops[idx]
            events[c].append(("op", op, tuple(slot_of[q][1] for q in qs), cs))
        else:
            _, x, w, A, B = item
            i, j = slot_of[x][1], slot_of[w][1]
            e = len(xswaps)
            xswaps.append((A, i, B, j))
            events[A].append(("port", e, i))
            events[B].append(("port", e, j))
            slots[A][i], slots[B][j] = w, x
            slot_of[w], slot_of[x] = (A, i), (B, j)
    return events, xswaps, init_slots, [list(s) for s in slots]


# =====================================================================================
# Elaboration workers (top-level for multiprocessing)
# =====================================================================================
def _layout_worker(args):
    k, edges, events, ncl, seed = args
    qc = QuantumCircuit(k, ncl)
    has2q = False
    for ev in events:
        if ev[0] == "op":
            _, op, sl, cs = ev
            qc.append(op, list(sl), list(cs))
            has2q |= len(sl) == 2
    if not has2q or k == 1:
        return list(range(k))
    pm = PassManager([SabreLayout(CouplingMap(_bidir(edges)), seed=seed, max_iterations=4,
                                  skip_routing=True)])
    pm.run(qc)
    lay = pm.property_set["layout"]
    return [lay[qc.qubits[s]] for s in range(k)]


def _chip_pass_manager(t: Target, level: int) -> PassManager:
    cmap = t.build_coupling_map()
    basis = list(t.operation_names)
    passes = []
    if level >= 3:
        passes += [ConsolidateBlocks(target=t), UnitarySynthesis(target=t)]
    passes += [BasisTranslator(_SEL, basis, target=t)]
    if cmap is not None:
        passes += [GateDirection(cmap, target=t)]
    if level >= 1:
        passes += [Optimize1qGatesDecomposition(target=t)]
    if level >= 2:
        passes += [CommutativeCancellation(target=t), Optimize1qGatesDecomposition(target=t)]
    passes += [BasisTranslator(_SEL, basis, target=t)]
    return PassManager(passes)



def _weighted_dist(k, edges, sub_target):
    """All-pairs distance on the chiplet graph, each edge weighted by its calibrated 2q
    error relative to the chiplet median (a median edge costs 1). This is the noise-aware
    SABRE cost of Sec. 3.2.2, refreshed on every elaboration."""
    if k == 1:
        return np.zeros((1, 1))
    errs = [_edge_error(sub_target, a, b) for a, b in edges]
    med = float(np.median(errs))
    g = rx.PyGraph()
    g.add_nodes_from(range(k))
    for (a, b), e in zip(edges, errs):
        w = math.log1p(-min(e, 0.5)) / math.log1p(-min(med, 0.5)) if med > 0 else 1.0
        g.add_edge(a, b, max(w, 1e-3))
    D = np.zeros((k, k))
    apl = rx.all_pairs_dijkstra_path_lengths(g, float)
    for x in range(k):
        for y, d in apl[x].items():
            D[x, y] = d
    return D


def _ev_qubits(ev):
    return ev[2] if ev[0] == "op" else (ev[2],)


def _deps(events):
    n = len(events)
    preds, succ = [0] * n, [[] for _ in range(n)]
    lq, lc = {}, {}
    last_port = None
    for i, ev in enumerate(events):
        qs = _ev_qubits(ev)
        cs = ev[3] if ev[0] == "op" else ()
        ps = {lq[q] for q in qs if q in lq} | {lc[c] for c in cs if c in lc}
        if ev[0] == "port":
            # Inter-chiplet SWAPs are globally ordered, so a chiplet's ports must be too.
            if last_port is not None:
                ps.add(last_port)
            last_port = i
        preds[i] = len(ps)
        for p in ps:
            succ[p].append(i)
        for q in qs:
            lq[q] = i
        for c in cs:
            lc[c] = i
    return preds, succ


def _sabre_trial(k, nbrs, adjset, D, events, deps, ports, lay0, rng,
                 ext_size=20, ext_weight=0.5, decay_inc=0.001, decay_reset=5):
    """One LightSABRE-style routing trial on a chiplet with pinned ports.

    events: ("op", op, slots, clbits) | ("port", e, slot); a port is executable only when
    its slot sits on the link endpoint ports[e]. Returns (emitted, final layout, #swaps)."""
    preds0, succ = deps
    preds = list(preds0)
    lay = list(lay0)
    occ = [0] * k
    for s, p in enumerate(lay):
        occ[p] = s
    decay = [1.0] * k
    front = [i for i, p in enumerate(preds) if p == 0]
    emitted, nsw, since, n_since_reset = [], 0, 0, 0

    def is_2q(i):
        ev = events[i]
        return ev[0] == "port" or len(ev[2]) == 2

    def cost(i, pos):
        ev = events[i]
        if ev[0] == "port":
            return D[pos(ev[2]), ports[ev[1]]]
        a, b = ev[2]
        return D[pos(a), pos(b)]

    def ready(i):
        ev = events[i]
        if ev[0] == "port":
            return lay[ev[2]] == ports[ev[1]]
        sl = ev[2]
        return len(sl) < 2 or (lay[sl[0]], lay[sl[1]]) in adjset

    def do_swap(p, q):
        nonlocal nsw
        a, b = occ[p], occ[q]
        occ[p], occ[q] = b, a
        lay[a], lay[b] = q, p
        emitted.append(("swap", p, q))
        nsw += 1

    def advance():
        nonlocal front
        progressed = False
        work = deque(front)
        blocked = []
        while work:
            i = work.popleft()
            if not ready(i):
                blocked.append(i)
                continue
            ev = events[i]
            if ev[0] == "op":
                emitted.append(("op", ev[1], tuple(lay[s] for s in ev[2]), ev[3]))
            else:
                emitted.append(("port", ev[1], lay[ev[2]]))
            progressed = True
            for j in succ[i]:
                preds[j] -= 1
                if preds[j] == 0:
                    work.append(j)
        front = blocked
        return progressed

    def extended():
        out, seen = [], set(front)
        dq = deque(front)
        while dq and len(out) < ext_size:
            i = dq.popleft()
            for j in succ[i]:
                if j not in seen:
                    seen.add(j)
                    dq.append(j)
                    if is_2q(j):
                        out.append(j)
                        if len(out) >= ext_size:
                            break
        return out

    advance()
    while front:
        ext = extended()
        cands = set()
        for i in front:
            for s in _ev_qubits(events[i]):
                p = lay[s]
                for q in nbrs[p]:
                    cands.add((min(p, q), max(p, q)))
        best, best_h = [], None
        for p, q in cands:
            def pos(s, p=p, q=q):
                x = lay[s]
                return q if x == p else (p if x == q else x)
            h = sum(cost(i, pos) for i in front) / len(front)
            if ext:
                h += ext_weight * sum(cost(i, pos) for i in ext) / len(ext)
            h *= max(decay[p], decay[q])
            if best_h is None or h < best_h - 1e-12:
                best, best_h = [(p, q)], h
            elif abs(h - best_h) <= 1e-12:
                best.append((p, q))
        p, q = rng.choice(sorted(best))
        do_swap(p, q)
        decay[p] += decay_inc
        decay[q] += decay_inc
        n_since_reset += 1
        if n_since_reset >= decay_reset:
            decay = [1.0] * k
            n_since_reset = 0
        if advance():
            since = 0
            decay = [1.0] * k
        else:
            since += 1
            if since > 10 * k:  # release valve: walk the cheapest front gate into place
                i = min(front, key=lambda i: cost(i, lambda s: lay[s]))
                ev = events[i]
                if ev[0] == "port":
                    src, dst = lay[ev[2]], ports[ev[1]]
                    path = list(rx.dijkstra_shortest_paths(_G_CACHE[id(nbrs)], src, dst)[dst])
                    steps = list(zip(path, path[1:]))
                else:
                    src, dst = lay[ev[2][0]], lay[ev[2][1]]
                    path = list(rx.dijkstra_shortest_paths(_G_CACHE[id(nbrs)], src, dst)[dst])
                    steps = list(zip(path, path[1:]))[:-1]
                for a, b in steps:
                    do_swap(a, b)
                advance()
                since = 0
    return emitted, lay, nsw


_G_CACHE: dict = {}


def _chip_router(k, edges, D, events, ports, lay0, seed, trials, refine_iters):
    nbrs = [[] for _ in range(k)]
    for a, b in edges:
        nbrs[a].append(b)
        nbrs[b].append(a)
    adjset = set(_bidir(edges))
    g = rx.PyGraph()
    g.add_nodes_from(range(k))
    for a, b in edges:
        g.add_edge(a, b, None)
    _G_CACHE[id(nbrs)] = g
    fwd, bwd = _deps(events), _deps(events[::-1])
    rng = random.Random(seed)

    def best_of(evs, deps, lay):
        runs = [_sabre_trial(k, nbrs, adjset, D, evs, deps, ports, lay, random.Random(rng.random()))
                for _ in range(trials)]
        return min(runs, key=lambda r: r[2])

    lay = list(lay0)
    best_init, best_run = lay, best_of(events, fwd, lay)
    for _ in range(refine_iters):  # SABRE forward/backward refinement, ports pinned
        _, back_lay, _ = best_of(events[::-1], bwd, best_run[1])
        run = best_of(events, fwd, back_lay)
        if run[2] < best_run[2]:
            best_init, best_run = back_lay, run
    del _G_CACHE[id(nbrs)]
    return best_init, best_run


def _route_optimize_worker(args):
    """Intra-chiplet routing with pinned link ports, then translation/optimization."""
    (k, edges, events, ports, init_layout, ncl, sub_target, level, seed, trials,
     refine_iters) = args
    D = _weighted_dist(k, edges, sub_target)
    init, (emitted, final, _) = _chip_router(k, edges, D, events, ports, init_layout, seed,
                                              trials, refine_iters)
    out = QuantumCircuit(k, ncl)
    for item in emitted:
        if item[0] == "op":
            _, op, qs, cs = item
            out.append(op, list(qs), list(cs))
        elif item[0] == "swap":
            out.swap(item[1], item[2])
        else:
            out.append(Barrier(1, label=f"{_PORT_LABEL}{item[1]}"), [item[2]])
    return _chip_pass_manager(sub_target, level).run(out), final, init


def _pmap(fn, argl, n_jobs):
    argl = list(argl)
    if n_jobs <= 1 or len(argl) <= 1:
        return [fn(a) for a in argl]
    with ProcessPoolExecutor(max_workers=min(n_jobs, len(argl))) as ex:
        return list(ex.map(fn, argl))


# =====================================================================================
# The compiler
# =====================================================================================
class SEQCCompiler:
    """Stratify-Elaborate Quantum Compiler.

    Parameters
    ----------
    backend : BackendV2 or Target
    chiplets : optional explicit partition of physical qubits into chiplets
    inter_link_gates : 2q gate names that identify an edge as an inter-chiplet link
    optimization_level : 0-3, controls the per-chiplet optimization stage
    sa_trials, sa_T0, sa_alpha, sa_beta : Algorithm 1 parameters (paper: T0=200,
        T/50 changes per round, 0.5% cooling per round)
    distance_aware_partition : weight the Alg. 1 cost by chiplet distance (paper: False)
    routing_trials : # of subcircuit-to-chiplet permutations tried with Alg. 2
    layout_refine_iters : forward/backward SABRE passes refining each chiplet's layout
        once its link ports are known
    swap_trials : randomized routing trials per chiplet (best kept)
    n_jobs : worker processes (default: all CPUs)
    """

    def __init__(self, backend, chiplets=None, inter_link_gates=("swap",), optimization_level=3,
                 seed: int | None = None, sa_trials: int = 8, sa_T0: float = 200.0,
                 sa_alpha: float = 1 / 50, sa_beta: float = 0.995,
                 distance_aware_partition: bool = False, routing_trials: int = 8,
                 layout_refine_iters: int = 2, swap_trials: int = 4,
                 n_jobs: int | None = None):
        self.target = _as_target(backend)
        self.arch = ChipletArchitecture.from_target(self.target, chiplets, inter_link_gates)
        self.optimization_level = optimization_level
        self.seed = seed if seed is not None else random.randrange(2**31)
        self.sa = (sa_trials, sa_T0, sa_alpha, sa_beta)
        self.distance_aware_partition = distance_aware_partition
        self.routing_trials = routing_trials
        self.swap_trials = swap_trials
        self.layout_refine_iters = layout_refine_iters
        self.n_jobs = n_jobs or os.cpu_count() or 1
        self._sub_targets = None

    # ------------------------------------------------------------------ stratification
    def stratify(self, circuit: QuantumCircuit) -> StratifiedCircuit:
        arch = self.arch
        N = arch.num_qubits
        if circuit.num_qubits > N:
            raise ValueError(f"Circuit has {circuit.num_qubits} qubits; device has {N}.")
        ops, gphase = _preprocess(circuit)
        adj = _interaction_graph(ops, N)       # padded with idle ancillas up to N
        caps = [len(c) for c in arch.chiplets]

        # 1. qubit-to-subcircuit mapping (Alg. 1), parallel independent trials
        trials, T0, alpha, beta = self.sa
        greedy = _greedy_initial_assignment(adj, caps, arch.chip_adj)
        rng = random.Random(self.seed)
        dist = arch.chip_dist.tolist() if self.distance_aware_partition else None
        inits = [greedy]
        for _ in range(trials - 1):
            perm = list(range(N))
            rng.shuffle(perm)
            a = [0] * N
            it = iter(perm)
            for c, k in enumerate(caps):
                for _ in range(k):
                    a[next(it)] = c
            inits.append(a)
        sa_args = [(adj, a, T0, alpha, beta, 20 * N, dist, self.seed + t) for t, a in enumerate(inits)]
        results = _pmap(_sa_worker, sa_args, self.n_jobs)
        assign, _ = min(results, key=lambda r: r[1])

        # 2. subcircuit-to-chiplet mapping: QAP seed + identity + random permutations,
        #    each scored by a full inter-chiplet routing run (3).
        perms = [_subcircuit_to_chip_qap(assign, adj, arch, self.seed), list(range(arch.num_chiplets))]
        perms += [_random_perm_same_capacity(arch, rng) for _ in range(max(0, self.routing_trials - 2))]
        uniq = []
        for p in perms:
            if p not in uniq:
                uniq.append(p)
        E = arch.chip_dist.tolist()
        route_args = [(ops, [p[c] for c in assign], arch.chip_adj, E, self.seed + i)
                      for i, p in enumerate(uniq)]
        routed = _pmap(_inter_chiplet_route, route_args, self.n_jobs)
        best = min(range(len(routed)), key=lambda i: (routed[i][1], len(routed[i][0])))
        schedule, n_x, _ = routed[best]
        assign0 = route_args[best][1]

        events, xswaps, init_slots, final_slots = _build_chip_events(ops, schedule, assign0, arch)
        return StratifiedCircuit(circuit, ops, gphase, circuit.num_qubits, N, assign0, schedule,
                                 n_x, events, xswaps, init_slots, final_slots)

    # ------------------------------------------------------------------ elaboration
    def elaborate(self, strat: StratifiedCircuit, backend=None) -> QuantumCircuit:
        """Recurring stage. Pass an updated ``backend`` (same topology, new calibration)
        to re-elaborate a cached stratified circuit."""
        if backend is not None:
            self.target = _as_target(backend)
            self._sub_targets = None
        arch, target = self.arch, self.target
        nc = arch.num_chiplets
        ncl = strat.source.num_clbits
        if self._sub_targets is None:
            self._sub_targets = [_sub_target(target, qs) for qs in arch.chiplets]

        # 4. intra-chiplet layout (parallel)
        lay_args = [(len(arch.chiplets[c]), arch.intra_edges[c], strat.chip_events[c], ncl,
                     self.seed + 101 * c) for c in range(nc)]
        layouts = _pmap(_layout_worker, lay_args, self.n_jobs)

        # 5. inter-chiplet SWAP lowering (serial, greedy nearest valid link)
        est = [list(l) for l in layouts]                 # slot -> estimated local phys
        occ = []
        for c in range(nc):
            o = [0] * len(est[c])
            for s, p in enumerate(est[c]):
                o[p] = s
            occ.append(o)
        usage = Counter()
        ports = [dict() for _ in range(nc)]              # chip -> {event: local endpoint}
        link_of_event = []
        for e, (A, i, B, j) in enumerate(strat.xswaps):
            key = (min(A, B), max(A, B))
            best, best_cost = None, None
            for (pa, pb, err) in arch.links[key]:
                a, b = (pa, pb) if A == key[0] else (pb, pa)
                la, lb = arch.local_index[a], arch.local_index[b]
                cost = (arch.local_dist[A][est[A][i], la] + arch.local_dist[B][est[B][j], lb]
                        + 10.0 * err + 0.25 * usage[(a, b)])
                if best_cost is None or cost < best_cost:
                    best, best_cost = (a, b, la, lb), cost
            a, b, la, lb = best
            usage[(a, b)] += 1
            for c, s, l in ((A, i, la), (B, j, lb)):   # slot s moves to l (approx.)
                other = occ[c][l]
                occ[c][est[c][s]], occ[c][l] = other, s
                est[c][other], est[c][s] = est[c][s], l
            ports[A][e], ports[B][e] = la, lb
            link_of_event.append((a, b))

        # 6+7. intra-chiplet routing, translation, optimization (parallel)
        ro_args = [(len(arch.chiplets[c]), arch.intra_edges[c], strat.chip_events[c], ports[c],
                    layouts[c], ncl, self._sub_targets[c], self.optimization_level,
                    self.seed + 7 * c, self.swap_trials, self.layout_refine_iters)
                   for c in range(nc)]
        results = _pmap(_route_optimize_worker, ro_args, self.n_jobs)

        # 8. stitch
        out = self._stitch(strat, results, link_of_event)
        self._attach_layout(out, strat, [r[2] for r in results], [r[1] for r in results])
        self._validate(out)
        return out

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        return self.elaborate(self.stratify(circuit))

    __call__ = run

    # ------------------------------------------------------------------ helpers
    def _stitch(self, strat, results, link_of_event):
        arch, target = self.arch, self.target
        src = strat.source
        out = QuantumCircuit(QuantumRegister(arch.num_qubits, "q"), name=src.name)
        for creg in src.cregs:
            out.add_register(creg)
        loose = [b for b in src.clbits if not src.find_bit(b).registers]
        if loose:
            out.add_bits(loose)
        clbits = src.clbits
        phys_q = out.qubits
        out.global_phase = strat.global_phase

        # Each chiplet circuit is a DAG; a port marker for event e fixes where the
        # inter-chiplet SWAP goes. Before emitting SWAP e we emit exactly the (not yet
        # emitted) ancestors of chiplet A's and B's markers for e, in topological order.
        chips = []
        for c, (circ, _, _) in enumerate(results):
            out.global_phase += circ.global_phase
            dag = circuit_to_dag(circ, copy_operations=False)
            order = list(dag.topological_op_nodes())
            rank = {nd: r for r, nd in enumerate(order)}
            markers = {}
            for nd in order:
                lab = getattr(nd.op, "label", None)
                if nd.op.name == "barrier" and lab and lab.startswith(_PORT_LABEL):
                    markers[int(lab[len(_PORT_LABEL):])] = nd
            qmap = {q: phys_q[arch.chiplets[c][i]] for i, q in enumerate(dag.qubits)}
            cmap = {b: clbits[i] for i, b in enumerate(dag.clbits)}
            chips.append((dag, order, rank, markers, qmap, cmap, set(markers.values())))

        def emit(c, nodes):
            dag, order, rank, markers, qmap, cmap, done = chips[c]
            for nd in sorted((n for n in nodes if n not in done), key=rank.__getitem__):
                done.add(nd)
                out.append(nd.op, [qmap[q] for q in nd.qargs], [cmap[b] for b in nd.cargs],
                           copy=False)

        consumed = [set() for _ in chips]
        marker_nodes = [set(ch[3].values()) for ch in chips]
        for e, (A, _, B, _) in enumerate(strat.xswaps):
            for c in (A, B):
                dag, _, _, markers, _, _, _ = chips[c]
                anc = [n for n in dag.ancestors(markers[e]) if isinstance(n, DAGOpNode)]
                mset = marker_nodes[c]
                if any(n in mset and n not in consumed[c] for n in anc):
                    raise RuntimeError(f"SEQC: port order violated on chiplet {c}.")
                emit(c, anc)
                consumed[c].add(markers[e])
            a, b = link_of_event[e]
            if not target.instruction_supported("swap", (a, b)) and target.instruction_supported("swap", (b, a)):
                a, b = b, a
            out.swap(a, b)
        for c in range(arch.num_chiplets):
            emit(c, chips[c][1])

        # If the inter-chiplet links are universal (no native SWAP), translate what's left.
        if any(inst.operation.name == "swap" and not target.instruction_supported(
                "swap", tuple(out.find_bit(q).index for q in inst.qubits)) for inst in out.data):
            out = PassManager([BasisTranslator(_SEL, list(target.operation_names), target=target),
                               GateDirection(target.build_coupling_map(), target=target),
                               Optimize1qGatesDecomposition(target=target)]).run(out)
        return out

    def _attach_layout(self, out, strat, init_layouts, final_layouts):
        arch = self.arch
        N, n = arch.num_qubits, strat.num_logical

        def mapping(slots, lays):
            m = [0] * N
            for c in range(arch.num_chiplets):
                for s, l in enumerate(slots[c]):
                    m[l] = arch.chiplets[c][lays[c][s]]
            return m

        init_map = mapping(strat.initial_slots, init_layouts)
        final_map = mapping(strat.final_slots, final_layouts)
        src = strat.source
        virt = list(src.qubits)
        if N > n:
            virt += list(AncillaRegister(N - n, "ancilla"))
        initial_layout = Layout({virt[l]: init_map[l] for l in range(N)})
        for reg in src.qregs:
            initial_layout.add_register(reg)
        inv_init = {p: l for l, p in enumerate(init_map)}
        final_layout = Layout({out.qubits[p]: final_map[inv_init[p]] for p in range(N)})
        out._layout = TranspileLayout(initial_layout, {q: i for i, q in enumerate(virt)},
                                      final_layout, _input_qubit_count=n,
                                      _output_qubit_list=list(out.qubits))
        out.metadata = dict(out.metadata or {})
        out.metadata["seqc"] = {
            "initial_mapping": init_map[:n],   # logical i -> physical at start
            "final_mapping": final_map[:n],    # logical i -> physical at end
            "inter_chiplet_swaps": strat.num_inter_chiplet_swaps,
        }

    def _validate(self, out):
        for inst in out.data:
            name = inst.operation.name
            if name == "barrier":
                continue
            qargs = tuple(out.find_bit(q).index for q in inst.qubits)
            if not self.target.instruction_supported(name, qargs):
                raise RuntimeError(f"SEQC produced unsupported instruction {name} on {qargs}.")


def seqc_transpile(circuit: QuantumCircuit, backend, **kwargs) -> QuantumCircuit:
    """One-shot convenience wrapper: circuit + backend -> executable circuit."""
    return SEQCCompiler(backend, **kwargs).run(circuit)


# =====================================================================================
# Peephole correction (Sec. 3.1) and the PA-Qiskit baseline
# =====================================================================================
class PeepholeCorrection(TransformationPass):
    """Replace every non-SWAP 2q gate G on an inter-chiplet link (a,b) by
    SWAP(n,a) SWAP(a,b) G(n,a) SWAP(a,b) SWAP(n,a), with n an intra-chiplet neighbour of a
    (or the mirror image on b's side): 2 intra + 2 inter SWAPs, no net permutation."""

    def __init__(self, arch: ChipletArchitecture, target: Target):
        super().__init__()
        self.arch, self.target = arch, target
        self._nbrs = {}
        for c, edges in enumerate(arch.intra_edges):
            for x, y in edges:
                gx, gy = arch.chiplets[c][x], arch.chiplets[c][y]
                self._nbrs.setdefault(gx, []).append(gy)
                self._nbrs.setdefault(gy, []).append(gx)

    def run(self, dag):
        new = dag.copy_empty_like()
        q = dag.qubits
        for node in dag.topological_op_nodes():
            qs = [dag.find_bit(x).index for x in node.qargs]
            if (len(qs) == 2 and node.op.name != "swap" and self.arch.is_inter_chip(*qs)):
                a, b = qs
                if self._nbrs.get(a):
                    n = self._nbrs[a][0]
                    pre = [(n, a), (a, b)]
                    gate_q = [n, a]         # x lands on n, y lands on a
                elif self._nbrs.get(b):
                    n = self._nbrs[b][0]
                    pre = [(n, b), (a, b)]
                    gate_q = [b, n]
                else:
                    raise RuntimeError(f"No intra-chiplet neighbour for link ({a},{b}).")
                for x, y in pre:
                    new.apply_operation_back(SwapGate(), (q[x], q[y]), ())
                new.apply_operation_back(node.op, (q[gate_q[0]], q[gate_q[1]]), node.cargs)
                for x, y in reversed(pre):
                    new.apply_operation_back(SwapGate(), (q[x], q[y]), ())
            else:
                new.apply_operation_back(node.op, node.qargs, node.cargs)
        return new


def pa_qiskit_transpile(circuit, backend, optimization_level=2, seed=None, chiplets=None):
    """Peephole-augmented stock Qiskit (PA-Qiskit): preset pipeline with PeepholeCorrection
    inserted between routing and basis translation."""
    from qiskit.transpiler import generate_preset_pass_manager

    target = _as_target(backend)
    arch = ChipletArchitecture.from_target(target, chiplets)
    pm = generate_preset_pass_manager(optimization_level, target=target, seed_transpiler=seed)
    pm.pre_translation = PassManager([PeepholeCorrection(arch, target)])
    return pm.run(circuit)


# =====================================================================================
# Metrics (Sec. 5)
# =====================================================================================
def circuit_metrics(circuit: QuantumCircuit, backend, chiplets=None) -> dict:
    target = _as_target(backend)
    try:
        arch = ChipletArchitecture.from_target(target, chiplets)
    except ValueError:
        arch = None
    busy = [0.0] * circuit.num_qubits
    log_esp, n2q, ninter = 0.0, 0, 0
    for inst in circuit.data:
        name = inst.operation.name
        if name == "barrier":
            continue
        qs = tuple(circuit.find_bit(x).index for x in inst.qubits)
        props = None
        try:
            props = target[name].get(qs)
        except KeyError:
            pass
        err = (props.error if props and props.error is not None else 0.0)
        dur = (props.duration if props and props.duration is not None else 0.0)
        log_esp += math.log(max(1e-300, 1 - err))
        t = max(busy[x] for x in qs) + dur
        for x in qs:
            busy[x] = t
        if len(qs) == 2:
            n2q += 1
            if arch is not None and arch.is_inter_chip(*qs):
                ninter += 1
    return {
        "size": circuit.size(),
        "depth": circuit.depth(),
        "two_qubit_gates": n2q,
        "inter_chiplet_gates": ninter,
        "esp": math.exp(log_esp),
        "duration_s": max(busy) if busy else 0.0,
    }


# =====================================================================================
# Demo
# =====================================================================================
if __name__ == "__main__":
    import time
    from qiskit import transpile

    def ghz(n):
        qc = QuantumCircuit(n, n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        qc.measure(range(n), range(n))
        return qc

    backend = make_chiplet_backend(2, 3)
    universal = make_chiplet_backend(2, 3, universal_links=True)
    qc = ghz(55)
    t0 = time.time()
    comp = SEQCCompiler(backend, seed=1)
    strat = comp.stratify(qc)
    t1 = time.time()
    exe = comp.elaborate(strat)
    t2 = time.time()
    print(f"SEQC        strat {t1-t0:.2f}s elab {t2-t1:.2f}s", circuit_metrics(exe, backend))
    base = transpile(qc, universal, optimization_level=3, seed_transpiler=1)
    print("Qiskit (universal links)", circuit_metrics(base, universal))
    pa = pa_qiskit_transpile(qc, backend, seed=1)
    print("PA-Qiskit              ", circuit_metrics(pa, backend))