"""Two Gross-code modules joined by a bridge: fault-tolerant measurement of Zbar_a (x) Zbar_b.

This is the inter-module logical operation of IBM's bicycle architecture (two [[144,12,12]] Gross
code modules on separate chips, connected by couplers). The joint logical measurement is realised
by *gauging* the logical operator (Williamson & Yoder, "Low-overhead fault-tolerant quantum
computation by gauging logical operators", 2024), with the two ancilla graphs connected by
bridge edges (Cross, He, Rall & Yoder, "Improved QLDPC surgery: logical measurements and bridging
codes", 2024).

Construction
------------
Let L = Zbar_a (x) Zbar_b with support V = V_a u V_b (a minimum-weight, i.e. weight-12, logical Z of
each module). We pick a graph G = (V, E) and add one *edge qubit* per edge, initialised in |+>.
The deformed code measured during the gauging phase has the checks

* Gauss-law checks   A_v = Z_v prod_{e ~ v} Z_e          (Z-type, one per vertex; prod_v A_v = L),
* flux checks        B_p = prod_{e in p} X_e              (X-type, one per cycle of a cycle basis),
* deformed X checks  s'  = s prod_{e in gamma_s} X_e      (every X stabiliser s that touches V; gamma_s
                                                          is an edge set with boundary s n V),
* all Z stabilisers and the X stabilisers not touching V, unchanged.

The graph G contains, for every X stabiliser s, edges pairing up the vertices of s n V (so every
gamma_s consists of 1 or 2 edges), edges that make each G_a / G_b connected, optional random
expansion edges, and ``num_bridge_edges`` bridge edges between V_a and V_b.

Timeline (``rounds_*`` are syndrome-extraction rounds)
    1. both modules initialised in |0>^n (so every logical Z is +1)
    2. ``rounds_before`` rounds of the two memories
    3. edge qubits initialised in |+>, ``rounds_gauging`` rounds of the deformed code
       (the product of the first-round A_v outcomes is the measurement result of L)
    4. edge qubits measured in X (ungauging), ``rounds_after`` rounds of the two memories
    5. transversal Z readout of both modules

Syndrome extraction
-------------------
Module checks use exactly the depth-8 IBM schedule of ``circuit_generator.build_circuit`` (same
CNOT layers and qubit order), so a module looks the same to the mapper as the Gross memory
benchmark. In gauging rounds the deformed X checks get their edge CNOTs after the module layers,
the flux checks run at the same time, and the Gauss-law checks run after all X-type checks have
been measured (no interleaving of X- and Z-type checks on shared qubits, so every round measures the
deformed stabilisers exactly).

Detectors and observables
-------------------------
* Z stabilisers: every round (first round against |0>), and against the final data readout.
* X stabilisers: round to round; across the gauging boundaries via
  s'(first gauging round) = s(previous round) and s(first round after) = s'(last) x prod X_e(gamma_s).
* A_v: round to round within the gauging phase.  B_p: first round (= +1), round to round, and
  against the final X readout of the edge qubits.
* Observable 0: the logical measurement outcome, prod_v A_v of the first gauging round
  (deterministic, since both modules start in |0>).
* Observables 1..12 / 13..24: a basis of the logical Z operators of module a / b, from the final
  readout.

Qubit order (as expected by Chipmunq's ``gross_code`` partitions)
    module a: [X checks (72), L data (72), R data (72), Z checks (72)]   qubits   0 .. 287
    module b: same                                                        qubits 288 .. 575
    bridge:   [edge qubits, Gauss-law ancillas, flux ancillas]            qubits 576 ..
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field

import numpy as np
import stim

# Gross code [[144, 12, 12]]: l = 12, m = 6, A = x^3 + y + y^2, B = y^3 + x + x^2
GROSS_L, GROSS_M = 12, 6
GROSS_DISTANCE = 12


# --------------------------------------------------------------------------------------
# GF(2) linear algebra
# --------------------------------------------------------------------------------------


def gf2_rref(mat: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Reduced row echelon form over GF(2). Returns (rref without zero rows, pivot columns)."""
    m = (np.array(mat, dtype=np.uint8) % 2).copy()
    rows, cols = m.shape
    pivots = []
    r = 0
    for c in range(cols):
        if r == rows:
            break
        nz = np.nonzero(m[r:, c])[0]
        if nz.size == 0:
            continue
        p = r + nz[0]
        if p != r:
            m[[r, p]] = m[[p, r]]
        hit = np.nonzero(m[:, c])[0]
        hit = hit[hit != r]
        m[hit] ^= m[r]
        pivots.append(c)
        r += 1
    return m[:r], pivots


def gf2_rank(mat: np.ndarray) -> int:
    return len(gf2_rref(mat)[1])


def gf2_nullspace(mat: np.ndarray) -> np.ndarray:
    """Basis (rows) of {x : mat @ x = 0} over GF(2)."""
    rref, pivots = gf2_rref(mat)
    n = mat.shape[1]
    free = [c for c in range(n) if c not in set(pivots)]
    basis = np.zeros((len(free), n), dtype=np.uint8)
    for i, f in enumerate(free):
        basis[i, f] = 1
        for r, p in enumerate(pivots):
            basis[i, p] = rref[r, f]
    return basis


def gf2_in_rowspace(rowspace_rref: np.ndarray, pivots: list[int], v: np.ndarray) -> bool:
    v = v.astype(np.uint8).copy()
    for r, p in enumerate(pivots):
        if v[p]:
            v ^= rowspace_rref[r]
    return not v.any()


# --------------------------------------------------------------------------------------
# Bivariate bicycle code
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BBCode:
    l: int
    m: int
    hx: np.ndarray  # (n/2, n): [A | B]
    hz: np.ndarray  # (n/2, n): [B^T | A^T]
    A_list: tuple[np.ndarray, ...]  # [x^3, y, y^2]
    B_list: tuple[np.ndarray, ...]  # [y^3, x, x^2]

    @property
    def n(self) -> int:
        return self.hx.shape[1]

    @property
    def n_half(self) -> int:
        return self.n // 2


def _shift(size: int) -> np.ndarray:
    # Same convention as circuit_generator.create_circulant_matrix(size, [-1])
    s = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        s[(i - 1) % size, i] = 1
    return s


def gross_code() -> BBCode:
    """[[144, 12, 12]] Gross code with the conventions of circuit_generator.create_bivariate_bicycle_codes."""
    l, m = GROSS_L, GROSS_M
    x = np.kron(_shift(l), np.eye(m, dtype=np.uint8))
    y = np.kron(np.eye(l, dtype=np.uint8), _shift(m))
    pw = lambda mat, p: np.linalg.matrix_power(mat.astype(np.int64), p) % 2
    A_list = (pw(x, 3), pw(y, 1), pw(y, 2))
    B_list = (pw(y, 3), pw(x, 1), pw(x, 2))
    A = sum(A_list) % 2
    B = sum(B_list) % 2
    hx = np.hstack([A, B]).astype(np.uint8)
    hz = np.hstack([B.T, A.T]).astype(np.uint8)
    assert not ((hx.astype(int) @ hz.T.astype(int)) % 2).any()
    return BBCode(l, m, hx, hz, A_list, B_list)


def logical_z_basis(code: BBCode) -> np.ndarray:
    """Basis of logical Z operators: ker(hx) modulo rowspace(hz)."""
    ker = gf2_nullspace(code.hx)
    rs, piv = gf2_rref(code.hz)
    basis = []
    cur, cur_piv = rs, piv
    for v in ker:
        if not gf2_in_rowspace(cur, cur_piv, v):
            basis.append(v)
            cur, cur_piv = gf2_rref(np.vstack([code.hz] + basis))
    k = code.n - gf2_rank(code.hx) - gf2_rank(code.hz)
    assert len(basis) == k, (len(basis), k)
    return np.array(basis, dtype=np.uint8)


def find_min_weight_logical_z(code: BBCode, target_weight: int = GROSS_DISTANCE, max_iter: int = 20000,
                              seed: int = 0) -> np.ndarray:
    """Randomised information-set search for a low-weight logical Z (weight ``target_weight``)."""
    rng = np.random.default_rng(seed)
    rs, piv = gf2_rref(code.hz)
    best = None
    n = code.n
    for _ in range(max_iter):
        perm = rng.permutation(n)
        ker = gf2_nullspace(code.hx[:, perm])  # systematic basis in permuted coordinates
        for v in ker:
            w = int(v.sum())
            if best is not None and w >= best.sum():
                continue
            x = np.zeros(n, dtype=np.uint8)
            x[perm] = v
            if not gf2_in_rowspace(rs, piv, x):
                best = x
        if best is not None and best.sum() <= target_weight:
            return best
    raise RuntimeError(f"No logical Z of weight <= {target_weight} found (best: "
                       f"{None if best is None else int(best.sum())}).")


# --------------------------------------------------------------------------------------
# Gauging graph with bridge
# --------------------------------------------------------------------------------------

Vertex = tuple[str, int]  # (module "a" / "b", data qubit index 0..n-1 within the module)
Edge = tuple[Vertex, Vertex]


@dataclass
class BridgeLayout:
    code: BBCode
    z_logical: np.ndarray  # logical Z of one module (same support used for both)
    vertices: list[Vertex]
    edges: list[Edge]
    bridge_edges: list[Edge]
    # (module, X-check index) -> edges of gamma_s (only for checks touching V)
    gamma: dict[tuple[str, int], list[int]]
    # flux checks: list of edge-index lists
    cycles: list[list[int]] = field(default_factory=list)


def build_bridge_graph(code: BBCode, z_logical: np.ndarray, num_bridge_edges: int = 2,
                       expansion_degree: int = 0, seed: int = 0) -> BridgeLayout:
    """Build the gauging graph G on V = supp(Zbar_a) u supp(Zbar_b).

    :param num_bridge_edges: number of edges between the two modules (>= 1); every additional
        bridge edge closes a cycle and adds one flux check
    :param expansion_degree: add random edges inside each module until every vertex has at least
        this degree (0 = only the edges required by the deformation and for connectivity)
    """
    import networkx as nx

    if num_bridge_edges < 1:
        raise ValueError("num_bridge_edges must be >= 1 (the graph must be connected).")
    rng = random.Random(seed)
    support = [int(i) for i in np.nonzero(z_logical)[0]]
    edges: list[Edge] = []
    edge_index: dict[frozenset, int] = {}

    def add_edge(u: Vertex, v: Vertex) -> int:
        key = frozenset((u, v))
        if key not in edge_index:
            edge_index[key] = len(edges)
            edges.append((u, v))
        return edge_index[key]

    gamma: dict[tuple[str, int], list[int]] = {}
    for mod in ("a", "b"):
        verts = [(mod, i) for i in support]
        # 1) Deformation edges: pair up s n V for every X check s.
        for c in range(code.hx.shape[0]):
            hit = [v for v in verts if code.hx[c, v[1]]]
            if not hit:
                continue
            if len(hit) % 2:
                raise AssertionError("X check overlaps the logical in an odd number of qubits.")
            gamma[(mod, c)] = [add_edge(hit[2 * j], hit[2 * j + 1]) for j in range(len(hit) // 2)]
        # 2) Connectivity inside the module.
        g = nx.Graph()
        g.add_nodes_from(verts)
        g.add_edges_from(e for e in edges if e[0][0] == mod)
        comps = [sorted(c) for c in nx.connected_components(g)]
        comps.sort()
        for c1, c2 in zip(comps, comps[1:]):
            add_edge(c1[-1], c2[0])
        # 3) Optional expansion edges.
        if expansion_degree > 0:
            deg = {v: 0 for v in verts}
            for u, v in edges:
                if u[0] == mod:
                    deg[u] += 1
                    deg[v] += 1
            for _ in range(10 * len(verts) * expansion_degree):
                low = [v for v in verts if deg[v] < expansion_degree]
                if not low:
                    break
                u = rng.choice(low)
                cand = [v for v in verts if v != u and frozenset((u, v)) not in edge_index]
                if not cand:
                    break
                v = min(cand, key=lambda w: (deg[w], rng.random()))
                add_edge(u, v)
                deg[u] += 1
                deg[v] += 1

    # 4) Bridge edges between the modules, evenly spread over the support.
    num_bridge_edges = min(num_bridge_edges, len(support))
    picks = [support[round(j * (len(support) - 1) / max(1, num_bridge_edges - 1))] for j in range(num_bridge_edges)]
    bridge = [edges[add_edge(("a", i), ("b", i))] for i in dict.fromkeys(picks)]

    # 5) Flux checks: a cycle basis of G.
    g = nx.Graph()
    vertices = [(mod, i) for mod in ("a", "b") for i in support]
    g.add_nodes_from(vertices)
    g.add_edges_from(edges)
    if not nx.is_connected(g):
        raise AssertionError("Gauging graph is not connected.")
    cycles = []
    for cyc in nx.minimum_cycle_basis(g):
        # minimum_cycle_basis returns node sets; recover the cycle's edges in order
        sub = g.subgraph(cyc)
        order = nx.find_cycle(sub) if nx.cycle_basis(sub) else []
        cyc_edges = sorted(edge_index[frozenset((u, v))] for u, v in order)
        cycles.append(cyc_edges)
    return BridgeLayout(code=code, z_logical=z_logical, vertices=vertices, edges=edges, bridge_edges=bridge,
                        gamma=gamma, cycles=cycles)


def deformed_code(layout: BridgeLayout) -> tuple[np.ndarray, np.ndarray]:
    """Check matrices (hx, hz) of the deformed code on [data a | data b | edges]."""
    code = layout.code
    n, ne = code.n, len(layout.edges)
    ntot = 2 * n + ne
    off = {"a": 0, "b": n}
    vidx = {v: off[v[0]] + v[1] for v in layout.vertices}
    hx, hz = [], []
    for mod in ("a", "b"):
        for c in range(code.hx.shape[0]):
            row = np.zeros(ntot, dtype=np.uint8)
            row[off[mod]:off[mod] + n] = code.hx[c]
            for e in layout.gamma.get((mod, c), []):
                row[2 * n + e] ^= 1
            hx.append(row)
        for c in range(code.hz.shape[0]):
            row = np.zeros(ntot, dtype=np.uint8)
            row[off[mod]:off[mod] + n] = code.hz[c]
            hz.append(row)
    for cyc in layout.cycles:
        row = np.zeros(ntot, dtype=np.uint8)
        row[[2 * n + e for e in cyc]] = 1
        hx.append(row)
    for v in layout.vertices:
        row = np.zeros(ntot, dtype=np.uint8)
        row[vidx[v]] = 1
        for e, (u, w) in enumerate(layout.edges):
            if v in (u, w):
                row[2 * n + e] ^= 1
        hz.append(row)
    return np.array(hx), np.array(hz)


def check_deformed_code(layout: BridgeLayout, distance_iters: int = 0, seed: int = 0) -> dict:
    """Consistency checks of the deformed code; optionally a randomised distance upper bound."""
    hx, hz = deformed_code(layout)
    if ((hx.astype(int) @ hz.T.astype(int)) % 2).any():
        raise AssertionError("Deformed code checks do not commute.")
    n = hx.shape[1]
    k = n - gf2_rank(hx) - gf2_rank(hz)
    k_expected = 2 * (layout.code.n - gf2_rank(layout.code.hx) - gf2_rank(layout.code.hz)) - 1
    if k != k_expected:
        raise AssertionError(f"Deformed code has k={k}, expected {k_expected} (flux checks incomplete?).")
    info = {"n": n, "k": k, "num_edges": len(layout.edges), "num_cycles": len(layout.cycles),
            "max_check_weight": int(max(hx.sum(1).max(), hz.sum(1).max()))}
    if distance_iters:
        rng = np.random.default_rng(seed)
        for name, h_same, h_other in (("dZ", hx, hz), ("dX", hz, hx)):
            rs, piv = gf2_rref(h_other)
            best = n
            for _ in range(distance_iters):
                perm = rng.permutation(n)
                for v in gf2_nullspace(h_same[:, perm]):
                    w = int(v.sum())
                    if w < best:
                        x = np.zeros(n, dtype=np.uint8)
                        x[perm] = v
                        if not gf2_in_rowspace(rs, piv, x):
                            best = w
            info[name + "_upper_bound"] = best
    return info


# --------------------------------------------------------------------------------------
# Stim circuit
# --------------------------------------------------------------------------------------


def _greedy_layers(ops: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Pack 2-qubit gates into layers in which every qubit is used at most once (order-preserving)."""
    layers: list[list[tuple[int, int]]] = []
    busy: list[set[int]] = []
    last_layer: dict[int, int] = {}
    for a, b in ops:
        start = max(last_layer.get(a, -1), last_layer.get(b, -1)) + 1
        i = start
        while i < len(layers) and (a in busy[i] or b in busy[i]):
            i += 1
        if i == len(layers):
            layers.append([])
            busy.append(set())
        layers[i].append((a, b))
        busy[i].update((a, b))
        last_layer[a] = last_layer[b] = i
    return layers


class _Builder:
    """stim.Circuit with absolute measurement bookkeeping."""

    def __init__(self, noise: float):
        self.c = stim.Circuit()
        self.num_meas = 0
        self.p = noise

    def reset(self, basis: str, qubits: list[int]) -> None:
        if qubits:
            self.c.append("R" if basis == "Z" else "RX", qubits)
            if self.p:
                self.c.append("X_ERROR" if basis == "Z" else "Z_ERROR", qubits, self.p)

    def measure(self, basis: str, qubits: list[int]) -> list[int]:
        if not qubits:
            return []
        if self.p:
            self.c.append("X_ERROR" if basis == "Z" else "Z_ERROR", qubits, self.p)
        self.c.append("M" if basis == "Z" else "MX", qubits)
        idx = list(range(self.num_meas, self.num_meas + len(qubits)))
        self.num_meas += len(qubits)
        return idx

    def cnots(self, pairs: list[tuple[int, int]]) -> None:
        if pairs:
            flat = [q for pr in pairs for q in pr]
            self.c.append("CX", flat)
            if self.p:
                self.c.append("DEPOLARIZE2", flat, self.p)

    def tick(self) -> None:
        self.c.append("TICK")

    def detector(self, meas: list[int]) -> None:
        self.c.append("DETECTOR", [stim.target_rec(m - self.num_meas) for m in meas])

    def observable(self, index: int, meas: list[int]) -> None:
        self.c.append("OBSERVABLE_INCLUDE", [stim.target_rec(m - self.num_meas) for m in meas], index)


def _nnz(mat: np.ndarray) -> np.ndarray:
    """Column of the single nonzero entry of every row (as in circuit_generator.build_circuit)."""
    rows, cols = np.nonzero(mat)
    return cols[np.argsort(rows, kind="stable")]


def _module_layers(code: BBCode, off: int) -> list[list[tuple[int, int]]]:
    """The 7 CNOT layers of the IBM depth-8 BB syndrome circuit (circuit_generator.build_circuit)."""
    nh = code.n_half
    Xo, Lo, Ro, Zo = off, off + nh, off + 2 * nh, off + 3 * nh
    a1, a2, a3 = code.A_list
    b1, b2, b3 = code.B_list
    A1, A2, A3 = _nnz(a1), _nnz(a2), _nnz(a3)
    B1, B2, B3 = _nnz(b1), _nnz(b2), _nnz(b3)
    A1T, A2T, A3T = _nnz(a1.T), _nnz(a2.T), _nnz(a3.T)
    B1T, B2T, B3T = _nnz(b1.T), _nnz(b2.T), _nnz(b3.T)
    r = range(nh)
    return [
        [(Ro + A1T[i], Zo + i) for i in r],
        [(Xo + i, Lo + A2[i]) for i in r] + [(Ro + A3T[i], Zo + i) for i in r],
        [(Xo + i, Ro + B2[i]) for i in r] + [(Lo + B1T[i], Zo + i) for i in r],
        [(Xo + i, Ro + B1[i]) for i in r] + [(Lo + B2T[i], Zo + i) for i in r],
        [(Xo + i, Ro + B3[i]) for i in r] + [(Lo + B3T[i], Zo + i) for i in r],
        [(Xo + i, Lo + A1[i]) for i in r] + [(Ro + A2T[i], Zo + i) for i in r],
        [(Xo + i, Lo + A3[i]) for i in r],
    ]


@dataclass
class QubitMap:
    n: int                      # data qubits per module (144)
    num_edges: int
    num_vertices: int
    num_cycles: int

    def module_offset(self, mod: str) -> int:
        return 0 if mod == "a" else 2 * self.n

    def x_anc(self, mod: str, i: int) -> int:
        return self.module_offset(mod) + i

    def data(self, mod: str, i: int) -> int:
        return self.module_offset(mod) + self.n // 2 + i

    def z_anc(self, mod: str, i: int) -> int:
        return self.module_offset(mod) + 3 * self.n // 2 + i

    @property
    def bridge_offset(self) -> int:
        return 4 * self.n

    def edge(self, e: int) -> int:
        return self.bridge_offset + e

    def gauss_anc(self, v: int) -> int:
        return self.bridge_offset + self.num_edges + v

    def flux_anc(self, p: int) -> int:
        return self.bridge_offset + self.num_edges + self.num_vertices + p

    @property
    def num_qubits(self) -> int:
        return self.bridge_offset + self.num_edges + self.num_vertices + self.num_cycles


def build_bridge_circuit(layout: BridgeLayout, rounds_before: int = 1, rounds_gauging: int = GROSS_DISTANCE,
                         rounds_after: int = 1, noise: float = 0.0) -> stim.Circuit:
    """Stim circuit of the joint logical measurement Zbar_a (x) Zbar_b (see module docstring)."""
    if rounds_gauging < 1 or rounds_before < 0 or rounds_after < 0:
        raise ValueError("Need rounds_gauging >= 1 and rounds_before, rounds_after >= 0.")
    code = layout.code
    nh = code.n_half
    qm = QubitMap(code.n, len(layout.edges), len(layout.vertices), len(layout.cycles))
    vpos = {v: k for k, v in enumerate(layout.vertices)}
    mods = ("a", "b")
    b = _Builder(noise)

    module_layers = [_module_layers(code, qm.module_offset(m)) for m in mods]
    x_ancs = [qm.x_anc(m, i) for m in mods for i in range(nh)]
    z_ancs = [qm.z_anc(m, i) for m in mods for i in range(nh)]
    data = [qm.data(m, i) for m in mods for i in range(code.n)]
    edges = [qm.edge(e) for e in range(qm.num_edges)]
    gauss = [qm.gauss_anc(k) for k in range(qm.num_vertices)]
    flux = [qm.flux_anc(p) for p in range(qm.num_cycles)]

    # X-type ops that touch edge qubits (deformed X checks and flux checks), and the Gauss-law ops.
    x_edge_ops = [(qm.x_anc(m, c), qm.edge(e)) for (m, c), es in sorted(layout.gamma.items()) for e in es]
    x_edge_ops += [(qm.flux_anc(p), qm.edge(e)) for p, cyc in enumerate(layout.cycles) for e in cyc]
    x_edge_layers = _greedy_layers(x_edge_ops)
    gauss_ops = []
    for v in layout.vertices:
        gauss_ops.append((qm.data(v[0], v[1]), qm.gauss_anc(vpos[v])))
    for e, (u, w) in enumerate(layout.edges):
        gauss_ops.append((qm.edge(e), qm.gauss_anc(vpos[u])))
        gauss_ops.append((qm.edge(e), qm.gauss_anc(vpos[w])))
    gauss_layers = _greedy_layers(gauss_ops)

    prev: dict[tuple, int] = {}  # check key -> measurement index of its last outcome

    def syndrome_round(gauging: bool) -> dict[tuple, int]:
        b.reset("X", x_ancs + (flux if gauging else []))
        b.reset("Z", z_ancs + (gauss if gauging else []))
        b.tick()
        for layer in range(7):
            b.cnots([op for ml in module_layers for op in ml[layer]])
            if layer == 6:
                mz = b.measure("Z", z_ancs)
            b.tick()
        if gauging:
            for layer in x_edge_layers:
                b.cnots(layer)
                b.tick()
        mx = b.measure("X", x_ancs + (flux if gauging else []))
        out = {}
        for k, (m, i) in enumerate((m, i) for m in mods for i in range(nh)):
            out[("Z", m, i)] = mz[k]
            out[("X", m, i)] = mx[k]
        if gauging:
            for p in range(qm.num_cycles):
                out[("B", p)] = mx[len(x_ancs) + p]
            b.tick()
            for layer in gauss_layers:
                b.cnots(layer)
                b.tick()
            ma = b.measure("Z", gauss)
            for k in range(qm.num_vertices):
                out[("A", k)] = ma[k]
        b.tick()
        return out

    # 1) initialisation
    b.reset("Z", data)
    b.tick()
    edge_meas: dict[int, int] = {}
    phase_rounds = [("before", False)] * rounds_before + [("gauging", True)] * rounds_gauging + \
                   [("after", False)] * rounds_after
    first_gauging = True
    ungauged = False
    seen_after: set = set()  # X checks already measured after ungauging
    for phase, gauging in phase_rounds:
        if gauging and first_gauging:
            b.reset("X", edges)
        if phase == "after" and not ungauged:
            edge_meas = dict(zip(range(qm.num_edges), b.measure("X", edges)))
            ungauged = True
            # flux checks against the final edge readout
            for p, cyc in enumerate(layout.cycles):
                b.detector([prev[("B", p)]] + [edge_meas[e] for e in cyc])
        out = syndrome_round(gauging)
        for key, m in out.items():
            kind = key[0]
            if kind == "Z":
                b.detector([m] + ([prev[key]] if key in prev else []))
            elif kind == "X":
                if key not in prev:
                    continue  # random in the first round (modules start in |0>)
                extra = []
                g = layout.gamma.get(key[1:], [])
                if g and phase == "after" and key not in seen_after:
                    # s(first round after) = s'(last gauging round) x prod_{e in gamma_s} X_e
                    extra = [edge_meas[e] for e in g]
                b.detector([m, prev[key]] + extra)
            elif kind == "B":
                b.detector([m] + ([prev[key]] if key in prev else []))
            elif kind == "A":
                if key in prev:
                    b.detector([m, prev[key]])
        if gauging and first_gauging:
            # Observable 0: the logical measurement outcome of Zbar_a Zbar_b
            b.observable(0, [out[("A", k)] for k in range(qm.num_vertices)])
            first_gauging = False
        if phase == "after":
            seen_after.update(k for k in out if k[0] == "X")
        prev.update(out)

    # 4) ungauging when there are no rounds after the gauging phase
    if not ungauged:
        edge_meas = dict(zip(range(qm.num_edges), b.measure("X", edges)))
        for p, cyc in enumerate(layout.cycles):
            b.detector([prev[("B", p)]] + [edge_meas[e] for e in cyc])

    # 5) final Z readout: Z-stabiliser detectors and logical Z observables
    final = dict(zip(data, b.measure("Z", data)))
    for m in mods:
        for c in range(nh):
            supp = np.nonzero(code.hz[c])[0]
            b.detector([prev[("Z", m, c)]] + [final[qm.data(m, int(i))] for i in supp])
    lz = logical_z_basis(code)
    obs = 1
    for m in mods:
        for row in lz:
            b.observable(obs, [final[qm.data(m, int(i))] for i in np.nonzero(row)[0]])
            obs += 1
    return b.c


# --------------------------------------------------------------------------------------
# Partitions for Chipmunq and public entry point
# --------------------------------------------------------------------------------------

BRIDGE_PARTITION_WIDTH = 24  # = chiplet width of the Gross backend (2 * l)


def bridge_partitions(layout: BridgeLayout, bridge_width: int = BRIDGE_PARTITION_WIDTH) -> list[dict]:
    """Chipmunq partitions: two ``gross_code`` modules and one ``grid`` partition for the bridge.

    The bridge qubits are ordered module a -> bridge edges -> module b, so that a row-major placement
    keeps the Gauss-law ancillas next to the edges they touch:
        [A_v (v in V_a), edges inside G_a, bridge edges, edges inside G_b, A_v (v in V_b), flux ancillas]
    """
    code = layout.code
    qm = QubitMap(code.n, len(layout.edges), len(layout.vertices), len(layout.cycles))
    parts = []
    for mod in ("a", "b"):
        off = qm.module_offset(mod)
        parts.append({
            "indices": list(range(off, off + 2 * code.n)),  # [X checks, L data, R data, Z checks]
            "width": code.l,
            "height": code.m,
            "distance": GROSS_DISTANCE,
            "type": "gross_code",
            "role": f"module_{mod}",
        })
    bridge_set = {frozenset(e) for e in layout.bridge_edges}
    side = lambda e: "bridge" if frozenset(e) in bridge_set else e[0][0]
    vpos = {v: k for k, v in enumerate(layout.vertices)}
    order = [qm.gauss_anc(vpos[v]) for v in layout.vertices if v[0] == "a"]
    for s in ("a", "bridge", "b"):
        order += [qm.edge(k) for k, e in enumerate(layout.edges) if side(e) == s]
    order += [qm.gauss_anc(vpos[v]) for v in layout.vertices if v[0] == "b"]
    order += [qm.flux_anc(p) for p in range(len(layout.cycles))]
    assert sorted(order) == list(range(qm.bridge_offset, qm.num_qubits))
    parts.append({
        "indices": order,
        "width": bridge_width,
        "height": math.ceil(len(order) / bridge_width),
        "distance": GROSS_DISTANCE,
        "type": "grid",
        "role": "bridge",
    })
    covered = sorted(q for p in parts for q in p["indices"])
    assert covered == list(range(qm.num_qubits)), "partitions must cover every qubit exactly once"
    return parts


def generate_gross_bridge(
    rounds_before: int = 1,
    rounds_gauging: int = GROSS_DISTANCE,
    rounds_after: int = 1,
    num_bridge_edges: int = GROSS_DISTANCE,
    expansion_degree: int = 0,
    seed: int = 0,
    noise: float = 0.0,
    check_code: bool = True,
    output_file: str | None = None,
) -> tuple[stim.Circuit, list[dict]]:
    """Two Gross-code modules with a bridge measuring Zbar_a (x) Zbar_b, plus Chipmunq partitions.

    :param num_bridge_edges: edges between the modules. Z on the bridge edges is equivalent to
        Zbar_a in the deformed code, so fewer than d = 12 bridge edges reduce the code distance to
        ``num_bridge_edges`` during the measurement. Keep the default unless that is intended.
    :param noise: circuit-level noise for testing (leave 0 if Chipmunq inserts noise after routing)
    :param check_code: verify that the deformed code is a valid CSS code with k = 23
    :return: (stim circuit, partitions); observable 0 is the outcome of Zbar_a Zbar_b,
        observables 1..24 are the logical Z operators of the two modules
    """
    code = gross_code()
    z = find_min_weight_logical_z(code, seed=seed)
    layout = build_bridge_graph(code, z, num_bridge_edges=num_bridge_edges,
                                expansion_degree=expansion_degree, seed=seed)
    if check_code:
        check_deformed_code(layout)
    circuit = build_bridge_circuit(layout, rounds_before=rounds_before, rounds_gauging=rounds_gauging,
                                   rounds_after=rounds_after, noise=noise)
    partitions = bridge_partitions(layout)
    if output_file is not None:
        with open(output_file, "w") as f:
            print(circuit, file=f)
    return circuit, partitions


def get_gross_bridge(**kwargs) -> tuple[stim.Circuit, list[dict]]:
    """Drop-in analogue of ``generate_gross_code`` for the experiment scripts."""
    return generate_gross_bridge(output_file="stim_gross_bridge.stim", **kwargs)