"""Clifford QFT on triangular 6.6.6 color codes with transversal gates, with Chipmunq partitions.

Logical circuit
---------------
The textbook QFT on n qubits (final qubit-reversal SWAPs omitted, as usual) is

    for j = 0 .. n-1:  H(j);  for k = j+1 .. n-1:  CP(pi / 2^(k-j)) between k and j

The controlled phases are not Clifford, and a fault-tolerant small-angle rotation needs T gates from
magic-state distillation, which neither our circuit generators nor Stim support. Every controlled phase
is therefore decomposed as

    CP(theta)(c, t) = Rz_c(theta/2) Rz_t(theta/2) CX(c, t) Rz_t(-theta/2) CX(c, t)

and every rotation is replaced by its nearest Clifford: Rz(+theta/2) -> S, Rz(-theta/2) -> S_DAG. The
result ("Clifford QFT") is not the QFT unitary, but it keeps the QFT's interaction structure, which is
what the compiler sees: all-to-all logical CNOTs, n(n-1) of them, in QFT order, interleaved with H and S
layers. ``max_distance`` (approximate QFT) drops the controlled phases between qubits more than
``max_distance`` apart, i.e. the smallest angles pi / 2^(k-j).

Transversal gates on the color code
-----------------------------------
The 6.6.6 color code has transversal H, CNOT and S. Physical S on every data qubit maps a weight-w face
X_f to i^w X_f Z_f, i.e. -X_f Z_f for weight-6 faces. The honeycomb data lattice is bipartite and every
face alternates between the two sublattices, so S on sublattice A and S_DAG on sublattice B maps
X_f -> X_f Z_f exactly for every face (logical S up to a Pauli frame), Z_f -> Z_f. Detector tracking:

    H on patch p:           X_f(p) <-> Z_f(p)
    CNOT(c -> t):           X_f(c) -> X_f(c) X_f(t),   Z_f(t) -> Z_f(c) Z_f(t)
    S / S_DAG on patch p:   X_f(p) -> X_f(p) Z_f(p)

Observables
-----------
All patches start in |0>_L and are read out transversally in X: like the real QFT, which maps |0...0> to
|+>^n, the Clifford QFT's output has no Z-type logical stabilizers but n independent X-type ones. The
observables are a basis of these X-type logical stabilizers of the output state (computed with stim on the
n-qubit logical circuit): each observable is a product of Xbar_i over a set of patches, n of them.

Scheduling: the controlled-phase blocks are diagonal and commute, so by default they run in parallel on
disjoint qubit pairs (``clifford_qft_schedule``: same unitary, O(n) steps instead of n(n-1)/2). Syndrome
rounds (``rounds_per_step``) follow the initialisation and every transversal CNOT layer; single-qubit
transversal gates (H, S) need none. ``parallel=False`` gives the sequential textbook order.

Layout and partitions: identical to ``bv_color_code_circuit_generator`` (one ``grid`` partition per patch,
flag qubits by default, chiplets with ``chiplet_long_range_offsets(flags)``).
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import stim

from experiments.exp_utils.bv_color_code_circuit_generator import (
    ColorCode,
    _Circuit,
    _greedy_layers,
    chiplet_long_range_offsets,  # noqa: F401  (re-exported for the experiment scripts)
    color_code,
)

ROLE_DATA = "data"


# --------------------------------------------------------------------------------------
# Logical circuit
# --------------------------------------------------------------------------------------


def clifford_qft_ops(n: int, max_distance: int | None = None) -> list[tuple]:
    """Logical gate list of the Clifford QFT: ("H", q), ("S", q), ("S_DAG", q), ("CX", c, t)."""
    if n < 2:
        raise ValueError("The QFT needs at least two qubits.")
    ops: list[tuple] = []
    for j in range(n):
        ops.append(("H", j))
        for k in range(j + 1, n):
            if max_distance is not None and k - j > max_distance:
                continue
            # CP(theta)(k, j) with Rz(+theta/2) -> S and Rz(-theta/2) -> S_DAG
            ops += [("S", k), ("S", j), ("CX", k, j), ("S_DAG", j), ("CX", k, j)]
    return ops


def clifford_qft_schedule(n: int, max_distance: int | None = None) -> list[dict]:
    """Parallel schedule of the Clifford QFT: list of steps {"h": [qubits], "blocks": [(k, j), ...]}.

    Every controlled-phase block S(k) S(j) CX(k,j) S_DAG(j) CX(k,j) is diagonal (CX (I x S_DAG) CX is a ZZ
    rotation), so blocks commute with each other; only the H gates fix the order. Block (k, j), k > j, must
    follow H(j) and precede H(k), and every qubit takes part in at most one block per step. The unitary is
    the same as the sequential ``clifford_qft_ops``; the depth drops from n(n-1)/2 blocks to O(n) steps.
    """
    if n < 2:
        raise ValueError("The QFT needs at least two qubits.")
    busy: list[set[int]] = []        # qubits used by blocks in every step
    h_at: dict[int, int] = {}        # step whose H layer applies H(q)
    last: dict[int, int] = {}        # last step in which qubit q is used
    blocks_at: list[list[tuple[int, int]]] = []

    def ensure(t: int) -> None:
        while len(busy) <= t:
            busy.append(set())
            blocks_at.append([])

    for j in range(n):
        # H(j) after every block on j so far (blocks (j, i), i < j); blocks of step t follow its H layer
        t_h = last.get(j, -1) + 1
        ensure(t_h)
        h_at[j] = t_h
        last[j] = t_h - 1
        for k in range(j + 1, n):
            if max_distance is not None and k - j > max_distance:
                continue
            t = max(t_h, last.get(k, -1) + 1, last[j] + 1)
            ensure(t)
            while j in busy[t] or k in busy[t]:
                t += 1
                ensure(t)
            busy[t].update((j, k))
            blocks_at[t].append((k, j))
            last[j] = max(last[j], t)
            last[k] = max(last.get(k, -1), t)
    steps = [{"h": sorted(q for q, t in h_at.items() if t == s), "blocks": blocks_at[s]} for s in range(len(busy))]
    return [st for st in steps if st["h"] or st["blocks"]]


def _schedule_ops(steps: list[dict]) -> list[tuple]:
    """Gate list of a schedule (for the logical tableau)."""
    ops: list[tuple] = []
    for st in steps:
        ops += [("H", q) for q in st["h"]]
        for k, j in st["blocks"]:
            ops += [("S", k), ("S", j), ("CX", k, j), ("S_DAG", j), ("CX", k, j)]
    return ops


def _x_type_logical_observables(n: int, ops: list[tuple]) -> list[list[int]]:
    """Basis of the X-type stabilizers of ops|0^n>, each as the list of logical qubits it acts on."""
    sim = stim.TableauSimulator()
    sim.set_num_qubits(n)
    for op in ops:
        if op[0] == "CX":
            sim.cx(op[1], op[2])
        else:
            getattr(sim, op[0].lower())(op[1])
    stabs = sim.canonical_stabilizers()
    xs = np.array([[p in (1, 2) for p in (s[q] for q in range(n))] for s in stabs], dtype=np.uint8)
    zs = np.array([[p in (2, 3) for p in (s[q] for q in range(n))] for s in stabs], dtype=np.uint8)
    # Combinations c of the generators with c . Z = 0 (mod 2): Gaussian elimination on [Z | I]
    m = np.concatenate([zs, np.eye(n, dtype=np.uint8)], axis=1)
    row = 0
    for col in range(n):
        piv = next((r for r in range(row, n) if m[r, col]), None)
        if piv is None:
            continue
        m[[row, piv]] = m[[piv, row]]
        for r in range(n):
            if r != row and m[r, col]:
                m[r] ^= m[row]
        row += 1
    out = []
    for r in range(row, n):  # rows with a zero Z part
        c = m[r, n:]
        x = (c @ xs) % 2
        out.append([q for q in range(n) if x[q]])
    return out


# --------------------------------------------------------------------------------------
# Transversal S: sublattice split of the honeycomb data lattice
# --------------------------------------------------------------------------------------


def data_sublattices(code: ColorCode) -> tuple[list[int], list[int]]:
    """Two-colouring (A, B) of the data qubits; every face has as many qubits in A as in B."""
    nbrs = {(0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (-1, 1)}
    pos = {code.coords[k]: k for k in range(code.n)}
    color = {}
    for start in range(code.n):
        if start in color:
            continue
        color[start] = 0
        dq = deque([start])
        while dq:
            k = dq.popleft()
            r, c = code.coords[k]
            for dr, dc in nbrs:
                other = pos.get((r + dr, c + dc))
                if other is None:
                    continue
                if other not in color:
                    color[other] = 1 - color[k]
                    dq.append(other)
                elif color[other] == color[k]:
                    raise AssertionError("Data lattice is not bipartite.")
    a = [k for k in range(code.n) if color[k] == 0]
    b = [k for k in range(code.n) if color[k] == 1]
    for f in code.faces:
        if 2 * len(set(f) & set(a)) != len(f):
            raise AssertionError("A face is not balanced between the sublattices.")
    return a, b


# --------------------------------------------------------------------------------------
# Circuit
# --------------------------------------------------------------------------------------


def build_qft_color_code_circuit(n: int, distance: int = 3, rounds_per_step: int = 1, flags: bool = True,
                                 max_distance: int | None = None, noise: float = 0.0, parallel: bool = True,
                                 patch_spacing: int = 2) -> tuple[stim.Circuit, ColorCode, list[list[int]]]:
    """Stim circuit of the transversal Clifford QFT on n color-code patches (see module docstring).

    :param parallel: run commuting controlled-phase blocks on disjoint qubit pairs in the same step (same
        unitary, O(n) instead of n(n-1)/2 steps); syndrome rounds then follow every transversal CNOT layer
        instead of every CNOT. ``False`` = the sequential textbook order.
    :return: (circuit, code, observables as lists of logical qubits)
    """
    if rounds_per_step < 1:
        raise ValueError("rounds_per_step must be >= 1.")
    code = color_code(distance)
    if parallel:
        steps = clifford_qft_schedule(n, max_distance)
    else:  # one block per step, textbook order
        steps = []
        seq = clifford_qft_ops(n, max_distance)
        i = 0
        while i < len(seq):
            if seq[i][0] == "H":
                steps.append({"h": [seq[i][1]], "blocks": []})
                i += 1
            else:
                steps.append({"h": [], "blocks": [(seq[i + 2][1], seq[i + 2][2])]})
                i += 5
    ops = _schedule_ops(steps)
    patches = list(range(n))
    qp = code.qubits_per_patch(flags)
    pcoords = code.patch_coords(flags)
    pwidth = code.width(flags)
    sub_a, sub_b = data_sublattices(code)

    def dq(p: int, k: int) -> int:
        return p * qp + k

    def fq(p: int, f: int) -> int:
        return p * qp + code.n + f

    def flq(p: int, f: int) -> int:
        return p * qp + code.n + code.num_faces + f

    b = _Circuit(noise)
    side = math.ceil(math.sqrt(n))
    for p in patches:
        for k, (r, c) in enumerate(pcoords):
            b.c.append("QUBIT_COORDS", [p * qp + k], [c + (p % side) * (pwidth + patch_spacing),
                                                       r + (p // side) * (code.height(flags) + patch_spacing)])

    # CNOT schedule of one syndrome sub-round (same as the BV color-code generator)
    face_seq = []
    for f, order in enumerate(code.face_order()):
        seq = [("d", order[0])] + ([("f", f)] if flags else []) + [("d", k) for k in order[1:-1]]
        seq += ([("f", f)] if flags else []) + [("d", order[-1])]
        face_seq.append(seq)

    def sub_round_layers(basis: str) -> list[list[tuple[int, int]]]:
        layer_ops = []
        for step in range(max(len(q) for q in face_seq)):
            for p in patches:
                for f, seq in enumerate(face_seq):
                    if step < len(seq):
                        kind, k = seq[step]
                        other = dq(p, k) if kind == "d" else flq(p, k)
                        layer_ops.append((fq(p, f), other) if basis == "X" else (other, fq(p, f)))
        return _greedy_layers(layer_ops)

    round_layers = {basis: sub_round_layers(basis) for basis in ("X", "Z")}
    ref: dict[tuple[str, int, int], set[int] | None] = {}
    rnd = [0]

    def data_qubits(p: int) -> list[int]:
        return [dq(p, k) for k in range(code.n)]

    def syndrome_round() -> None:
        faces_all = [fq(p, f) for p in patches for f in range(code.num_faces)]
        flags_all = [flq(p, f) for p in patches for f in range(code.num_faces)] if flags else []
        for basis in ("X", "Z"):
            flag_basis = "Z" if basis == "X" else "X"
            b.gate1("RX" if basis == "X" else "R", faces_all)
            b.gate1("R" if flag_basis == "Z" else "RX", flags_all)
            b.tick()
            for layer in round_layers[basis]:
                b.cx(layer)
                b.tick()
            meas = b.measure(basis, faces_all)
            if flags:
                for i, m in enumerate(b.measure(flag_basis, flags_all)):
                    p, f = divmod(i, code.num_faces)
                    b.detector({m}, [p, f, rnd[0], 2])
            for i, (p, f) in enumerate((p, f) for p in patches for f in range(code.num_faces)):
                key = (basis, p, f)
                if ref.get(key) is not None:
                    b.detector(ref[key] ^ {meas[i]}, [p, f, rnd[0], 0 if basis == "X" else 1])
                ref[key] = {meas[i]}
            b.tick()
        rnd[0] += 1

    # Transversal gate layers (several patches at once, then one TICK)
    def transversal_h(ps: list[int]) -> None:
        b.gate1("H", [q for p in ps for q in data_qubits(p)])
        b.tick()
        for p in ps:
            for f in range(code.num_faces):
                ref[("X", p, f)], ref[("Z", p, f)] = ref[("Z", p, f)], ref[("X", p, f)]

    def transversal_s(ps_s: list[int], ps_sdg: list[int]) -> None:
        b.gate1("S", [dq(p, k) for p in ps_s for k in sub_a] + [dq(p, k) for p in ps_sdg for k in sub_b])
        b.gate1("S_DAG", [dq(p, k) for p in ps_s for k in sub_b] + [dq(p, k) for p in ps_sdg for k in sub_a])
        b.tick()
        for p in ps_s + ps_sdg:
            for f in range(code.num_faces):
                x, z = ref[("X", p, f)], ref[("Z", p, f)]
                ref[("X", p, f)] = None if x is None or z is None else x ^ z

    def transversal_cnot(pairs: list[tuple[int, int]]) -> None:
        b.cx([(dq(c, k), dq(t, k)) for c, t in pairs for k in range(code.n)])
        b.tick()
        for ctrl, tgt in pairs:
            for f in range(code.num_faces):
                xc, xt = ref[("X", ctrl, f)], ref[("X", tgt, f)]
                zc, zt = ref[("Z", ctrl, f)], ref[("Z", tgt, f)]
                ref[("X", ctrl, f)] = None if xc is None or xt is None else xc ^ xt
                ref[("Z", tgt, f)] = None if zc is None or zt is None else zc ^ zt

    # 1) initialisation: every patch in |0>_L
    b.gate1("R", [q for p in patches for q in data_qubits(p)])
    b.tick()
    for p in patches:
        for f in range(code.num_faces):
            ref[("Z", p, f)] = set()
            ref[("X", p, f)] = None
    for _ in range(rounds_per_step):
        syndrome_round()

    # 2) Clifford QFT, step by step: H layer, then the step's controlled-phase blocks in parallel
    #    S(k) S(j) | CX(k, j) | S_DAG(j) | CX(k, j), with syndrome rounds after every CNOT layer
    for st in steps:
        if st["h"]:
            transversal_h(st["h"])
        if st["blocks"]:
            pairs = st["blocks"]
            transversal_s([q for k, j in pairs for q in (k, j)], [])
            for layer in range(2):
                transversal_cnot(pairs)
                for _ in range(rounds_per_step):
                    syndrome_round()
                if layer == 0:
                    transversal_s([], [j for _, j in pairs])

    # 3) readout: every patch transversally in X
    final = {}
    for p in patches:
        m = b.measure("X", data_qubits(p))
        final[p] = m
        for f, supp in enumerate(code.faces):
            r = ref[("X", p, f)]
            if r is not None:
                b.detector(r ^ {m[k] for k in supp}, [p, f, rnd[0], 0])
    observables = _x_type_logical_observables(n, ops)
    if len(observables) != n:
        raise AssertionError(f"Expected {n} X-type logical observables, found {len(observables)}.")
    for i, qs in enumerate(observables):
        recs: set[int] = set()
        for q in qs:
            recs ^= set(final[q])
        b.observable(i, sorted(recs))
    return b.c, code, observables


def qft_color_code_partitions(n: int, code: ColorCode, flags: bool = True) -> list[dict]:
    """One ``grid`` partition per patch, with local (row, col) coordinates for the placement."""
    qp = code.qubits_per_patch(flags)
    return [{
        "indices": list(range(p * qp, (p + 1) * qp)),
        "coords": code.patch_coords(flags),
        "width": code.width(flags),
        "height": code.height(flags),
        "distance": code.distance,
        "type": "grid",
        "role": ROLE_DATA,
        "logical_qubit": p,
    } for p in range(n)]


def generate_qft_color_code(n: int, distance: int = 3, rounds_per_step: int = 1, flags: bool = True,
                            max_distance: int | None = None, noise: float = 0.0, parallel: bool = True,
                            output_file: str | None = None) -> tuple[stim.Circuit, list[dict]]:
    """Generate the transversal Clifford-QFT circuit and its Chipmunq partitions.

    :param n: number of logical qubits (one color-code patch each)
    :param max_distance: approximate QFT; drop controlled phases between qubits more than this apart
    :return: (stim circuit, partitions)
    """
    circuit, code, observables = build_qft_color_code_circuit(
        n, distance=distance, rounds_per_step=rounds_per_step, flags=flags, max_distance=max_distance,
        noise=noise, parallel=parallel)
    partitions = qft_color_code_partitions(n, code, flags=flags)
    if output_file is not None:
        with open(output_file, "w") as f:
            print(circuit, file=f)
    return circuit, partitions


def get_color_code_qft(n: int, distance: int = 3, **kwargs) -> tuple[stim.Circuit, list[dict]]:
    """Drop-in analogue of ``get_color_code_bv`` for the experiment scripts."""
    md = kwargs.get("max_distance")
    tag = f"_aqft{md}" if md is not None else ""
    return generate_qft_color_code(n, distance=distance,
                                   output_file=f"stim_qft_color_n_{n}{tag}_d_{distance}.stim", **kwargs)