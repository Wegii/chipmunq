"""
Noise-adaptive mapping of Murali et al., "Noise-Adaptive Compiler Mappings for
Noisy Intermediate-Scale Quantum Computers", ASPLOS'19 — reimplemented for Qiskit.

    routed = noise_adaptive_transpilation(circuit, backend)                  # R-SMT*  (optimal)
    routed = noise_adaptive_transpilation(circuit, backend, method="greedy_e")  # GreedyE* (heuristic)

Model (as in the paper):
  * Static placement: every program qubit gets one physical qubit for the whole program.
  * A 2-qubit gate between non-adjacent qubits moves one operand along a path with SWAPs,
    executes the gate, and swaps back (the paper's duration model 2*(d-1)*t_SWAP + t_CNOT).
  * Objective (Eq. 12):  maximize  w * sum_readouts log(r_ro)  +  (1-w) * sum_2q-gates log(r_2q),
    where the reliability of a routed 2q gate is the product of the reliabilities of its
    SWAPs (3 CNOTs each) and of the final CNOT (footnote 3). Single-qubit errors are ignored.

Deliberate deviations (documented so you can state them in the paper):
  * Paths: instead of the grid-only "one bend paths", each routed gate uses the most reliable
    path on the actual coupling graph (Dijkstra on -log reliability, SWAPs weighted x3). This
    generalizes 1BP to arbitrary topologies (heavy-hex, chiplet graphs) and dominates it.
  * Timing: the coherence bound (Eq. 6) and time-overlap routing constraints (Eqs. 7-9) are
    omitted. They don't change the R-SMT* objective, and the paper reports the coherence bound
    was never binding. The emitted circuit is sequential and valid; parallelism comes from
    whatever scheduler runs afterwards.
  * count_return_swaps=False reproduces footnote 3 (only the outbound SWAPs count toward
    reliability). Set True to also charge the swap-back, which the emitted circuit does perform.
"""

import math
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from qiskit import QuantumCircuit
from qiskit.circuit import ControlFlowOp
from qiskit.circuit.library import SwapGate
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import Unroll3qOrMore

_MIN_REL = 1e-9  # clamp so log() stays finite


# ----------------------------------------------------------------------------- noise data
def _noise_from_backend(backend, cx_errors=None, readout_errors=None):
    """Return (n_phys, {undirected edge: 2q error}, [readout error per qubit])."""
    if isinstance(backend, CouplingMap):
        cmap, target = backend, None
    elif isinstance(backend, (list, tuple)):
        cmap, target = CouplingMap(couplinglist=[tuple(e) for e in backend]), None
    else:
        target = getattr(backend, "target", None)
        cmap = backend.coupling_map if getattr(backend, "coupling_map", None) is not None \
            else target.build_coupling_map()
        if not isinstance(cmap, CouplingMap):
            cmap = CouplingMap(couplinglist=cmap)
    n = cmap.size()

    edge_err = {}
    for a, b in cmap.get_edges():
        if a != b:
            edge_err.setdefault(tuple(sorted((a, b))), None)

    if target is not None:  # take the best native 2q gate on each edge
        for name in target.operation_names:
            op = target.operation_from_name(name)
            if getattr(op, "num_qubits", 0) != 2 or isinstance(op, type):
                continue
            for qargs, props in (target[name] or {}).items():
                if qargs is None or props is None or props.error is None:
                    continue
                e = tuple(sorted(qargs))
                if e in edge_err:
                    edge_err[e] = props.error if edge_err[e] is None else min(edge_err[e], props.error)
    if cx_errors:
        for (a, b), err in cx_errors.items():
            edge_err[tuple(sorted((a, b)))] = err

    ro_err = [None] * n
    if target is not None and "measure" in target.operation_names:
        for qargs, props in (target["measure"] or {}).items():
            if qargs is not None and props is not None and props.error is not None:
                ro_err[qargs[0]] = props.error
    if readout_errors:
        for q, err in readout_errors.items():
            ro_err[q] = err

    known = [v for v in edge_err.values() if v is not None]
    default_2q = float(np.mean(known)) if known else 0.01
    edge_err = {e: (default_2q if v is None else v) for e, v in edge_err.items()}
    known_ro = [v for v in ro_err if v is not None]
    default_ro = float(np.mean(known_ro)) if known_ro else 0.02
    ro_err = [default_ro if v is None else v for v in ro_err]
    return n, edge_err, ro_err


# ------------------------------------------------------------------ swap-aware path costs
def _routing_tables(n, edge_err, swap_factor):
    """cost[m, f] = -log reliability of moving qubit at m next to f (SWAPs) and doing the gate.
    route[(m, f)] = physical path [m, ..., nbr_of_f]; SWAPs along it, gate on (path[-1], f)."""
    w = {e: -math.log(max(1.0 - err, _MIN_REL)) for e, err in edge_err.items()}
    nbrs = defaultdict(list)
    for (a, b) in w:
        nbrs[a].append(b)
        nbrs[b].append(a)

    cost = np.full((n, n), np.inf)
    best = {}  # (m, f) -> (neighbor n_f, predecessor row, sources index)
    for f in range(n):
        if not nbrs[f]:
            continue
        rows, cols, vals = [], [], []
        for (a, b), wt in w.items():
            if f in (a, b):
                continue  # path may not pass through the fixed qubit
            rows += [a, b]; cols += [b, a]; vals += [swap_factor * wt] * 2
        g = csr_matrix((vals, (rows, cols)), shape=(n, n))
        srcs = nbrs[f]
        dist, pred = dijkstra(g, directed=True, indices=srcs, return_predecessors=True)
        for i, nf in enumerate(srcs):
            tot = dist[i] + w[tuple(sorted((nf, f)))]
            better = tot < cost[:, f]
            better[f] = False
            cost[better, f] = tot[better]
            for m in np.nonzero(better)[0]:
                best[(int(m), f)] = (nf, pred[i])
    route = {}

    def get_route(m, f):
        if (m, f) not in route:
            nf, pred = best[(m, f)]
            path = [m]
            while path[-1] != nf:  # walk predecessors from m toward nf (tree rooted at nf)
                path.append(int(pred[path[-1]]))
            route[(m, f)] = path
        return route[(m, f)]

    return cost, get_route


# ---------------------------------------------------------------------------- placement
def _rsmt_placement(n_prog, n_phys, pair_counts, ro_weight, pair_cost, ro_cost,
                    omega, solver, time_limit, verbose):
    """R-SMT* placement: exact minimization of -Eq.12 over static placements (a QAP).

    solver="milp": HiGHS via scipy, Adams-Johnson linearization (fast, same optimum).
    solver="z3":   Z3 Optimize, as in the paper (much slower beyond ~5 qubits on ~9 sites).
    """
    pairs = list(pair_counts.items())
    if solver == "z3":
        import z3
        S = 100_000
        b = [[z3.Bool(f"b_{q}_{h}") for h in range(n_phys)] for q in range(n_prog)]
        opt = z3.Optimize()
        if time_limit:
            opt.set("timeout", int(1000 * time_limit))
        for q in range(n_prog):
            opt.add(z3.PbEq([(x, 1) for x in b[q]], 1))
        for h in range(n_phys):
            opt.add(z3.PbLe([(b[q][h], 1) for q in range(n_prog)], 1))
        terms = [z3.If(b[q][h], int(round(S * omega * ro_weight[q] * ro_cost[h])), 0)
                 for q in range(n_prog) if ro_weight[q] for h in range(n_phys)]
        for i, ((p, r), cnt) in enumerate(pairs):
            cp = z3.Int(f"pair_{i}")
            opt.add(cp >= 0)
            terms.append(cp)
            for h1 in range(n_phys):
                row = []
                for h2 in range(n_phys):
                    if h2 == h1:
                        continue
                    if not np.isfinite(pair_cost[h1, h2]):
                        opt.add(z3.Not(z3.And(b[p][h1], b[r][h2])))
                        continue
                    row.append(z3.If(b[r][h2], int(round(S * (1 - omega) * cnt * pair_cost[h1, h2])), 0))
                opt.add(z3.Implies(b[p][h1], cp >= z3.Sum(row) if row else cp >= 0))
        obj = z3.Sum(terms) if terms else z3.IntVal(0)
        opt.minimize(obj)
        res = opt.check()
        m = opt.model() if res != z3.unsat else None
        if m is None:
            raise RuntimeError(f"R-SMT* (z3): {res}, no placement found")
        if verbose:
            print(f"R-SMT* (z3) status={res}, objective={m.eval(obj).as_long() / S:.5f}")
        return [next(h for h in range(n_phys) if z3.is_true(m.eval(b[q][h], model_completion=True)))
                for q in range(n_prog)]

    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import coo_matrix
    n, nb = n_phys, n_prog * n_phys
    N = nb + len(pairs) * n * n
    c, ub = np.zeros(N), np.ones(N)
    for q in range(n_prog):
        c[q * n:(q + 1) * n] = omega * ro_weight[q] * np.asarray(ro_cost)
    pc = np.where(np.isfinite(pair_cost), pair_cost, 0.0)
    feasible = np.isfinite(pair_cost) & ~np.eye(n, dtype=bool)
    for i, (_, cnt) in enumerate(pairs):
        s = nb + i * n * n
        c[s:s + n * n] = ((1 - omega) * cnt * pc).ravel()
        ub[s:s + n * n] = feasible.ravel()
    R, C, V, lo, hi = [], [], [], [], []
    r = 0
    for q in range(n_prog):                       # each program qubit placed once
        R += [r] * n; C += list(range(q * n, (q + 1) * n)); V += [1] * n
        lo.append(1); hi.append(1); r += 1
    for h in range(n):                            # each physical qubit used at most once
        R += [r] * n_prog; C += [q * n + h for q in range(n_prog)]; V += [1] * n_prog
        lo.append(0); hi.append(1); r += 1
    for i, ((p, q2), _) in enumerate(pairs):      # y[h1,h2] marginals = placements
        s = nb + i * n * n
        for h1 in range(n):
            R += [r] * (n + 1); C += [s + h1 * n + h2 for h2 in range(n)] + [p * n + h1]
            V += [1] * n + [-1]; lo.append(0); hi.append(0); r += 1
        for h2 in range(n):
            R += [r] * (n + 1); C += [s + h1 * n + h2 for h1 in range(n)] + [q2 * n + h2]
            V += [1] * n + [-1]; lo.append(0); hi.append(0); r += 1
    A = coo_matrix((V, (R, C)), shape=(r, N)).tocsr()
    integ = np.zeros(N); integ[:nb] = 1
    opts = {"disp": verbose}
    if time_limit:
        opts["time_limit"] = time_limit
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=integ,
               bounds=Bounds(0, ub), options=opts)
    if res.x is None:
        raise RuntimeError(f"R-SMT* (milp): {res.message}")
    if verbose or res.status != 0:
        print(f"R-SMT* (milp): {res.message} objective={res.fun:.5f}")
    x = res.x[:nb].reshape(n_prog, n)
    return [int(np.argmax(x[q])) for q in range(n_prog)]


def _greedy_e_placement(n_prog, n_phys, pair_counts, pair_cost, ro_cost, edge_err):
    """GreedyE*: place program edges in descending CNOT count (Sec. 5.2)."""
    layout, used = {}, set()
    edges = sorted(pair_counts.items(), key=lambda kv: -kv[1])
    hw_edges = sorted(edge_err, key=lambda e: -math.log(max(1 - edge_err[e], _MIN_REL))
                      + ro_cost[e[0]] + ro_cost[e[1]])  # best CNOT + readout first
    neigh = defaultdict(list)
    for (p, r), c in pair_counts.items():
        neigh[p].append((r, c)); neigh[r].append((p, c))

    def place_best(q):
        best_h, best_c = None, np.inf
        for h in range(n_phys):
            if h in used:
                continue
            c = sum(cnt * pair_cost[h, layout[o]] for o, cnt in neigh[q] if o in layout)
            if c < best_c:
                best_h, best_c = h, c
        if best_h is None or not np.isfinite(best_c):
            raise RuntimeError("GreedyE*: no reachable free qubit")
        layout[q] = best_h; used.add(best_h)

    while len(layout) < n_prog:
        frontier = [(e, c) for e, c in edges if (e[0] in layout) != (e[1] in layout)]
        if frontier:
            (p, r), _ = frontier[0]
            place_best(r if p in layout else p)
            continue
        fresh = [(e, c) for e, c in edges if e[0] not in layout and e[1] not in layout]
        if fresh:  # new connected component: seed it on the best free hardware edge
            (p, r), _ = fresh[0]
            for a, b in hw_edges:
                if a not in used and b not in used:
                    layout[p], layout[r] = a, b; used.update((a, b)); break
            else:
                raise RuntimeError("GreedyE*: no free hardware edge")
            continue
        for q in range(n_prog):  # qubits without 2q gates: best free readout
            if q not in layout:
                h = min((h for h in range(n_phys) if h not in used), key=lambda h: ro_cost[h])
                layout[q] = h; used.add(h)
    return [layout[q] for q in range(n_prog)]


# ------------------------------------------------------------------------------ main API
def noise_adaptive_transpilation(
    circuit: QuantumCircuit,
    backend,
    method: str = "rsmt",            # "rsmt" (R-SMT*, optimal) or "greedy_e" (GreedyE*, heuristic)
    omega: float = 0.5,              # readout weight in Eq. 12 (paper's best: 0.5)
    count_return_swaps: bool = False,
    cx_errors: dict = None,          # optional overrides {(a, b): err}
    readout_errors: dict = None,     # optional overrides {q: err}
    solver: str = "milp",            # R-SMT* backend: "milp" (HiGHS, fast) or "z3" (as in paper)
    time_limit: float = None,        # R-SMT*: seconds; returns best placement found so far
    verbose: bool = False,
) -> QuantumCircuit:
    """Place and route `circuit` on `backend` following Murali et al. (ASPLOS'19).

    backend: BackendV2 (errors read from backend.target), CouplingMap, or edge list
             (then pass cx_errors / readout_errors, else uniform defaults are used).
    Returns a QuantumCircuit on physical qubits with explicit SwapGates. The mapping is static,
    so initial == final layout; it's stored in circuit.metadata['noise_adaptive_layout'].
    """
    if method not in ("rsmt", "greedy_e"):
        raise ValueError("method must be 'rsmt' or 'greedy_e'")
    n_phys, edge_err, ro_err = _noise_from_backend(backend, cx_errors, readout_errors)
    circ = PassManager([Unroll3qOrMore()]).run(circuit)
    n_prog = circ.num_qubits
    if n_prog > n_phys:
        raise ValueError(f"circuit has {n_prog} qubits, backend only {n_phys}")
    qidx = {q: i for i, q in enumerate(circ.qubits)}

    pair_counts, ro_weight = Counter(), [0] * n_prog
    for inst in circ.data:
        if isinstance(inst.operation, ControlFlowOp):
            raise NotImplementedError("control-flow ops are not supported")
        qs = [qidx[q] for q in inst.qubits]
        if inst.operation.name == "measure":
            ro_weight[qs[0]] += 1
        elif len(qs) == 2 and inst.operation.name != "barrier":
            pair_counts[tuple(sorted(qs))] += 1
    if not any(ro_weight):  # no measurements in the input: assume every qubit is read out once
        ro_weight = [1] * n_prog

    swap_factor = 3 * (2 if count_return_swaps else 1)
    cost, get_route = _routing_tables(n_phys, edge_err, swap_factor)
    pair_cost = np.minimum(cost, cost.T)  # either operand may be the one that moves
    ro_cost = [-math.log(max(1 - e, _MIN_REL)) for e in ro_err]

    if method == "rsmt":
        layout = _rsmt_placement(n_prog, n_phys, pair_counts, ro_weight, pair_cost, ro_cost,
                                 omega, solver, time_limit, verbose)
    else:
        layout = _greedy_e_placement(n_prog, n_phys, pair_counts, pair_cost, ro_cost, edge_err)

    # ---- emit routed circuit
    out = QuantumCircuit(n_phys, name=f"{circuit.name}_noise_adaptive")
    for creg in circ.cregs:
        out.add_register(creg)
    reg_bits = {b for r in circ.cregs for b in r}
    loose = [c for c in circ.clbits if c not in reg_bits]
    if loose:
        out.add_bits(loose)

    n_swaps, log_rel = 0, 0.0
    for inst in circ.data:
        op = inst.operation
        phys = [layout[qidx[q]] for q in inst.qubits]
        if len(phys) != 2 or op.name == "barrier" or tuple(sorted(phys)) in edge_err:
            out.append(op, [out.qubits[p] for p in phys], list(inst.clbits))
            continue
        h1, h2 = phys
        mover, fixed = (h1, h2) if cost[h1, h2] <= cost[h2, h1] else (h2, h1)
        path = get_route(mover, fixed)
        for a, b in zip(path, path[1:]):
            out.append(SwapGate(), [a, b]); n_swaps += 1
        new = [path[-1] if p == mover else p for p in phys]
        out.append(op, [out.qubits[p] for p in new], list(inst.clbits))
        for a, b in reversed(list(zip(path, path[1:]))):
            out.append(SwapGate(), [a, b]); n_swaps += 1

    # estimated success probability of the emitted circuit under the paper's model
    for inst in out.data:
        qs = [out.find_bit(q).index for q in inst.qubits]
        if inst.operation.name == "swap":
            log_rel += 3 * math.log(max(1 - edge_err[tuple(sorted(qs))], _MIN_REL))
        elif len(qs) == 2 and inst.operation.name != "barrier":
            log_rel += math.log(max(1 - edge_err[tuple(sorted(qs))], _MIN_REL))
        elif inst.operation.name == "measure":
            log_rel += math.log(max(1 - ro_err[qs[0]], _MIN_REL))

    out.metadata = dict(circuit.metadata or {})
    out.metadata.update(
        noise_adaptive_method=method,
        noise_adaptive_layout=layout,          # virtual qubit i -> physical qubit (static)
        noise_adaptive_swap_count=n_swaps,
        noise_adaptive_est_success=math.exp(log_rel),
    )
    return out


if __name__ == "__main__":
    from qiskit.providers.fake_provider import GenericBackendV2

    # BV4 from the paper (Fig. 2a)
    qc = QuantumCircuit(4, 3)
    qc.x(3)
    qc.h(range(4))
    for i in range(3):
        qc.cx(i, 3)
    qc.h(range(3))
    qc.measure(range(3), range(3))

    backend = GenericBackendV2(num_qubits=16, coupling_map=CouplingMap.from_grid(2, 8).get_edges(),
                               seed=3)
    for m in ("rsmt", "greedy_e"):
        r = noise_adaptive_transpilation(qc, backend, method=m)
        print(m, r.metadata)