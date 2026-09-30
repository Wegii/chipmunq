"""
Ablation: correlated chiplet-local (cosmic-ray-like, CRL) noise on a distributed lattice-surgery CNOT.

Motivation: Wu et al., PR Applied 24, 044022 (2025) [R1]. CRL events suppress T1 on all qubits of the
struck module (85-94 % intra-module coincidence), are nearly confined to it (~2 % inter-module), and last
~2-6 ms, i.e. thousands of QEC cycles.

Workload: the same logical CNOT as experiment_distributed_lattice_surgery (tqec, control/ancilla/target
patches + merge strips, predefined partitions), on the same BackendChipletV2 sizes, so each patch fits on
one chiplet. Chipmunq places one patch per chiplet; LightSABRE (TrivialLayout + SabreSwap) packs them.

Noise = modsi1000 (incl. inter-chiplet links) + CRL bursts:
  * An event hits a BFS cluster of k qubits inside ONE chiplet for D ticks; each cluster qubit gets
    DEPOLARIZE1(p_b) every tick of the window (Pauli-twirled T1 collapse). k=None -> whole chiplet.
  * With probability rho it spills into one other chiplet (R1: ~2 %).
  * Stationary Poisson process whose start rate keeps the expected extra error per hardware qubit per
    tick equal to `delta` for every (k, D). Only the correlation structure changes.
  * Duration is wall-clock (ticks), converted from rounds with the ticks/round of the *uncompiled*
    reference circuit, so SWAP-heavy schedules are correctly exposed for longer.
  * Events only matter on chiplets the circuit touches, so sampling is restricted to those ("thinning");
    this is exact up to the O(rho) case of a spill from an unused chiplet into a used one.
  * i.i.d. control: DEPOLARIZE1(delta) on every qubit every tick (same average, no correlation).
Decoder: pymatching from the i.i.d.-control DEM (calibrated to the average, unaware of bursts).
Estimator: P_L = e^-Lambda P_L(no event) + (1 - e^-Lambda) E[P_L | >=1 event].

Parallelisation: the circuits are compiled in parallel (one task per distance x compiler), then the
sampling is split into small independent tasks -- chunks of event configurations, chunks of baseline
shots -- that all run at once on --jobs worker processes. Every worker receives the compiled circuits
once (pool initializer) and builds each decoder once. The no-event baseline P_L(no event) depends only
on the circuit, so it is sampled once per (d, compiler) and shared by all (k, D) points.

Sampling budget: defaults below (~10x fewer event shots and 5x fewer baseline shots than the original
200 x 5000 / 500k); override with --configs, --shots-per-config and --shots-base.

Run from the repo root:  python experiments/ablation/experiment_cosmic_ray_impact.py [--toy] [--jobs N]
"""
from __future__ import annotations

import argparse
import math
import multiprocessing
import os
import pickle
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

sys.path.append(os.path.join(os.getcwd(), "."))

import matplotlib.pyplot as plt
import numpy as np
import pymatching
import stim
from matplotlib.lines import Line2D

from experiments.exp_utils.circuit_generator import get_tqec_cnot_rotated
from experiments.exp_utils.circuit_noise import get_noise_model
from experiments.exp_utils.simulation_utils import transpile_stim_circuit
from experiments.exp_utils.transpilation_utils import sabre_transpilation
from experiments.exp_utils.utils import FONTSIZE, HEIGHT_FIGSIZE, WIDTH_FIGSIZE
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit
from glue.qiskit_qec.stim_tools import get_stim_circuits_with_detectors
from qeccm.backends.BackendChipletV2 import BackendChipletV2

# ======================================================================================
# Experiment parameters
# ======================================================================================
DISTANCE_SCALES = [2, 3]                 # d = 2k+1 -> d = 5, 7
COMPILERS = ["compiled", "sabre"]        # naming as in experiment_distributed_lattice_surgery
LABELS = {"compiled": "Chipmunq", "sabre": "LightSABRE"}
K_VALUES = [1, 4, 16, 64, None]          # None = whole chiplet (the R1-realistic footprint)
D_ROUNDS = [1, "d", "long"]              # "long" = ms-scale event, outlasts the whole CNOT
LONG_ROUNDS = 1000
P_PHYS = 1e-3                            # modsi1000 strength
P_INTER = 1e-3                           # inter-chiplet link noise
DELTA = 0.1 * P_PHYS                     # extra avg error / qubit / tick, identical everywhere
P_BURST = 0.1                            # per-tick depolarizing strength inside an event
RHO = 0.02                               # inter-chiplet spill-over (R1)
# Sampling budget (reduced; was 200 configs x 5000 shots and 500k baseline shots)
N_CONFIGS, SHOTS_PER_CONFIG, SHOTS_BASE = 50, 2_000, 100_000
CONFIGS_PER_TASK = 5                     # event configurations per parallel task
SHOTS_PER_TASK = 20_000                  # baseline / i.i.d. shots per parallel task
OUT_DIR = Path("experiments/evaluation/ablation")

# Same backends as experiment_distributed_lattice_surgery: patch (h x w) fits in one chiplet
BACKEND_CFG = {1: ((6, 6, 11, 6), 5), 2: ((6, 6, 15, 8), 7), 3: ((6, 6, 19, 10), 9), 4: ((6, 6, 23, 12), 11)}


# ======================================================================================
# Hardware view used by the noise model
# ======================================================================================
@dataclass
class Hardware:
    chiplet_of: dict[int, int]                  # every backend qubit -> chiplet
    edges: list[tuple[int, int]]                # coupling edges (inter-chiplet ones are ignored)
    active_chiplets: list[int] | None = None    # chiplets the circuit touches (None = all)
    adj: dict[int, list[int]] = field(init=False)
    chiplets: dict[int, list[int]] = field(init=False)

    def __post_init__(self):
        self.adj = {q: [] for q in self.chiplet_of}
        for a, b in self.edges:
            if self.chiplet_of[a] == self.chiplet_of[b]:
                self.adj[a].append(b)
                self.adj[b].append(a)
        self.chiplets = {}
        for q, c in self.chiplet_of.items():
            self.chiplets.setdefault(c, []).append(q)
        self._all = sorted(self.chiplets)
        self._act = sorted(self.active_chiplets) if self.active_chiplets is not None else self._all

    def _probs(self, ids):
        s = np.array([len(self.chiplets[c]) for c in ids], float)   # hit prob ∝ area ∝ #qubits
        return s / s.sum()

    @property
    def n_active(self) -> int:
        return sum(len(self.chiplets[c]) for c in self._act)

    def k_eff(self, k):
        sizes = np.array([len(self.chiplets[c]) for c in self._act])
        return float(((sizes if k is None else np.minimum(k, sizes)) * self._probs(self._act)).sum())

    def random_active_chiplet(self, rng):
        return int(rng.choice(self._act, p=self._probs(self._act)))

    def random_other_chiplet(self, rng, exclude):
        ids = [c for c in self._all if c != exclude]
        return int(rng.choice(ids, p=self._probs(ids)))

    def cluster(self, chiplet, k, rng) -> frozenset[int]:
        members = self.chiplets[chiplet]
        if k is None or k >= len(members):
            return frozenset(members)
        seed = int(rng.choice(members))
        seen, out, dq = {seed}, [], deque([seed])
        while dq and len(out) < k:
            q = dq.popleft()
            out.append(q)
            nbrs = list(self.adj[q])
            rng.shuffle(nbrs)
            for nb in nbrs:
                if nb not in seen:
                    seen.add(nb)
                    dq.append(nb)
        return frozenset(out)


# ======================================================================================
# Circuit utilities
# ======================================================================================
_ANNOT = {"TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"}


def _qubits_of(inst):
    return [t.value for t in inst.targets_copy() if t.is_qubit_target]


def num_layers(circ):
    return circ.flattened().num_ticks + 1


def used_qubits(circ):
    return {q for inst in circ.flattened() for q in _qubits_of(inst)}


def active_qubits(circ):
    """Qubits acted on by real operations (excludes pure noise channels / annotations)."""
    qs = set()
    for inst in circ.flattened():
        if inst.name in _ANNOT:
            continue
        gd = stim.gate_data(inst.name)
        if gd.is_noisy_gate and not gd.produces_measurements:
            continue
        qs.update(_qubits_of(inst))
    return qs


def ticks_per_round(circ):
    """Tick layers per syndrome round (layers / layers containing measurements)."""
    layers, has_m = 1, 0
    cur_m = False
    for inst in circ.flattened():
        if inst.name == "TICK":
            layers += 1
            has_m += cur_m
            cur_m = False
        elif inst.name not in _ANNOT and stim.gate_data(inst.name).produces_measurements:
            cur_m = True
    has_m += cur_m
    return layers / max(has_m, 1)


def inject_layer_noise(circ, noise_at):
    out = stim.Circuit()

    def emit(layer):
        for p, qs in noise_at(layer):
            if qs and p > 0:
                out.append("DEPOLARIZE1", sorted(qs), min(p, 0.75))

    layer = 0
    emit(0)
    for inst in circ.flattened():
        out.append(inst)
        if inst.name == "TICK":
            layer += 1
            emit(layer)
    return out


def iid_control(circ, delta):
    qs = used_qubits(circ)
    return inject_layer_noise(circ, lambda _l: [(delta, qs)])


def event_circuit(circ, events, p_b, used):
    return inject_layer_noise(
        circ, lambda layer: [(p_b, qs & used) for qs, t0, t1 in events if t0 <= layer < t1])


# ======================================================================================
# Sampling / decoding
# ======================================================================================
def _matcher(circ):
    dem = circ.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
    return pymatching.Matching.from_detector_error_model(dem)


def _failures(circ, matcher, shots, rng):
    det, obs = circ.compile_detector_sampler(seed=int(rng.integers(2**63))).sample(
        shots, separate_observables=True)
    return int(np.any(matcher.decode_batch(det) != obs, axis=1).sum()), shots


def _sample_ztp(lam, rng):
    if lam > 10:
        while (n := int(rng.poisson(lam))) == 0:
            pass
        return n
    u = rng.uniform(math.exp(-lam), 1.0)
    k, p = 0, math.exp(-lam)
    c = p
    while c < u:
        k += 1
        p *= lam / k
        c += p
    return max(k, 1)


def estimate_ler(base, hw, *, k, D_ticks, p_b, delta, rho, n_configs, shots_per_config,
                 shots_base, seed=0):
    rng = np.random.default_rng(seed)
    T = num_layers(base)
    used = used_qubits(base)
    matcher = _matcher(iid_control(base, delta))

    start_rate = delta * hw.n_active / (hw.k_eff(k) * D_ticks * p_b * (1 + rho))
    lam = start_rate * (T + D_ticks - 1)
    p_none = math.exp(-lam)

    f0, n0 = _failures(base, matcher, shots_base, rng) if p_none > 1e-6 else (0, 1)
    P0 = f0 / n0

    rates = []
    for _ in range(n_configs):
        events = []
        for _e in range(_sample_ztp(lam, rng)):
            t0 = int(rng.integers(-(D_ticks - 1), T))
            win = (max(t0, 0), min(t0 + D_ticks, T))
            c = hw.random_active_chiplet(rng)
            events.append((hw.cluster(c, k, rng), *win))
            if rng.random() < rho:
                events.append((hw.cluster(hw.random_other_chiplet(rng, c), k, rng), *win))
        f, n = _failures(event_circuit(base, events, p_b, used), matcher, shots_per_config, rng)
        rates.append(f / n)
    rates = np.array(rates)
    P1 = rates.mean()
    P = p_none * P0 + (1 - p_none) * P1
    se = math.sqrt(p_none**2 * P0 * (1 - P0) / n0 + (1 - p_none) ** 2 * rates.var(ddof=1) / n_configs)
    return dict(P_L=P, se=se, Lambda=lam, P_base=P0, P_given_event=P1, T_layers=T)


def estimate_iid(base, delta, shots, seed=0):
    rng = np.random.default_rng(seed)
    ctrl = iid_control(base, delta)
    f, n = _failures(ctrl, _matcher(ctrl), shots, rng)
    return dict(P_L=f / n, se=math.sqrt((f / n) * (1 - f / n) / n))


# ======================================================================================
# Chipmunq pipeline: backend, compilation, noise
# ======================================================================================
def get_backend(k):
    size, nic = BACKEND_CFG[k]
    return BackendChipletV2(size=size, n_inter=nic, connectivity="nn", topology="rotated_grid",
                            inter_chiplet_noise=P_INTER, inter_chiplet_amplification=1,
                            inter_chiplet_noise_type="constant", num_defective_qubits=0)


def compile_cnot(k, t, backend):
    """Returns (noisy stim circuit on physical qubits, reference ticks/round, patch->chiplets)."""
    circuit, partitions = get_tqec_cnot_rotated(distance_scale=k, n1=1, n2=0)
    ref = get_stim_circuits_with_detectors(StimCodeCircuit(circuit).qc)[0][0]
    if t == "compiled":
        _, qc, _, _ = transpile_stim_circuit(circuit, backend, pre_defined_partitions=partitions,
                                             routing_type="cost", routing_alpha=0, routing_beta=0)
    else:
        qc = sabre_transpilation(StimCodeCircuit(stim_circuit=circuit).qc, backend)
    lay = qc.layout.initial_index_layout(filter_ancillas=True)
    patch_chiplets = [sorted({backend.node_to_chiplet[lay[i]] for i in p["indices"]}) for p in partitions]
    phys = get_stim_circuits_with_detectors(qc)[0][0]
    noisy = get_noise_model("modsi1000", None, P_PHYS, None,
                            remote=backend.inter_chiplet_connections).noisy_circuit(phys)
    return noisy, ticks_per_round(ref), patch_chiplets


def hardware_for(backend, circ):
    act = {backend.node_to_chiplet[q] for q in active_qubits(circ) if q in backend.node_to_chiplet}
    edges = [tuple(map(int, e)) for e in backend.coupling_map.get_edges()]
    return Hardware(chiplet_of=dict(backend.node_to_chiplet), edges=edges, active_chiplets=sorted(act))


# ======================================================================================
# Sweep (parallel)
# ======================================================================================
def _compile_task(task):
    """Compile one (distance, compiler) circuit; runs in a worker. The tqec circuit is generated once per
    distance in the main process (tqec's detector cache is not safe for concurrent writers)."""
    k, t, circuit_str, partitions = task
    t0 = time.time()
    circuit = stim.Circuit(circuit_str)
    backend = get_backend(k)
    ref = get_stim_circuits_with_detectors(StimCodeCircuit(circuit).qc)[0][0]
    if t == "compiled":
        _, qc, _, _ = transpile_stim_circuit(circuit, backend, pre_defined_partitions=partitions,
                                             routing_type="cost", routing_alpha=0, routing_beta=0)
    else:
        qc = sabre_transpilation(StimCodeCircuit(stim_circuit=circuit).qc, backend)
    lay = qc.layout.initial_index_layout(filter_ancillas=True)
    patch_chiplets = [sorted({backend.node_to_chiplet[lay[i]] for i in p["indices"]}) for p in partitions]
    phys = get_stim_circuits_with_detectors(qc)[0][0]
    noisy = get_noise_model("modsi1000", None, P_PHYS, None,
                            remote=backend.inter_chiplet_connections).noisy_circuit(phys)
    hw = hardware_for(backend, noisy)
    return dict(k=k, compiler=t, circ=str(noisy), hw=hw, tpr=ticks_per_round(ref),
                patch_chiplets=patch_chiplets, elapsed=time.time() - t0)


# Per-worker state: compiled circuits (sent once by the pool initializer) and lazily built decoders.
_CIRCUITS: dict = {}
_CACHE: dict = {}


def _init_sampler(circuits):
    global _CIRCUITS, _CACHE
    _CIRCUITS, _CACHE = circuits, {}


def _prepared(key):
    """(base circuit, i.i.d.-control circuit, matcher, hardware) of a (d, compiler), built once per worker."""
    if key not in _CACHE:
        circ_str, hw = _CIRCUITS[key]
        base = stim.Circuit(circ_str)
        ctrl = iid_control(base, DELTA)
        _CACHE[key] = (base, ctrl, _matcher(ctrl), hw)
    return _CACHE[key]


def _event_rate(hw, k, D_ticks, T):
    """Poisson rate of event starts in the window (identical to the serial estimate_ler)."""
    start_rate = DELTA * hw.n_active / (hw.k_eff(k) * D_ticks * P_BURST * (1 + RHO))
    return start_rate * (T + D_ticks - 1)


def _sample_task(task):
    """One small sampling task. Returns failures / shots (baseline, i.i.d.) or per-config rates (events)."""
    kind, key, spec, seed = task
    rng = np.random.default_rng(seed)
    base, ctrl, matcher, hw = _prepared(key)
    if kind == "base":
        return dict(failures=_failures(base, matcher, spec["shots"], rng))
    if kind == "iid":
        return dict(failures=_failures(ctrl, matcher, spec["shots"], rng))
    # kind == "events": `n` event configurations for one (k, D) point, conditioned on >= 1 event
    T = num_layers(base)
    used = used_qubits(base)
    lam = _event_rate(hw, spec["k"], spec["D_ticks"], T)
    rates = []
    for _ in range(spec["n"]):
        events = []
        for _e in range(_sample_ztp(lam, rng)):
            t0 = int(rng.integers(-(spec["D_ticks"] - 1), T))
            win = (max(t0, 0), min(t0 + spec["D_ticks"], T))
            c = hw.random_active_chiplet(rng)
            events.append((hw.cluster(c, spec["k"], rng), *win))
            if rng.random() < RHO:
                events.append((hw.cluster(hw.random_other_chiplet(rng, c), spec["k"], rng), *win))
        f, n = _failures(event_circuit(base, events, P_BURST, used), matcher, spec["shots"], rng)
        rates.append(f / n)
    return dict(rates=rates)


def _chunks(total, size):
    out = []
    while total > 0:
        out.append(min(size, total))
        total -= out[-1]
    return out


def run_exp_cosmic_ray_impact(reproduce=True, toy=False, jobs=None,
                              n_configs=None, shots_per_config=None, shots_base=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pkl = OUT_DIR / f"cosmic_ray_impact{'_toy' if toy else ''}.pkl"
    scales = [1] if toy else DISTANCE_SCALES
    budget = dict(n_configs=20, shots_per_config=1_000, shots_base=20_000) if toy else \
        dict(n_configs=N_CONFIGS, shots_per_config=SHOTS_PER_CONFIG, shots_base=SHOTS_BASE)
    # Command-line overrides of the sampling budget
    for name, val in (("n_configs", n_configs), ("shots_per_config", shots_per_config),
                      ("shots_base", shots_base)):
        if val is not None:
            budget[name] = val
    if budget["n_configs"] < 2:
        raise ValueError("n_configs must be >= 2 (the standard error uses the sample variance)")
    jobs = jobs or os.cpu_count() or 1

    if reproduce:
        t_start = time.time()
        print(f"Budget: {budget['n_configs']} configs x {budget['shots_per_config']} shots per (k, D) point, "
              f"{budget['shots_base']} baseline / i.i.d. shots per circuit", flush=True)
        # One thread per worker: N workers each spawning a full Rust/BLAS thread pool oversubscribes the CPU
        for var in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        ctx = multiprocessing.get_context("spawn")  # Qiskit's Rust code is not fork-safe

        # --- 1) tqec circuits (serial, cheap) and compilation (parallel) ---------------------------
        compile_tasks = []
        for k in scales:
            circuit, partitions = get_tqec_cnot_rotated(distance_scale=k, n1=1, n2=0)
            compile_tasks += [(k, t, str(circuit), partitions) for t in COMPILERS]
        with ctx.Pool(min(jobs, len(compile_tasks))) as pool:
            compiled = pool.map(_compile_task, compile_tasks, chunksize=1)

        circuits, layouts, info = {}, {}, {}
        for c in compiled:
            d, t = 2 * c["k"] + 1, c["compiler"]
            base = stim.Circuit(c["circ"])
            circuits[(d, t)] = (c["circ"], c["hw"])
            info[(d, t)] = dict(T=num_layers(base), tpr=c["tpr"], hw=c["hw"])
            layouts[(d, t)] = dict(patch_chiplets=c["patch_chiplets"], active_chiplets=c["hw"]._act,
                                   ticks=num_layers(base), ticks_per_round_ref=c["tpr"])
            print(f"[layout] d={d} {LABELS[t]}: patch->chiplets {c['patch_chiplets']}, "
                  f"active chiplets {c['hw']._act}, {num_layers(base)} ticks ({c['elapsed']:.0f} s)", flush=True)

        # --- 2) all sampling tasks at once --------------------------------------------------------
        seeds = iter(np.random.SeedSequence(2025).generate_state(100_000, dtype=np.uint64))
        tasks, points = [], []
        for (d, t), inf in info.items():
            need_base = False
            for D in D_ROUNDS:
                rounds = {"d": d, "long": LONG_ROUNDS}.get(D, D)
                D_ticks = max(1, round(rounds * inf["tpr"]))
                for kk in K_VALUES:
                    lam = _event_rate(inf["hw"], kk, D_ticks, inf["T"])
                    points.append(dict(d=d, compiler=t, k=kk if kk else "chiplet", D=D, kk=kk,
                                       D_ticks=D_ticks, Lambda=lam, T_layers=inf["T"]))
                    need_base |= math.exp(-lam) > 1e-6
                    for n in _chunks(budget["n_configs"], CONFIGS_PER_TASK):
                        tasks.append(("events", (d, t), dict(k=kk, D_ticks=D_ticks, n=n,
                                                             shots=budget["shots_per_config"],
                                                             point=len(points) - 1), int(next(seeds))))
            for n in _chunks(budget["shots_base"], SHOTS_PER_TASK):
                tasks.append(("iid", (d, t), dict(shots=n), int(next(seeds))))
                if need_base:  # P_L(no event) is shared by all (k, D) points of this circuit
                    tasks.append(("base", (d, t), dict(shots=n), int(next(seeds))))
        # Longest tasks first, so the pool does not end with a few stragglers
        tasks.sort(key=lambda tk: 0 if tk[0] == "events" and tk[2]["D_ticks"] > 1000 else 1)
        print(f"{len(tasks)} sampling tasks on {jobs} workers", flush=True)

        base_f, iid_f, rates = {}, {}, {i: [] for i in range(len(points))}
        with ctx.Pool(jobs, initializer=_init_sampler, initargs=(circuits,)) as pool:
            for i, (task, out) in enumerate(zip(tasks, pool.imap(_sample_task, tasks, chunksize=1)), 1):
                kind, key, spec, _ = task
                if kind == "events":
                    rates[spec["point"]] += out["rates"]
                else:
                    acc = base_f if kind == "base" else iid_f
                    f0, n0 = acc.get(key, (0, 0))
                    acc[key] = (f0 + out["failures"][0], n0 + out["failures"][1])
                if i % max(1, len(tasks) // 20) == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)} tasks done ({time.time() - t_start:.0f} s)", flush=True)

        # --- 3) combine exactly as the serial estimator did ---------------------------------------
        results = []
        for (d, t) in info:
            f, n = iid_f[(d, t)]
            results.append(dict(d=d, compiler=t, k="iid", D="iid", P_L=f / n,
                                se=math.sqrt((f / n) * (1 - f / n) / n)))
        for i, p in enumerate(points):
            key = (p["d"], p["compiler"])
            p_none = math.exp(-p["Lambda"])
            f0, n0 = base_f.get(key, (0, 1)) if p_none > 1e-6 else (0, 1)
            P0 = f0 / n0
            r = np.array(rates[i])
            P1 = r.mean()
            P = p_none * P0 + (1 - p_none) * P1
            se = math.sqrt(p_none**2 * P0 * (1 - P0) / n0 + (1 - p_none) ** 2 * r.var(ddof=1) / len(r))
            results.append(dict(d=p["d"], compiler=p["compiler"], k=p["k"], D=p["D"], P_L=P, se=se,
                                Lambda=p["Lambda"], P_base=P0, P_given_event=P1, T_layers=p["T_layers"]))
        for r in results:
            print({k: r[k] for k in ("d", "compiler", "k", "D")}, f"P_L={r['P_L']:.3e} ± {r['se']:.1e}")
        print(f"total {time.time() - t_start:.0f} s")
        with open(pkl, "wb") as f:
            pickle.dump(dict(results=results, layouts=layouts, budget=budget), f)

    with open(pkl, "rb") as f:
        data = pickle.load(f)
    plot_evaluation(data["results"], str(pkl.with_suffix("")))


# ======================================================================================
# Single figure
# ======================================================================================
def plot_evaluation(results, filename):
    plt.rcParams.update({"font.family": "serif", "axes.labelsize": FONTSIZE * 1.2,
                         "font.size": FONTSIZE, "legend.fontsize": FONTSIZE - 3,
                         "lines.markeredgecolor": "black", "lines.markeredgewidth": 1.0})
    colors = {"compiled": "#3B6FA8", "sabre": "#C85E59"}
    ds = sorted({r["d"] for r in results})
    ls = dict(zip(ds, ["-", "--"]))
    markers = {1: "o", "d": "s", "long": "^"}
    xlab = ["i.i.d."] + [str(k) if k else "chiplet" for k in K_VALUES]
    xpos = {lab: i for i, lab in enumerate(xlab)}

    fig, ax = plt.subplots(figsize=(HEIGHT_FIGSIZE * 2.5 * 0.9, WIDTH_FIGSIZE * 0.6))
    for d in ds:
        for t in COMPILERS:
            rows = [r for r in results if r["d"] == d and r["compiler"] == t]
            iid = next(r for r in rows if r["k"] == "iid")
            ax.errorbar(0, iid["P_L"], yerr=iid["se"], fmt="*", ms=11, color=colors[t],
                        mfc=colors[t] if d == ds[0] else "white")
            ax.axhline(iid["P_L"], color=colors[t], ls=ls[d], lw=0.6, alpha=0.35)
            for D in D_ROUNDS:
                pts = [r for r in rows if r["D"] == D]
                ax.errorbar([xpos[str(r["k"])] for r in pts], [r["P_L"] for r in pts],
                            yerr=[r["se"] for r in pts], ls=ls[d], marker=markers[D],
                            color=colors[t], ms=5, lw=1.2, capsize=2,
                            mfc=colors[t] if d == ds[0] else "white")
    ax.set_xticks(range(len(xlab)), xlab)
    ax.set_xlabel("Qubits affected per event $k$ (equal average error rate)")
    ax.set_ylabel("LER (logical CNOT)")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    h = [Line2D([], [], color=colors[t], lw=2, label=LABELS[t]) for t in COMPILERS]
    h += [Line2D([], [], color="gray", ls=ls[d], label=f"$d={d}$") for d in ds]
    h += [Line2D([], [], color="gray", marker=markers[D], ls="none",
                 label={1: "1 round", "d": "$d$ rounds", "long": "ms-scale"}[D]) for D in D_ROUNDS]
    ax.legend(handles=h, ncol=3, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(filename + ".pdf", format="pdf")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", action="store_true", help="d=3, tiny budget: pipeline smoke test")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 1,
                    help="worker processes (default: all cores)")
    ap.add_argument("--configs", type=int, default=None,
                    help=f"event configurations per (k, D) point (default {N_CONFIGS})")
    ap.add_argument("--shots-per-config", type=int, default=None,
                    help=f"shots per event configuration (default {SHOTS_PER_CONFIG})")
    ap.add_argument("--shots-base", type=int, default=None,
                    help=f"no-event baseline and i.i.d. shots per circuit (default {SHOTS_BASE})")
    a = ap.parse_args()
    run_exp_cosmic_ray_impact(reproduce=not a.plot_only, toy=a.toy, jobs=a.jobs,
                              n_configs=a.configs, shots_per_config=a.shots_per_config,
                              shots_base=a.shots_base)