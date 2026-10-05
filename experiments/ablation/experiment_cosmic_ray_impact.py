"""
Ablation: correlated chiplet-local (cosmic-ray-like, CRL) noise on a distributed lattice-surgery CNOT.

Motivation: Wu et al., PR Applied 24, 044022 (2025) [R1]. CRL events suppress T1 on all qubits of the
struck module (85-94 % intra-module coincidence), are nearly confined to it (~2 % inter-module), and last
~2-6 ms, i.e. thousands of QEC cycles.

Configurations: Ideal (untranspiled CNOT, no links), Chipmunq, LightSABRE.

Workload: the same logical CNOT as experiment_distributed_lattice_surgery (tqec, control/ancilla/target
patches + merge strips, predefined partitions), on the same BackendChipletV2 sizes, so each patch fits on
one chiplet. Chipmunq places one patch per chiplet; LightSABRE (TrivialLayout + SabreSwap) packs them.

Noise = modsi1000 (incl. inter-chiplet links) + CRL bursts:
  * An event hits the k circuit qubits nearest a random used qubit inside ONE chiplet for a ms-scale window (outlasting the CNOT); each cluster qubit gets
    DEPOLARIZE1(p_b) every tick of the window (Pauli-twirled T1 collapse). k=None -> whole chiplet.
  * With probability rho it spills into one other chiplet (R1: ~2 %).
  * Stationary Poisson process whose start rate keeps the expected extra error per hardware qubit per
    tick equal to `delta` for every (k, D). Only the correlation structure changes.
  * Duration is wall-clock (ticks), converted from rounds with the ticks/round of the *uncompiled*
    reference circuit, so SWAP-heavy schedules are correctly exposed for longer.
  * Events only matter on chiplets the circuit touches, so sampling is restricted to those ("thinning");
    this is exact up to the O(rho) case of a spill from an unused chiplet into a used one.
  * Reference: plain modsi1000 without cosmic rays. Every k adds the same average extra error (delta),
    so curves vs k isolate the correlation structure; the gap to the reference is the total CRL cost.
Decoder: pymatching from the plain modsi1000 DEM (unaware of bursts, as events are unheralded).
Estimator: P_L = e^-Lambda P_L(modsi1000) + (1 - e^-Lambda) E[P_L | >=1 event]; the modsi1000 term is
the reference point itself, so it is simulated once and shared by every k.

Run from the repo root:  python experiments/ablation/experiment_cosmic_ray_impact.py [--quick] [--plot-only]
"""
from __future__ import annotations

import argparse
import math
import multiprocessing
import os
import pickle
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

sys.path.append(os.path.join(os.getcwd(), "."))

import matplotlib.pyplot as plt
import numpy as np
import pymatching
import stim
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation

from experiments.exp_utils.circuit_generator import get_tqec_cnot_rotated
from experiments.exp_utils.circuit_noise import get_noise_model
from experiments.exp_utils.simulation_utils import transpile_stim_circuit
from experiments.exp_utils.transpilation_utils import sabre_transpilation
from experiments.exp_utils.utils import plot_lib_color
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit
from glue.qiskit_qec.stim_tools import get_stim_circuits_with_detectors
from qeccm.backends.BackendChipletV2 import BackendChipletV2

# ======================================================================================
# Experiment parameters
# ======================================================================================
DISTANCE_SCALES = [2, 3]                 # d = 2k+1 -> d = 5, 7
COMPILERS = ["default", "compiled", "sabre"]   # naming as in experiment_distributed_lattice_surgery
LABELS = {"default": "Ideal", "compiled": "Chipmunq", "sabre": "LightSABRE"}
# default = Ideal: the CNOT without any transpilation (modsi1000, no inter-chiplet links). Its patches sit on
# the chiplets Chipmunq assigns them to, so cosmic-ray events strike the same patch groups.
K_VALUES = [1, 4, 16, 64, None]          # None = whole chiplet (the R1-realistic footprint)
EVENT_ROUNDS = 1000                      # ms-scale event (R1: tau ≈ 2-6 ms at ~1 µs cycles); outlasts the CNOT
P_PHYS = 1e-3                            # modsi1000 strength
P_INTER = 1e-3                           # inter-chiplet link noise
DELTA = 0.1 * P_PHYS                     # extra avg error / qubit / tick, identical everywhere
P_BURST = 0.1                            # per-tick depolarizing strength inside an event
RHO = 0.02                               # inter-chiplet spill-over (R1)
N_CONFIGS = 200                          # event placements per point (dominates the error bar)
SHOTS_PER_CONFIG = 500                   # P(fail | event) is O(0.1-1): few shots per placement suffice
SHOTS_BASE = 500_000                     # plain modsi1000 reference, computed ONCE per (d, compiler)
CONFIGS_PER_JOB = 10                     # job granularity -> keeps all cores busy
SHOTS_BASE_PER_JOB = 100_000
OUT_DIR = Path("experiments/evaluation/ablation")

# Same backends as experiment_distributed_lattice_surgery: patch (h x w) fits in one chiplet
BACKEND_CFG = {1: ((6, 6, 11, 6), 5), 2: ((6, 6, 15, 8), 7), 3: ((6, 6, 19, 10), 9), 4: ((6, 6, 23, 12), 11)}


# ======================================================================================
# Hardware view used by the noise model
# ======================================================================================
@dataclass
class Hardware:
    """Chiplet geometry + which qubits the compiled circuit actually uses.

    k counts *circuit* qubits: an event of size k hits the k used qubits closest (in hardware-graph
    distance, through idle qubits too) to a random used seed qubit on the struck chiplet.
    "Whole chiplet" = every circuit qubit on that chiplet. Chiplets are hit with probability ∝ area.
    """
    chiplet_of: dict[int, int]                  # every backend qubit -> chiplet
    edges: list[tuple[int, int]]                # coupling edges (inter-chiplet ones are ignored)
    used: set[int]                              # qubits the circuit acts on
    areas: dict[int, int] | None = None         # chiplet -> #qubits (default: from chiplet_of); lists ALL chiplets
    adj: dict[int, list[int]] = field(init=False)
    chiplets: dict[int, list[int]] = field(init=False)
    used_on: dict[int, list[int]] = field(init=False)

    def __post_init__(self):
        self.adj = {q: [] for q in self.chiplet_of}
        for a, b in self.edges:
            if self.chiplet_of[a] == self.chiplet_of[b]:
                self.adj[a].append(b)
                self.adj[b].append(a)
        self.chiplets, self.used_on = {}, {}
        for q, c in self.chiplet_of.items():
            self.chiplets.setdefault(c, []).append(q)
            if q in self.used:
                self.used_on.setdefault(c, []).append(q)
        if self.areas is None:
            self.areas = {c: len(qs) for c, qs in self.chiplets.items()}
        self._all = sorted(self.areas)
        self._act = sorted(self.used_on)          # chiplets the circuit touches

    def _probs(self, ids):
        s = np.array([self.areas[c] for c in ids], float)            # hit prob ∝ chiplet area
        return s / s.sum()

    @property
    def n_active(self) -> int:
        """Number of circuit qubits (the population the average error budget is spread over)."""
        return sum(len(v) for v in self.used_on.values())

    def k_eff(self, k):
        """Expected number of circuit qubits hit by one event on an active chiplet."""
        a = np.array([len(self.used_on[c]) for c in self._act])
        return float(((a if k is None else np.minimum(k, a)) * self._probs(self._act)).sum())

    def random_active_chiplet(self, rng):
        return int(rng.choice(self._act, p=self._probs(self._act)))

    def random_other_chiplet(self, rng, exclude):
        ids = [c for c in self._all if c != exclude]
        return int(rng.choice(ids, p=self._probs(ids)))

    def cluster(self, chiplet, k, rng) -> frozenset[int]:
        targets = self.used_on.get(chiplet, [])
        if not targets:
            return frozenset()                    # spill into a chiplet the circuit doesn't use
        if k is None or k >= len(targets):
            return frozenset(targets)
        seed = int(rng.choice(targets))
        seen, out, dq = {seed}, [], deque([seed])
        while dq and len(out) < k:
            q = dq.popleft()
            if q in self.used:
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


class LayeredCircuit:
    """Flattened circuit pre-split into tick layers (as text) for fast burst injection.

    Building an event circuit = string assembly + one C++ parse, instead of appending every
    instruction in Python. Noise is inserted at the start of each tick layer.
    """

    def __init__(self, circ: stim.Circuit):
        flat = circ.flattened()
        layers = [[]]
        for inst in flat:
            if inst.name == "TICK":
                layers.append([])
            else:
                layers[-1].append(str(inst))
        self.layers = ["\n".join(L) for L in layers]
        self.T = len(self.layers)
        self.used = used_qubits(flat)

    def with_events(self, events, p_b) -> stim.Circuit:
        """events: list of (qubits, t0, t1), t1 exclusive."""
        events = [(sorted(qs & self.used), t0, t1) for qs, t0, t1 in events]
        events = [e for e in events if e[0] and e[1] < e[2]]
        bps = sorted({0, self.T} | {t for _, t0, t1 in events for t in (t0, t1)})
        p = min(p_b, 0.75)
        parts = []
        for a, b in zip(bps[:-1], bps[1:]):
            hdr = "\n".join(f"DEPOLARIZE1({p}) " + " ".join(map(str, qs))
                            for qs, t0, t1 in events if t0 <= a < t1)
            for layer in range(a, b):
                if hdr:
                    parts.append(hdr)
                if self.layers[layer]:
                    parts.append(self.layers[layer])
                if layer < self.T - 1:
                    parts.append("TICK")
        return stim.Circuit("\n".join(parts))


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
    """Zero-truncated Poisson."""
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


def event_lambda(hw, T, *, k, D_ticks, p_b, delta, rho):
    """Expected number of events whose window intersects [0, T) (equal-average normalization)."""
    start_rate = delta * hw.n_active / (hw.k_eff(k) * D_ticks * p_b * (1 + rho))
    return start_rate * (T + D_ticks - 1)


def sample_events(hw, T, rng, *, lam, k, D_ticks, rho):
    events = []
    for _ in range(_sample_ztp(lam, rng)):
        t0 = int(rng.integers(-(D_ticks - 1), T))          # stationary: may start before t=0
        win = (max(t0, 0), min(t0 + D_ticks, T))
        c = hw.random_active_chiplet(rng)
        events.append((hw.cluster(c, k, rng), *win))
        if rng.random() < rho:
            events.append((hw.cluster(hw.random_other_chiplet(rng, c), k, rng), *win))
    return events


# ======================================================================================
# Chipmunq pipeline: backend, compilation, noise
# ======================================================================================
def get_backend(k):
    size, nic = BACKEND_CFG[k]
    return BackendChipletV2(size=size, n_inter=nic, connectivity="nn", topology="rotated_grid",
                            inter_chiplet_noise=P_INTER, inter_chiplet_amplification=1,
                            inter_chiplet_noise_type="constant", num_defective_qubits=0)


def compile_cnot(k, t, backend):
    """Returns (noisy stim circuit, reference ticks/round, patch->chiplets, partitions).

    t = "default": no transpilation (ideal), circuit qubit indices, modsi1000 without inter-chiplet links;
    patch->chiplets is None (filled in from Chipmunq's placement by hardware_ideal)."""
    circuit, partitions = get_tqec_cnot_rotated(distance_scale=k, n1=1, n2=0)
    ref = get_stim_circuits_with_detectors(StimCodeCircuit(circuit).qc)[0][0]
    if t == "default":
        noisy = get_noise_model("modsi1000", None, P_PHYS, None, remote=None).noisy_circuit(ref)
        return noisy, ticks_per_round(ref), None, partitions
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
    return noisy, ticks_per_round(ref), patch_chiplets, partitions


def hardware_ideal(backend, circ, partitions, patch_chiplets):
    """Chiplet view of the untranspiled circuit: every patch / merge strip on the chiplet Chipmunq puts it on,
    chiplets as large as the backend's, neighbourhood = the circuit's own two-qubit interactions."""
    chiplet_of = {}
    for part, chips in zip(partitions, patch_chiplets):
        for q in part["indices"]:
            chiplet_of.setdefault(int(q), chips[0])
    edges = set()
    for inst in circ.flattened():
        if inst.name in _ANNOT:
            continue
        gd = stim.gate_data(inst.name)
        if gd.is_two_qubit_gate and not gd.is_noisy_gate:
            qs = _qubits_of(inst)
            edges.update(tuple(sorted(e)) for e in zip(qs[::2], qs[1::2]))
    used = active_qubits(circ)
    # qubits outside every partition join the chiplet of an interaction neighbour (BFS)
    nbrs = defaultdict(set)
    for a, b in edges:
        nbrs[a].add(b)
        nbrs[b].add(a)
    todo = deque(q for q in chiplet_of)
    while todo:
        q = todo.popleft()
        for nb in nbrs[q]:
            if nb not in chiplet_of:
                chiplet_of[nb] = chiplet_of[q]
                todo.append(nb)
    for q in used - set(chiplet_of):         # isolated qubits (none expected)
        chiplet_of[q] = patch_chiplets[0][0]
    areas = {c: len(qs) for c, qs in backend.chiplet_to_nodes.items()}
    return Hardware(chiplet_of=chiplet_of, edges=sorted(edges), used=used, areas=areas)


def hardware_for(backend, circ):
    used = {q for q in active_qubits(circ) if q in backend.node_to_chiplet}
    edges = [tuple(map(int, e)) for e in backend.coupling_map.get_edges()]
    return Hardware(chiplet_of=dict(backend.node_to_chiplet), edges=edges, used=used)


# ======================================================================================
# Sweep (parallel over batches of event configurations)
# ======================================================================================
# Per-worker cache: circuits, layered circuits, hardware and decoders are built once per worker.
_W = {}


def _init_worker(circ_strs, hws):
    _W.clear()
    _W["src"], _W["hw"], _W["obj"] = circ_strs, hws, {}


def _get(key):
    if key not in _W["obj"]:
        base = stim.Circuit(_W["src"][key])
        _W["obj"][key] = (base, LayeredCircuit(base), _matcher(base))
    return _W["obj"][key]


def _worker(job):
    kind, key, p, seed = job
    rng = np.random.default_rng(seed)
    base, layered, matcher = _get(key)
    if kind == "ref":
        f, n = _failures(base, matcher, p["shots"], rng)
        return kind, key, None, (f, n)
    hw = _W["hw"][key]
    rates = []
    for _ in range(p["n_cfg"]):
        events = sample_events(hw, layered.T, rng, lam=p["lam"], k=p["k"], D_ticks=p["D_ticks"],
                               rho=RHO)
        f, n = _failures(layered.with_events(events, P_BURST), matcher, p["shots"], rng)
        rates.append(f / n)
    return kind, key, p["k"], rates


def run_exp_cosmic_ray_impact(reproduce=True, quick=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pkl = OUT_DIR / f"cosmic_ray_impact{'_quick' if quick else ''}.pkl"
    scales = [1] if quick else DISTANCE_SCALES
    n_configs, shots_cfg, shots_base, cfg_per_job, base_per_job = (
        (20, 200, 20_000, 5, 10_000) if quick else
        (N_CONFIGS, SHOTS_PER_CONFIG, SHOTS_BASE, CONFIGS_PER_JOB, SHOTS_BASE_PER_JOB))

    if reproduce:
        circ_strs, hws, layouts, lams, jobs = {}, {}, {}, {}, []
        for k in scales:
            d = 2 * k + 1
            backend = get_backend(k)
            chipmunq_patches = None
            for t in sorted(COMPILERS, key=lambda c: c == "default"):    # Ideal last: needs Chipmunq's placement
                key = (d, t)
                circ, tpr, patch_chiplets, partitions = compile_cnot(k, t, backend)
                if t == "compiled":
                    chipmunq_patches = patch_chiplets
                if t == "default":
                    if chipmunq_patches is None:
                        chipmunq_patches = compile_cnot(k, "compiled", backend)[2]
                    patch_chiplets = chipmunq_patches
                    hw = hardware_ideal(backend, circ, partitions, patch_chiplets)
                else:
                    hw = hardware_for(backend, circ)
                T = num_layers(circ)
                circ_strs[key], hws[key] = str(circ), hw
                occ = {c: len(hw.used_on[c]) for c in hw._act}
                layouts[key] = dict(patch_chiplets=patch_chiplets, circuit_qubits_per_chiplet=occ,
                                    ticks=T, ticks_per_round_ref=tpr)
                print(f"[layout] d={d} {LABELS[t]}: patch->chiplets {patch_chiplets}, "
                      f"circuit qubits per chiplet {occ}, {T} ticks", flush=True)
                for i in range(math.ceil(shots_base / base_per_job)):
                    jobs.append(("ref", key, dict(shots=base_per_job), len(jobs)))
                D_ticks = max(1, round(EVENT_ROUNDS * tpr))
                for kk in K_VALUES:
                    lam = event_lambda(hw, T, k=kk, D_ticks=D_ticks, p_b=P_BURST, delta=DELTA, rho=RHO)
                    lams[(key, kk)] = lam
                    for i in range(math.ceil(n_configs / cfg_per_job)):
                        jobs.append(("cfg", key, dict(k=kk, lam=lam, D_ticks=D_ticks, n_cfg=cfg_per_job,
                                                      shots=shots_cfg), len(jobs)))
        # Longest jobs first (event batches on long circuits) for better load balancing
        jobs.sort(key=lambda j: (j[0] == "cfg") * layouts[j[1]]["ticks"], reverse=True)

        ref_counts = {key: [0, 0] for key in circ_strs}
        cfg_rates = {pk: [] for pk in lams}
        n_workers = max(1, multiprocessing.cpu_count() // 2)
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(n_workers, initializer=_init_worker, initargs=(circ_strs, hws)) as pool:
            for i, (kind, key, kk, out) in enumerate(pool.imap_unordered(_worker, jobs), 1):
                if kind == "ref":
                    ref_counts[key][0] += out[0]
                    ref_counts[key][1] += out[1]
                else:
                    cfg_rates[(key, kk)].extend(out)
                if i % 10 == 0 or i == len(jobs):
                    print(f"[{i}/{len(jobs)}] jobs done", flush=True)

        results = []
        for key, (f0, n0) in ref_counts.items():
            d, t = key
            P0 = f0 / n0
            results.append(dict(d=d, compiler=t, k="ref", P_L=float(P0), se=math.sqrt(P0 * (1 - P0) / n0)))
            for kk in K_VALUES:
                lam, rates = lams[(key, kk)], np.array(cfg_rates[(key, kk)])
                p_none = math.exp(-lam)
                P1 = rates.mean()
                se = math.sqrt(p_none**2 * P0 * (1 - P0) / n0
                               + (1 - p_none) ** 2 * rates.var(ddof=1) / len(rates))
                results.append(dict(d=d, compiler=t, k=kk if kk else "chiplet",
                                    P_L=float(p_none * P0 + (1 - p_none) * P1), se=se, Lambda=lam,
                                    P_base=P0, P_given_event=float(P1)))
                print(results[-1], flush=True)
        with open(pkl, "wb") as f:
            pickle.dump(dict(results=results, layouts=layouts), f)

    with open(pkl, "rb") as f:
        data = pickle.load(f)
    plot_evaluation(data["results"], str(pkl.with_suffix("")))


# ======================================================================================
# Paper figure style
# ======================================================================================
# Figure style: identical to experiment_defective_qubits (Fig. 10). Figures are drawn at their printed
# size (7 pt fonts) -- include them at their natural width, no scaling. The components ablation and the
# cosmic-ray experiment share one row: 2/3 and 1/3 of the text width.
# (identical block in experiment_mapping_routing_ablation.py and experiment_cosmic_ray_impact.py)
TEXT_WIDTH_IN = 7.0                   # full text width of the paper (two-column IEEE/ACM: ~7.0 in)
PANEL_H = 1.6
FONT_PT = 7
# Absolute margins [in] = Fig. 10's MARGINS on its 1.75 x 1.6 in panels, so the axes line up
MARGIN_IN = dict(left=0.27 * 1.75, right=0.03 * 1.75, top=0.20 * PANEL_H, bottom=0.24 * PANEL_H)


def _fonts():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": FONT_PT,
        "axes.labelsize": FONT_PT,
        "axes.titlesize": FONT_PT,
        "legend.fontsize": FONT_PT,
        "xtick.labelsize": FONT_PT - 1,
        "ytick.labelsize": FONT_PT - 1,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "ytick.minor.size": 1.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.labelpad": 2,
        "axes.linewidth": 0.6,
        "hatch.linewidth": 0.4,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "errorbar.capsize": 1.5,
    })


def _panel(width: float):
    fig, ax = plt.subplots(figsize=(width, PANEL_H))
    fig.subplots_adjust(left=MARGIN_IN["left"] / width, right=1 - MARGIN_IN["right"] / width,
                        top=1 - MARGIN_IN["top"] / PANEL_H, bottom=MARGIN_IN["bottom"] / PANEL_H)
    return fig, ax


def _keep_inside(fig, text, pad_in: float = 0.02) -> None:
    """Shift a text horizontally (in inches, dpi-independent) just enough to stay inside the figure."""
    bb = text.get_window_extent(fig.canvas.get_renderer())
    lo, hi = fig.bbox.x0 + pad_in * fig.dpi, fig.bbox.x1 - pad_in * fig.dpi
    shift = min(0.0, hi - bb.x1) or max(0.0, lo - bb.x0)
    if shift:
        text.set_transform(text.get_transform() + ScaledTranslation(shift / fig.dpi, 0, fig.dpi_scale_trans))


def _title(fig, ax, text: str, better: str = "Lower is better ↓") -> None:
    """Centred panel title with the "better" hint on a second line above it (as in Fig. 10)."""
    for y, s, kw in ((1.03, text, {}), (1.16, better, {"color": plot_lib_color})):
        _keep_inside(fig, ax.text(0.5, y, s, transform=ax.transAxes, fontweight="bold", ha="center",
                                  va="bottom", **kw))


# ======================================================================================
# Single figure
# ======================================================================================
def plot_evaluation(results, filename):
    _fonts()
    colors = {"default": "black", "compiled": "#3B6FA8", "sabre": "#C85E59"}
    ds = sorted({r["d"] for r in results})
    comps = [t for t in COMPILERS if any(r["compiler"] == t for r in results)]
    ls = {d: ["-", "--", ":", "-."][i % 4] for i, d in enumerate(ds)}
    mk = {d: ["o", "s", "^", "D"][i % 4] for i, d in enumerate(ds)}
    xkeys = ["ref"] + [str(k) if k else "chiplet" for k in K_VALUES]     # keys as stored in results
    xlab = ["0"] + [str(k) if k else "All" for k in K_VALUES]            # 0 = plain SI1000, All = whole chiplet
    xpos = {key: i for i, key in enumerate(xkeys)}

    fig, ax = _panel(TEXT_WIDTH_IN / 4)  # four plots across the full text width (1.75 in each)
    for d in ds:
        for t in comps:
            rows = [r for r in results if r["d"] == d and r["compiler"] == t]
            pts = sorted(rows, key=lambda r: xpos[str(r["k"])])
            ax.errorbar([xpos[str(r["k"])] for r in pts], [r["P_L"] for r in pts],
                        yerr=[r["se"] for r in pts], ls=ls[d], marker=mk[d], color=colors[t],
                        markeredgecolor="black", markeredgewidth=0.5, elinewidth=0.6, capthick=0.6)
    ax.set_xticks(range(len(xlab)), xlab)
    ax.set_xlabel("Qubits affected per event $k$")
    ax.set_ylabel("LER")  # per logical CNOT (state in the caption)
    ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    _title(fig, ax, "b) Correlated errors")  # long title does not fit the 1.75 in panel (caption has the rest)
    fig.savefig(filename + ".pdf", format="pdf")
    plt.close(fig)

    # Legend outside the plot, in its own file, two rows: compilers on top, code distances below
    top = [Line2D([], [], color=colors[t], label=LABELS[t]) for t in comps]
    bottom = [Line2D([], [], color="gray", ls=ls[d], marker=mk[d], mfc="gray", mec="black",
                     markeredgewidth=0.5, label=f"$d={d}$") for d in ds]
    # Two separate one-row legends, each centred, so the shorter row sits in the middle under the longer one
    legend_fig = plt.figure(figsize=(TEXT_WIDTH_IN / 4, 0.4))
    kw = dict(frameon=False, handlelength=1.8, handletextpad=0.4, columnspacing=1.0, borderaxespad=0.0)
    legend_fig.legend(handles=top, loc="lower center", bbox_to_anchor=(0.5, 0.5), ncols=len(top), **kw)
    legend_fig.legend(handles=bottom, loc="upper center", bbox_to_anchor=(0.5, 0.5), ncols=len(bottom), **kw)
    legend_fig.savefig(filename + "legend.pdf", bbox_inches="tight", format="pdf")
    plt.close(legend_fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="d=3, tiny budget: pipeline smoke test")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    run_exp_cosmic_ray_impact(reproduce=not a.plot_only, quick=a.quick)