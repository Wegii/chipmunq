"""Evaluate Chipmunq's mapping and routing independently (ablation).

Question: how much of Chipmunq's LER / overhead advantage comes from each stage?

All variants of one backend instance are compiled for the *same* circuit on the *same*
BackendChipletV2 object (identical defects and link noise); only one stage changes.

    Variant                   mapping                                   routing
    ------------------------  ----------------------------------------  --------------------
    chipmunq_basic            Chipmunq (full)                           Basic      <- reference
    chipmunq_sabre_default    Chipmunq                                  SABRE-SWAP, qubit map restored after each TICK segment
    no_partitioning           KaHyPar blocks instead of patch IR        Basic
    no_patch_contraction      qubit-level SABRE inside assigned chiplet Basic
    no_sequencing             random patch order instead of BFS         Basic
    no_global_mapping         random patch->chiplet assignment          Basic
    lightsabre                SabreLayout                               SABRE-SWAP (default)       (baseline)
    seqc                      SEQC (own layout)                         SEQC, noise-aware links    (baseline)
    olsq2                     OLSQ2 (SMT layout synthesis, SWAP objective)                         (baseline)
    murali                    Murali et al. GreedyE* placement          noise-adaptive routing     (baseline)

Baselines compile the full circuit themselves on the working qubits only (defective qubits removed,
indices mapped back), each in its own process with a wall-clock limit (``baseline_timeout_s``);
runs over the limit are recorded and drawn as T/O.

``chipmunq_sabre_default`` routes the Chipmunq mapping with LightSABRE's SabreSwap (standard trial count)
as a pure router: every segment between Stim TICKs is routed from the same placement and the permutation
is undone afterwards, so the qubit map stays fixed as with Basic. The reported budget B (runtime of Basic
routing) is informational only.

Reported per variant and backend (clean / defective): mapping + routing runtime, LER at P_PHYS,
2q-gate overhead (SWAP = 3), depth overhead, inter-chiplet 2q gates, compile failures.

Noise: circuit-level ``modsi1000`` with the same inter-chiplet noise ``ps_inter`` on every link,
applied orientation-independently (see ``symmetric_remote_noise``).

Paper figure (a, next to the cosmic-ray experiment b): ``ablation_ler_ratio_*.pdf``, drawn at its printed
size (2/3 of the text width, Fig. 10 style) -- include it at its natural width, no scaling.

Usage:
    python experiments/ablation/experiment_components.py            # full
    python experiments/ablation/experiment_components.py --quick    # smoke test
    python experiments/ablation/experiment_components.py --plot-only
    python experiments/ablation/experiment_components.py --clean-only [--plot-only]  # clean backend only
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import pickle
import sys
import traceback
from collections import Counter, defaultdict
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

sys.path.append(os.path.join(os.getcwd(), "."))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sinter  # noqa: E402
import stim  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

from qiskit import QuantumCircuit  # noqa: E402

from experiments.exp_utils.ablation_utils import (  # noqa: E402
    RandomSequenceMapper,
    check_routed,
    chipmunq_mapping,
    intra_chiplet_sabre_mapping,
    kahypar_grid_partitions,
    lightsabre,
    murali_noise,
    overhead_stats,
    permute_chiplets,
    quiet,
    repair_defects,
    reduced_coupling_map,
    route_from_mapping,
    sabre_mapping,
    split_annotations,
    stim_with_annotations,
    symmetric_remote_noise,
)
from experiments.exp_utils.transpilation_utils import TIMEOUT, run_with_timeout  # noqa: E402
from experiments.exp_utils.circuit_generator import get_tqec_cnot_rotated  # noqa: E402
from experiments.exp_utils.circuit_noise import get_noise_model  # noqa: E402
from experiments.exp_utils.simulation_utils import run_sinter_simulation  # noqa: E402
from experiments.exp_utils.utils import FONTSIZE, HEIGHT_FIGSIZE, WIDTH_FIGSIZE, plot_lib_color  # noqa: E402
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit  # noqa: E402
from qeccm.backends.BackendChipletV2 import BackendChipletV2  # noqa: E402
from qeccm.backends.backend_utils import generate_coordinates  # noqa: E402

OUTPUT_DIR = Path("experiments/evaluation/mapping_routing_ablation")

# (key, label, colour, hatch) -- order is the plotting order
VARIANTS = [
    ("chipmunq_basic", "Chipmunq (Basic)", "#A7D9ED", ""),
    ("chipmunq_sabre_default", "Chipmunq map + SABRE-SWAP (fixed map)", "#E6A96B", "\\\\"),
    ("no_partitioning", "- patch-IR partitioning", "#CFE8CF", ".."),
    ("no_patch_contraction", "- patch contraction", "#A9D3A9", "xx"),
    ("no_sequencing", "- sequencing", "#82BD82", "--"),
    ("no_global_mapping", "- global mapping", "#5E9E5E", "++"),
    # Baselines: colours as in the other paper figures (transpilation_utils.METHOD_STYLES)
    ("lightsabre", "LightSABRE", "lightcoral", "o"),
    ("seqc", "SEQC", "#C9B7E3", ".."),
    ("olsq2", "OLSQ2", "#F3C98B", "\\\\"),
    ("murali", "Murali et al.", "#D5D5D5", "--"),
]
BASELINES = ("seqc", "olsq2", "murali")  # full compilers run under baseline_timeout_s (LightSABRE runs inline)
LABEL = {k: l for k, l, _, _ in VARIANTS}
COLOR = {k: c for k, _, c, _ in VARIANTS}
HATCH = {k: h for k, _, _, h in VARIANTS}
REFERENCE = "chipmunq_basic"

FULL_CONFIG = dict(
    k=3,  # distance_scale; d = 2k + 1
    ps=[1e-3],  # single physical error rate: the paper reports one relative LER per variant
    ps_inter=1e-3,
    n_seeds=4,
    defects={"clean": 0, "defective": 2},  # defective qubits per chiplet
    max_backend_retries=20,
    max_variant_retries=5,
    num_shots=10_000_000,
    max_errors=5_000,
    num_workers=None,
    baseline_timeout_s=600,  # wall-clock limit per baseline compilation (SEQC, OLSQ2, Murali et al.)
    compile_workers=None,    # parallel configurations; None -> #CPUs / (1 + #baselines), 1 -> serial
)
QUICK_CONFIG = dict(FULL_CONFIG, k=2, n_seeds=1, defects={"clean": 0, "defective": 1},
                    num_shots=20_000, max_errors=200, baseline_timeout_s=1000)

BACKEND_SIZES = {1: ((2, 2, 11, 6), 5), 2: ((2, 2, 15, 8), 7), 3: ((2, 2, 19, 10), 9), 4: ((2, 2, 23, 12), 11)}


def make_backend(k: int, n_defects: int, seed: int, cfg: dict) -> BackendChipletV2:
    size, nic = BACKEND_SIZES[k]
    return BackendChipletV2(
        size=size, n_inter=nic, connectivity="nn", topology="rotated_grid",
        inter_chiplet_noise=cfg["ps_inter"], inter_chiplet_amplification=1,
        inter_chiplet_noise_type="constant",  # identical noise on every link
        num_defective_qubits=n_defects, rng_seed=seed,
    )


# --------------------------------------------------------------------------------------
# Compilation
# --------------------------------------------------------------------------------------
def _finish(name, routed, route_info, t_map, qc, stim_ref, backend, extra=None, stim_circuit=None):
    stim_c = check_routed(routed, stim_ref, backend, stim_circuit=stim_circuit)
    rec = {
        "ok": True,
        "mapping_s": t_map,
        "routing_s": route_info["routing_s"],
        "total_s": t_map + route_info["routing_s"],
        "trials": route_info.get("trials"),
        "budget_s": route_info.get("budget_s"),
        "stim": str(stim_c),
        **overhead_stats(routed, qc, backend),
    }
    rec.update(extra or {})
    return rec


# --------------------------------------------------------------------------------------
# Baselines (full compilers). Module-level so they can run in a spawned, killable process.
# --------------------------------------------------------------------------------------
def _working_view(backend: BackendChipletV2):
    """Coupling map on the working qubits only, with local <-> global index maps."""
    cmap, l2g = reduced_coupling_map(backend)
    return cmap, l2g, {g: i for i, g in enumerate(l2g)}


def _to_global(out: QuantumCircuit, l2g: list, n_total: int) -> QuantumCircuit:
    """Place a circuit compiled on the working-qubit view back onto the backend's physical qubits
    (clbit indices are kept, so detectors can be re-attached)."""
    full = QuantumCircuit(n_total, out.num_clbits, name=out.name)
    full.compose(out, qubits=[l2g[i] for i in range(out.num_qubits)], clbits=list(range(out.num_clbits)),
                 inplace=True)
    return full


def _seqc_target_local(backend: BackendChipletV2, core: QuantumCircuit, cmap, l2g, g2l):
    """SEQC device model (as ablation_utils.seqc_target) on the working-qubit view: the circuit's own gates
    native, 2q gates on intra-chiplet couplers with their error, links SWAP-only with the per-link noise."""
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import SwapGate, get_standard_gate_name_mapping
    from qiskit.transpiler import InstructionProperties, Target

    from experiments.exp_utils.ablation_utils import ANNOTATIONS

    _, errs = murali_noise(backend)
    remote = symmetric_remote_noise(backend)
    std = get_standard_gate_name_mapping()
    t = Target(num_qubits=len(l2g))
    names = {i.operation.name for i in core.data} - ANNOTATIONS - {"swap"}
    names |= {"measure", "reset"}
    err = lambda a, b: errs[(min(l2g[a], l2g[b]), max(l2g[a], l2g[b]))]  # noqa: E731
    edges = list(cmap.get_edges())
    intra = [(a, b) for a, b in edges if (l2g[a], l2g[b]) not in remote]
    links = [(a, b) for a, b in edges if (l2g[a], l2g[b]) in remote]
    for name in sorted(names):
        op = std[name]
        if op.params:
            op = op.__class__(*[Parameter(f"{name}_{i}") for i in range(len(op.params))])
        if op.num_qubits == 1:
            t.add_instruction(op, {(q,): None for q in range(len(l2g))})
        else:
            t.add_instruction(op, {e: InstructionProperties(error=err(*e)) for e in intra})
    t.add_instruction(SwapGate(), {**{e: InstructionProperties(error=err(*e)) for e in intra},
                                   **{e: InstructionProperties(error=remote[(l2g[e[0]], l2g[e[1]])]) for e in links}})
    chiplets = [[g2l[int(q)] for q in backend.get_chiplet_at(c) if int(q) in g2l] for c in range(backend.get_num_chips())]
    return t, [c for c in chiplets if c]


def compile_baseline(method: str, qc: QuantumCircuit, backend: BackendChipletV2, seed: int = 0):
    """Full mapping + routing by a baseline on the working qubits. Returns (routed circuit on physical
    qubits without annotations, Stim circuit as string with the detectors re-attached, initial placement
    virtual qubit -> physical qubit)."""
    core, anns = split_annotations(qc)
    cmap, l2g, g2l = _working_view(backend)
    with quiet():
        if method == "seqc":
            from external.baseline.seqc.seqc import SEQCCompiler

            target, chiplets = _seqc_target_local(backend, core, cmap, l2g, g2l)
            out = SEQCCompiler(target, chiplets=chiplets, optimization_level=0, seed=seed, n_jobs=1).run(core)
            init = out.layout.initial_index_layout(filter_ancillas=True)
        elif method == "olsq2":
            from external.baseline.qls.qlsq2 import olsq2_transpilation

            out = olsq2_transpilation(core, cmap, objective="swap", mode="transition")
            init = out.metadata["olsq2_initial_layout"]
        elif method == "murali":
            from external.baseline.noise_aware_mapping.murali import noise_adaptive_transpilation

            _, errs = murali_noise(backend)
            local = {(g2l[a], g2l[b]): e for (a, b), e in errs.items() if a in g2l and b in g2l}
            local.update({(b, a): e for (a, b), e in list(local.items())})
            out = noise_adaptive_transpilation(core, cmap, method="greedy_e", cx_errors=local)
            init = out.metadata["noise_adaptive_layout"]
        else:
            raise ValueError(method)
    routed = _to_global(out, l2g, backend.num_qubits_total)
    layout = {v: l2g[int(init[v])] for v in range(qc.num_qubits)}  # initial placement, physical indices
    return routed, str(stim_with_annotations(routed, anns)), layout


def compile_instance(qc, stim_ref, partitions, backend, n_defects, cfg, seed) -> dict:
    """Compile every variant on one backend instance. Returns {variant: record}."""
    ps_inter = cfg["ps_inter"]
    res = {}

    def attempt(name, fn, retries=1):
        last = None
        for r in range(retries):
            try:
                res[name] = fn(r)
                return
            except Exception as e:  # record, never abort the whole instance
                last = f"{type(e).__name__}: {e}"
        res[name] = {"ok": False, "error": last}
        print(f"  [{name}] failed: {last}")

    base, t_base = chipmunq_mapping(qc, backend, partitions)

    # Chipmunq mapping + the three Chipmunq routers
    for router in ("basic",):
        def run(_, router=router):
            out, info = route_from_mapping(qc, backend, base, router, ps_inter)
            return _finish(router, out, info, t_base, qc, stim_ref, backend, {"initial_layout": dict(base)})
        attempt(f"chipmunq_{router}", run)

    budget = res["chipmunq_basic"]["routing_s"] if res["chipmunq_basic"]["ok"] else float("nan")

    def run_sabre_default(_):
        out, info = route_from_mapping(qc, backend, base, "sabre_fixed", ps_inter, seed=seed)
        return _finish("sabre", out, info, t_base, qc, stim_ref, backend, {"initial_layout": dict(base)})
    attempt("chipmunq_sabre_default", run_sabre_default)

    # Mapping ablations, all routed with Basic (the reference router), so only the mapping stage changes
    def basic_from(mapping, t_map, extra=None):
        out, info = route_from_mapping(qc, backend, mapping, "basic", ps_inter)
        return _finish("basic", out, info, t_map, qc, stim_ref, backend,
                       {**(extra or {}), "initial_layout": dict(mapping)})

    def run_no_partitioning(r):
        import time
        t0 = time.perf_counter()
        parts = kahypar_grid_partitions(qc, partitions, seed=42 + r)
        t_part = time.perf_counter() - t0
        m, t = chipmunq_mapping(qc, backend, parts)
        return basic_from(m, t + t_part)
    attempt("no_partitioning", run_no_partitioning, cfg["max_variant_retries"])

    def run_no_contraction(r):
        import time
        t0 = time.perf_counter()
        m = intra_chiplet_sabre_mapping(base, qc, backend, seed=seed + r)
        return basic_from(m, t_base + time.perf_counter() - t0)
    attempt("no_patch_contraction", run_no_contraction, cfg["max_variant_retries"])

    def run_no_sequencing(r):
        m, t = chipmunq_mapping(qc, backend, partitions, mapper_cls=RandomSequenceMapper,
                                mapper_kwargs={"seed": 1000 * seed + r})
        return basic_from(m, t)
    attempt("no_sequencing", run_no_sequencing, cfg["max_variant_retries"])

    def run_no_global(r):
        import time
        t0 = time.perf_counter()
        for s in range(100):  # skip the identity permutation
            m = permute_chiplets(base, backend, seed=1000 * seed + 100 * r + s)
            if m != base:
                break
        m, moved = repair_defects(m, backend)
        return basic_from(m, t_base + time.perf_counter() - t0, {"repaired_qubits": moved})
    attempt("no_global_mapping", run_no_global, cfg["max_variant_retries"])

    def run_lightsabre(_):
        # = ablation_utils.lightsabre (SabreLayout on the working qubits + SabreSwap), keeping the placement
        mapping, t_map = sabre_mapping(qc, backend, seed=seed)
        out, info = route_from_mapping(qc, backend, mapping, "sabre", 0.0, seed=seed)
        return _finish("sabre", out, info, t_map, qc, stim_ref, backend, {"initial_layout": dict(mapping)})
    attempt("lightsabre", run_lightsabre)

    # Full baselines, each in its own process with a wall-clock limit (mapping + routing in one call,
    # so the whole runtime is reported as mapping time)
    # Baselines. Serial compilation (compile_workers == 1): one after another, each with the whole machine,
    # as in a standalone run. Parallel compilation: concurrently (each is its own killable process, the
    # threads here only wait), so an instance takes as long as its slowest baseline.
    run = lambda m: run_with_timeout(compile_baseline, m, qc, backend, seed,  # noqa: E731
                                     timeout=cfg["baseline_timeout_s"])
    if cfg.get("compile_workers") == 1:
        futures = {}
        for m in BASELINES:
            f = Future()
            f.set_result(run(m))
            futures[m] = f
    else:
        with ThreadPoolExecutor(max_workers=len(BASELINES)) as pool:
            futures = {m: pool.submit(run, m) for m in BASELINES}
    for method in BASELINES:
        def run_baseline(_, method=method):
            out, runtime, status = futures[method].result()
            if status != "ok":
                raise RuntimeError("timeout" if status == TIMEOUT else f"{method} failed")
            routed, stim_s, layout = out
            return _finish(method, routed, {"routing_s": 0.0, "trials": 1}, runtime, qc, stim_ref, backend,
                           {"initial_layout": layout}, stim_circuit=stim.Circuit(stim_s))
        attempt(method, run_baseline)

    res["_budget_s"] = budget
    return res


def _compile_config(job):
    """One experiment configuration (backend kind, seed): find a valid backend, compile every variant.
    Module level so it can run in a spawned worker process."""
    bkind, n_def, s, cfg, circuit_str, partitions = job
    circuit = stim.Circuit(circuit_str)
    qc = StimCodeCircuit(stim_circuit=circuit).qc
    # Retry backend seeds until Chipmunq itself compiles a valid circuit (same protocol as
    # experiment_defective_qubits). Every retry is recorded.
    retries = []
    for attempt in range(cfg["max_backend_retries"]):
        bseed = 1000 * s + attempt
        backend = make_backend(cfg["k"], n_def, bseed, cfg)
        try:
            base, _ = chipmunq_mapping(qc, backend, partitions)
            out, _ = route_from_mapping(qc, backend, base, "basic", cfg["ps_inter"])
            check_routed(out, circuit, backend)
            break
        except Exception as e:
            retries.append(f"seed {bseed}: {type(e).__name__}: {str(e)[:120]}")
    else:
        raise RuntimeError(f"No valid {bkind} backend after {cfg['max_backend_retries']} tries")
    print(f"[{bkind} #{s}] backend seed {bseed} ({len(retries)} rejected)", flush=True)
    inst = compile_instance(qc, circuit, partitions, backend, n_def, cfg, seed=s)
    inst["_backend_seed"] = bseed
    inst["_rejected_backends"] = retries
    inst["_remote"] = symmetric_remote_noise(backend)
    print(f"[{bkind} #{s}] done", flush=True)
    return (bkind, s), inst


def compile_all(cfg: dict) -> dict:
    """Compile all configurations (backend kind x seed) in parallel worker processes.

    Each worker also starts up to len(BASELINES) baseline processes, so the default worker count is
    #CPUs / (1 + len(BASELINES)) and every compiler process gets a capped thread count. Baselines are
    slower under these caps; use ``compile_workers=1`` (--compile-workers 1) for the paper run, where
    configurations and baselines run one after another, each with the whole machine."""
    with quiet():
        circuit, partitions = get_tqec_cnot_rotated(distance_scale=cfg["k"], n1=1, n2=0)
    jobs = [(bkind, n_def, s, cfg, str(circuit), partitions)
            for bkind, n_def in cfg["defects"].items() for s in range(cfg["n_seeds"])]
    workers = cfg.get("compile_workers") or max(1, (os.cpu_count() or 1) // (1 + len(BASELINES)))
    workers = min(workers, len(jobs))
    print(f"Compiling {len(jobs)} configurations on {workers} worker(s)", flush=True)
    instances = {}
    if workers == 1:
        for job in jobs:
            key, inst = _compile_config(job)
            instances[key] = inst
    else:
        # Cap the threads of every compiler process (Qiskit's Rust passes otherwise start a pool of #CPUs
        # threads each), so workers x baselines processes do not oversubscribe the machine. The spawned
        # workers and their baseline processes inherit these variables.
        threads = max(1, (os.cpu_count() or 1) // (workers * (1 + len(BASELINES))))
        caps = ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        saved = {v: os.environ.get(v) for v in caps}
        os.environ.update({v: str(threads) for v in caps})
        print(f"  {threads} thread(s) per compiler process; use --compile-workers 1 for unconstrained "
              "baselines (e.g. when they time out)", flush=True)
        try:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                for key, inst in pool.map(_compile_config, jobs):
                    instances[key] = inst
        finally:
            for v, old in saved.items():
                if old is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = old
    # Same order as the serial loop (backend kind, then seed)
    order = [(b, s) for b in cfg["defects"] for s in range(cfg["n_seeds"])]
    return {"config": {k: v for k, v in cfg.items()}, "instances": {k: instances[k] for k in order}}


# --------------------------------------------------------------------------------------
# LER simulation
# --------------------------------------------------------------------------------------
def simulate(results: dict, cfg: dict):
    tasks = []
    for (bkind, s), inst in results["instances"].items():
        for v, rec in inst.items():
            if v.startswith("_") or not rec.get("ok"):
                continue
            base = stim.Circuit(rec["stim"])
            for p in cfg["ps"]:
                noisy = get_noise_model("modsi1000", None, p, None, remote=inst["_remote"]).noisy_circuit(base)
                tasks.append(sinter.Task(circuit=noisy, json_metadata={
                    "variant": v, "backend": bkind, "seed": s, "p": p, "d": 2 * cfg["k"] + 1}))
    print(f"Simulating {len(tasks)} tasks")
    return sinter.collect(
        num_workers=cfg["num_workers"] or max(1, os.cpu_count() // 2), tasks=tasks, decoders=["pymatching"],
        max_shots=cfg["num_shots"], max_errors=cfg["max_errors"], print_progress=True,
    )


# --------------------------------------------------------------------------------------
# Aggregation / plotting
# --------------------------------------------------------------------------------------
def _rc():
    plt.rcParams.update({
        "font.family": "serif", "axes.labelsize": FONTSIZE * 1.2, "font.size": FONTSIZE,
        "legend.fontsize": FONTSIZE - 3, "xtick.labelsize": FONTSIZE - 2, "ytick.labelsize": FONTSIZE - 2,
        "lines.linewidth": 1.5, "lines.markersize": 5, "lines.markeredgecolor": "black", "errorbar.capsize": 2,
    })


# --------------------------------------------------------------------------------------
# Paper figure style
# --------------------------------------------------------------------------------------
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


def _ci95(vals):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return np.nan, 0, 0
    m = vals.mean()
    if len(vals) == 1:
        return m, 0, 0
    boot = [np.mean(np.random.choice(vals, len(vals))) for _ in range(2000)]
    return m, m - np.percentile(boot, 2.5), np.percentile(boot, 97.5) - m


def ler_table(stats):
    """{(backend, variant, p): [ler per seed]} (zero-error points are kept as 0)."""
    t = defaultdict(list)
    for s in stats:
        md = s.json_metadata
        n = s.shots - s.discards
        if n > 0:
            t[(md["backend"], md["variant"], md["p"])].append(s.errors / n)
    return t


def present_variants(results, bkind):
    seen = {v for (b, _), inst in results["instances"].items() if b == bkind for v in inst if not v.startswith("_")}
    return [v for v, *_ in VARIANTS if v in seen]


def plot_compile_metrics(results, filename):
    _rc()
    bkinds = list(results["config"]["defects"])
    metrics = [("total_s", "Compile time [s]", True), ("2q_overhead", "2q-gate overhead", False),
               ("depth_overhead", "Depth overhead", False), ("inter_chiplet_2q", "Inter-chiplet 2q gates", False)]
    fig, axes = plt.subplots(len(metrics), len(bkinds), figsize=(WIDTH_FIGSIZE * 1.6, HEIGHT_FIGSIZE * 4.2),
                             squeeze=False, sharex="col")
    for j, b in enumerate(bkinds):
        vs = present_variants(results, b)
        insts = [inst for (bk, _), inst in results["instances"].items() if bk == b]
        for i, (key, ylabel, logy) in enumerate(metrics):
            ax = axes[i][j]
            for x, v in enumerate(vs):
                vals = [inst[v][key] for inst in insts if v in inst and inst[v].get("ok")]
                fails = sum(1 for inst in insts if v in inst and not inst[v].get("ok"))
                m, lo, hi = _ci95(vals)
                if key == "total_s":  # stacked: mapping | routing
                    mm = np.mean([inst[v]["mapping_s"] for inst in insts if v in inst and inst[v].get("ok")] or [0])
                    ax.bar(x, mm, color=COLOR[v], edgecolor="black", hatch=HATCH[v], alpha=0.5)
                    ax.bar(x, m - mm, bottom=mm, color=COLOR[v], edgecolor="black", hatch=HATCH[v],
                           yerr=[[lo], [hi]])
                else:
                    ax.bar(x, m, color=COLOR[v], edgecolor="black", hatch=HATCH[v], yerr=[[lo], [hi]])
                if fails:
                    ax.text(x, ax.get_ylim()[1] * 0.9 if not logy else 1, f"{fails}✗", ha="center", fontsize=7,
                            color="red")
            if logy:
                ax.set_yscale("log")
            ax.set_ylabel(ylabel if j == 0 else "")
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
            if i == 0:
                ax.set_title(f"{b} backend (budget B = {np.mean([inst['_budget_s'] for inst in insts]) * 1e3:.0f} ms)")
            ax.set_xticks(range(len(vs)))
            ax.set_xticklabels([LABEL[v] for v in vs], rotation=60, ha="right", fontsize=7)
    fig.text(0.01, 0.995, "Compile time: light = mapping, dark = routing. Lower is better ↓", va="top",
             fontweight="bold", color=plot_lib_color)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(filename, format="pdf")
    plt.close(fig)


def plot_ler_cross(results, stats, filename,
                   variants=("chipmunq_basic", "chipmunq_sabre_default", "no_global_mapping", "lightsabre", "seqc", "murali")):
    """The todo figure: Chipmunq map + SABRE routing vs SABRE map + noise routing, 2 backends."""
    _rc()
    t = ler_table(stats)
    bkinds = list(results["config"]["defects"])
    ps = sorted(results["config"]["ps"])
    fig, axes = plt.subplots(1, len(bkinds), figsize=(HEIGHT_FIGSIZE * 2.5 * len(bkinds), WIDTH_FIGSIZE * 0.55),
                             squeeze=False, sharey=True)
    markers = dict(zip(variants, "osx^vD"))
    handles = []
    for j, b in enumerate(bkinds):
        ax = axes[0][j]
        ax.plot(ps, ps, "--", color="#000000B3", linewidth=1)
        for v in variants:
            pts = [(p, *_ci95(t[(b, v, p)])) for p in ps if t.get((b, v, p))]
            if not pts:
                continue
            x, m, lo, hi = map(np.array, zip(*pts))
            h, = ax.plot(x, m, marker=markers[v], color=COLOR[v], label=LABEL[v])
            ax.fill_between(x, np.maximum(m - lo, 1e-9), m + hi, color=COLOR[v], alpha=0.2, edgecolor="none")
            if j == 0:
                handles.append(h)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Physical error rate")
        ax.set_title(f"{'a' if j == 0 else 'b'}) {b} backend", fontweight="bold", loc="left")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
    axes[0][0].set_ylabel("Logical error rate")
    fig.legend(handles=handles, loc="upper center", ncols=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.78, bottom=0.2, wspace=0.08)
    fig.savefig(filename, format="pdf")
    plt.close(fig)


def ler_ratio(t, b, v, ref, ps):
    """Geometric mean over p (and seeds) of LER_v / LER_ref, using points where both > 0."""
    r = []
    for p in ps:
        a, c = t.get((b, v, p), []), t.get((b, ref, p), [])
        if a and c and np.mean(a) > 0 and np.mean(c) > 0:
            r.append(np.mean(a) / np.mean(c))
    return float(np.exp(np.mean(np.log(r)))) if r else np.nan


# Tick labels for the paper figure (a): at 7 pt and 2/3 text width each bar pair gets ~0.3 in, so labels
# are two short lines; the one-word group names carry the context (spell the abbreviations out in the caption).
SHORT = {
    "chipmunq_basic": "Basic",
    "chipmunq_sabre_default": "SABRE-\nSWAP",
    "no_partitioning": "w/o\nPart.",
    "no_patch_contraction": "w/o\nContr.",
    "no_sequencing": "w/o\nSeq.",
    "no_global_mapping": "w/o\nMapper",
    "lightsabre": "Light-\nSABRE",
    "seqc": "SEQC",
    "olsq2": "OLSQ2",
    "murali": "Murali\net al.",
}
# Routing: Basic (reference) and SABRE-SWAP on the Chipmunq mapping. Mapping: one mapping stage removed (Basic routing).
# Baselines: full compilers (own mapping and routing).
RATIO_GROUPS = [
    ("Routing", ["chipmunq_basic", "chipmunq_sabre_default"]),
    ("Mapping", ["no_partitioning", "no_patch_contraction", "no_sequencing", "no_global_mapping"]),
    ("Baselines", ["lightsabre", "seqc", "olsq2", "murali"]),
]
GROUP_GAP = 0.4                       # extra space between groups [bar slots]


def plot_ler_ratio(results, stats, filename):
    """Paper figure a): LER of each variant relative to Chipmunq with Basic routing, per backend."""
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, LogLocator

    _fonts()
    t = ler_table(stats)
    ps = sorted(results["config"]["ps"])
    bkinds = list(results["config"]["defects"])
    fig, ax = _panel(TEXT_WIDTH_IN * 2 / 3)

    # x positions with a gap between groups; skip variants that were never compiled
    present = {v for b in bkinds for v in present_variants(results, b)}
    xs, order, spans, x = [], [], [], 0.0
    for name, vs in RATIO_GROUPS:
        vs = [v for v in vs if v in present]
        if not vs:
            continue
        start = x
        for v in vs:
            xs.append(x)
            order.append(v)
            x += 1
        spans.append((name, start, x - 1))
        x += GROUP_GAP
    xs = np.array(xs)

    w = 0.8 / len(bkinds)
    for j, b in enumerate(bkinds):
        vals = [ler_ratio(t, b, v, REFERENCE, ps) for v in order]
        bx = xs + (j - (len(bkinds) - 1) / 2) * w
        ax.bar(bx, vals, w, color=[COLOR[v] for v in order],
               edgecolor="black", linewidth=0.4, hatch="" if j == 0 else "////", label=b)
        # No result: T/O if every run of this variant hit the time limit, N/A otherwise
        for xb, v, val in zip(bx, order, vals):
            if np.isfinite(val):
                continue
            recs = [inst[v] for (bk, _), inst in results["instances"].items() if bk == b and v in inst]
            label = "T/O" if recs and all("timeout" in str(r.get("error", "")) for r in recs) else "N/A"
            ax.text(xb, 0.04, label, transform=ax.get_xaxis_transform(), rotation=90, ha="center",
                    va="bottom", fontsize=FONT_PT - 1.5)

    ax.axhline(1, color="black", linestyle="--", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[v] for v in order], linespacing=0.95)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(xs[0] - 0.55, xs[-1] + 0.55)
    ax.set_ylabel("Relative LER")  # LER_variant / LER_Chipmunq(Basic) at p = ps[0] (caption)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda y, _: ""))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.grid(True, axis="y", which="major", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    # Group names inside the axes (top) and separators between the groups
    for i, (name, a, b) in enumerate(spans):
        ax.text((a + b) / 2, 0.97, name, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontweight="bold")
        if i:
            ax.axvline(a - (1 + GROUP_GAP) / 2, color="grey", linewidth=0.6, alpha=0.6)
    lo, hi = ax.get_ylim()
    # start below 1 so the reference bars (= 1) stay visible, headroom for group names and legend (log axis)
    ax.set_ylim(min(lo, 0.6), hi * 2.5)

    # Backend legend (hatching = backend; colour = variant), above the axes right of the title.
    # Not needed when only one backend is shown.
    if len(bkinds) > 1:
        ax.legend(handles=[Patch(facecolor="white", edgecolor="black", linewidth=0.4,
                                 hatch="" if j == 0 else "////", label=f"{b} backend")
                           for j, b in enumerate(bkinds)],
                  loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=1, frameon=False, handlelength=1.2,
                  handletextpad=0.35, labelspacing=0.15, borderaxespad=0.1)
    _title(fig, ax, "a) Contribution of each stage")
    fig.savefig(filename, format="pdf")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Diagnostic: mapping and routing paths of every method
# --------------------------------------------------------------------------------------
PATCH_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]


def _swap_counts(stim_str: str) -> Counter:
    """SWAPs per coupler (unordered physical qubit pair) in a compiled circuit."""
    c = Counter()
    for inst in stim.Circuit(stim_str).flattened():
        if inst.name == "SWAP":
            t = [x.value for x in inst.targets_copy()]
            c.update(tuple(sorted(e)) for e in zip(t[::2], t[1::2]))
    return c


def plot_layouts(results: dict, out_dir: Path) -> None:
    """One figure per configuration (backend kind, seed), one panel per method: qubits at their initial
    position coloured by patch (partition), couplers coloured by the number of SWAPs routed over them,
    inter-chiplet links dashed, defective qubits as red crosses."""
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    _rc()
    cfg = results["config"]
    from experiments.exp_utils.ablation_utils import ANNOTATIONS

    with quiet():
        circuit, partitions = get_tqec_cnot_rotated(distance_scale=cfg["k"], n1=1, n2=0)
    qc = StimCodeCircuit(stim_circuit=circuit).qc
    # only virtual qubits the circuit acts on (the patch specification also lists unused indices)
    used = {qc.find_bit(q).index for ins in qc.data if ins.operation.name not in ANNOTATIONS | {"barrier"}
            for q in ins.qubits}
    patch_of = {int(q): i for i, part in enumerate(partitions) for q in part["indices"]}
    out_dir.mkdir(parents=True, exist_ok=True)

    for (bkind, seed), inst in results["instances"].items():
        backend = make_backend(cfg["k"], cfg["defects"][bkind], inst["_backend_seed"], cfg)
        xy = np.asarray(generate_coordinates(backend), dtype=float)
        edges = {tuple(sorted(map(int, e))) for e in backend.coupling_map.get_edges()}
        links = {tuple(sorted(e)) for e in inst["_remote"]}
        defective = sorted(set(backend.all_defective_qubits))
        variants = [v for v, *_ in VARIANTS if v in inst]
        swaps = {v: _swap_counts(inst[v]["stim"]) for v in variants if inst[v].get("ok")}
        vmax = max([max(c.values()) for c in swaps.values() if c] or [1])
        norm = LogNorm(vmin=1, vmax=max(vmax, 2))
        cmap = plt.cm.inferno_r

        ncol = min(5, len(variants))
        nrow = -(-len(variants) // ncol)
        span = np.ptp(xy, axis=0) + 1
        w = 3.6
        fig, axes = plt.subplots(nrow, ncol, figsize=(w * ncol, w * nrow * span[1] / span[0] + 0.6),
                                 squeeze=False)
        for ax in axes.flat:
            ax.set_axis_off()
        for ax, v in zip(axes.flat, variants):
            rec = inst[v]
            for a, b in edges:  # hardware couplers
                ax.plot(*xy[[a, b]].T, color="#7B2CBF" if (a, b) in links else "#D0D0D0",
                        ls="--" if (a, b) in links else "-", lw=0.6 if (a, b) in links else 0.4, zorder=1)
            ax.scatter(*xy.T, s=3, color="#E0E0E0", zorder=2)
            if not rec.get("ok"):
                msg = "T/O" if "timeout" in str(rec.get("error", "")) else "failed"
                ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center", fontsize=14,
                        fontweight="bold", color="#555555")
            else:
                for (a, b), n in swaps[v].items():  # routing paths
                    ax.plot(*xy[[a, b]].T, color=cmap(norm(n)), lw=0.8 + 1.6 * norm(n), zorder=3,
                            solid_capstyle="round")
                lay = rec.get("initial_layout") or {}
                if lay:  # mapping: initial position of every virtual qubit, coloured by patch
                    vq = [q for q in lay if int(q) in used]
                    ph = [lay[q] for q in vq]
                    cols = [PATCH_COLORS[patch_of.get(int(q), -1) % len(PATCH_COLORS)] if int(q) in patch_of
                            else "#000000" for q in vq]
                    ax.scatter(*xy[ph].T, s=9, c=cols, edgecolors="black", linewidths=0.2, zorder=4)
                ax.set_title(f"{LABEL[v]}\n2q ovh {rec['2q_overhead']}, link 2q {rec['inter_chiplet_2q']}, "
                             f"SWAPs {sum(swaps[v].values())}", fontsize=7)
            if defective:
                ax.scatter(*xy[defective].T, marker="x", s=18, color="red", linewidths=1.0, zorder=5)
            if not rec.get("ok"):
                ax.set_title(LABEL[v], fontsize=7)
            ax.set_aspect("equal")

        handles = [Line2D([], [], marker="o", ls="none", color=PATCH_COLORS[i % len(PATCH_COLORS)],
                          markeredgecolor="black", markeredgewidth=0.3, label=f"patch {i}")
                   for i in range(len(partitions))]
        handles += [Line2D([], [], color="#7B2CBF", ls="--", label="inter-chiplet link"),
                    Line2D([], [], marker="x", ls="none", color="red", label="defective qubit")]
        fig.legend(handles=handles, loc="lower center", ncols=len(handles), frameon=False, fontsize=8)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.01, label="SWAPs on coupler")
        fig.suptitle(f"Mapping (initial positions) and routing (SWAPs): {bkind} backend, seed {seed}",
                     fontweight="bold")
        fig.savefig(out_dir / f"layouts_{bkind}_{seed}.pdf", format="pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"layouts_{bkind}_{seed}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def summary_table(results, stats) -> str:
    t = ler_table(stats) if stats else {}
    ps = sorted(results["config"]["ps"])
    lines = []
    for b in results["config"]["defects"]:
        insts = [inst for (bk, _), inst in results["instances"].items() if bk == b]
        budget = np.mean([inst["_budget_s"] for inst in insts]) * 1e3
        rejected = sum(len(inst["_rejected_backends"]) for inst in insts)
        lines += [f"\n### {b} backend  (budget B = {budget:.1f} ms, {rejected} backend seeds rejected "
                  "because Chipmunq itself failed)\n",
                  "| variant | ok | map [ms] | route [ms] | trials | 2q ovh | depth ovh | inter-chip 2q | "
                  "LER / Chipmunq |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for v in present_variants(results, b):
            recs = [inst[v] for inst in insts if v in inst]
            ok = [r for r in recs if r.get("ok")]
            def f(k):  # mean over successful runs; missing / None values (e.g. baseline trials) -> nan
                vals = [np.nan if r.get(k) is None else r[k] for r in ok]
                return np.nanmean(vals) if vals and not np.all(np.isnan(vals)) else np.nan
            ratio = ler_ratio(t, b, v, REFERENCE, ps) if t else np.nan
            lines.append(f"| {LABEL[v]} | {len(ok)}/{len(recs)} | {f('mapping_s') * 1e3:.0f} | "
                         f"{f('routing_s') * 1e3:.0f} | {f('trials'):.0f} | {f('2q_overhead'):.0f} | "
                         f"{f('depth_overhead'):.0f} | {f('inter_chiplet_2q'):.0f} | {ratio:.2f} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
def _only_backends(results: dict, stats, keep: list[str]):
    """Restrict compiled results and LER stats to the given backend kinds (for plotting)."""
    res = dict(results)
    res["config"] = dict(results["config"], defects={b: n for b, n in results["config"]["defects"].items()
                                                     if b in keep})
    res["instances"] = {k: v for k, v in results["instances"].items() if k[0] in keep}
    return res, [s_ for s_ in stats if s_.json_metadata["backend"] in keep]


def run_mapping_routing_ablation(reproduce: bool = True, quick: bool = False, clean_only: bool = False,
                                 compile_workers: int | None = None) -> None:
    """clean_only: compile/simulate (and plot) only the clean backend. Results go to separate files with
    suffix ``_clean``; with --plot-only, a full run's results are used (restricted to clean) if no
    clean-only results exist."""
    cfg = QUICK_CONFIG if quick else FULL_CONFIG
    if clean_only:
        cfg = dict(cfg, defects={"clean": cfg["defects"]["clean"]})
    if compile_workers:
        cfg = dict(cfg, compile_workers=compile_workers)
    out = OUTPUT_DIR / ("quick" if quick else "")
    out.mkdir(parents=True, exist_ok=True)
    base_tag = f"k{cfg['k']}_pinter{cfg['ps_inter']}"
    tag = base_tag + ("_clean" if clean_only else "")
    if reproduce:
        results = compile_all(cfg)
        with open(out / f"compile_{tag}.pkl", "wb") as f:
            pickle.dump(results, f)
        stats = simulate(results, cfg)
        with open(out / f"ler_{tag}.pkl", "wb") as f:
            pickle.dump(stats, f)
    src = tag if (out / f"compile_{tag}.pkl").exists() else base_tag
    with open(out / f"compile_{src}.pkl", "rb") as f:
        results = pickle.load(f)
    with open(out / f"ler_{src}.pkl", "rb") as f:
        stats = pickle.load(f)
    if clean_only:
        results, stats = _only_backends(results, stats, ["clean"])

    plot_compile_metrics(results, out / f"ablation_compile_{tag}.pdf")
    plot_ler_cross(results, stats, out / f"ablation_ler_cross_{tag}.pdf")
    plot_ler_ratio(results, stats, out / f"ablation_ler_ratio_{tag}.pdf")
    plot_layouts(results, out / f"layouts_{tag}")
    table = summary_table(results, stats)
    (out / f"ablation_summary_{tag}.md").write_text(table)
    print(table)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="d=3, 1 seed, 2 error rates, few shots")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--clean-only", action="store_true",
                    help="only the clean backend (run and plot); with --plot-only, also replots a full run")
    ap.add_argument("--compile-workers", type=int, default=None,
                    help="configurations compiled in parallel (default: #CPUs / 4, 1 = serial)")
    a = ap.parse_args()
    try:
        run_mapping_routing_ablation(reproduce=not a.plot_only, quick=a.quick, clean_only=a.clean_only,
                                     compile_workers=a.compile_workers)
    except Exception:
        traceback.print_exc()
        raise