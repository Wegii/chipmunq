"""Evaluate Chipmunq's mapping and routing independently (ablation).

Question: how much of Chipmunq's LER / overhead advantage comes from each stage?

All variants of one backend instance are compiled for the *same* circuit on the *same*
BackendChipletV2 object (identical defects and link noise); only one stage changes.

    Variant                   mapping                                   routing
    ------------------------  ----------------------------------------  --------------------
    chipmunq_tradeoff         Chipmunq (full)                           Tradeoff   <- reference
    chipmunq_focus            Chipmunq                                  Focus
    chipmunq_basic            Chipmunq                                  Basic      (-noise-aware routing)
    chipmunq_sabre            Chipmunq                                  SABRE-SWAP, budget-matched  [todo 2]
    chipmunq_sabre_default    Chipmunq                                  SABRE-SWAP, Qiskit default trials
    no_partitioning           KaHyPar blocks instead of patch IR        Tradeoff
    no_patch_contraction      qubit-level SABRE inside assigned chiplet Tradeoff
    no_sequencing             random patch order instead of BFS         Tradeoff
    no_global_mapping         random patch->chiplet assignment          Tradeoff
    no_defect_aware           placement blind to defects + repair       Tradeoff   (defective backend only)
    sabre_tradeoff            SabreLayout (qubit level)                 Tradeoff               [todo 1]
    lightsabre                SabreLayout                               SABRE-SWAP (default)

Equal compilation-time budgets: Basic, Focus and Tradeoff are deterministic single-pass
routers, so their budget is their runtime. The budget B of a backend instance is the
largest of the three; ``chipmunq_sabre`` gets exactly B, spent on independent SabreSwap
restarts, keeping the one with the fewest SWAPs (SABRE's own criterion: it gets no noise
information). ``chipmunq_sabre_default`` shows SABRE with its standard trial count.

Reported per variant and backend (clean / defective): mapping + routing runtime, LER vs p,
2q-gate overhead (SWAP = 3), depth overhead, inter-chiplet 2q gates, compile failures.

Noise: circuit-level ``modsi1000`` with per-link inter-chiplet noise, applied
orientation-independently (see ``symmetric_remote_noise``).

Paper figure (a, next to the cosmic-ray experiment b): ``ablation_ler_ratio_*.pdf``, drawn at its printed
size (2/3 of the text width, Fig. 10 style) -- include it at its natural width, no scaling.

Usage:
    python experiments/qec_exps/experiment_mapping_routing_ablation.py            # full
    python experiments/qec_exps/experiment_mapping_routing_ablation.py --quick    # smoke test
    python experiments/qec_exps/experiment_mapping_routing_ablation.py --plot-only
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import traceback
from collections import defaultdict
from pathlib import Path

sys.path.append(os.path.join(os.getcwd(), "."))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sinter  # noqa: E402
import stim  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

from experiments.exp_utils.ablation_utils import (  # noqa: E402
    DefectBlindMapper,
    RandomSequenceMapper,
    check_routed,
    chipmunq_mapping,
    intra_chiplet_sabre_mapping,
    kahypar_grid_partitions,
    lightsabre,
    overhead_stats,
    permute_chiplets,
    quiet,
    repair_defects,
    route_from_mapping,
    sabre_mapping,
    symmetric_remote_noise,
)
from experiments.exp_utils.circuit_generator import get_tqec_cnot_rotated  # noqa: E402
from experiments.exp_utils.circuit_noise import get_noise_model  # noqa: E402
from experiments.exp_utils.simulation_utils import run_sinter_simulation  # noqa: E402
from experiments.exp_utils.utils import FONTSIZE, HEIGHT_FIGSIZE, WIDTH_FIGSIZE, plot_lib_color  # noqa: E402
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit  # noqa: E402
from qeccm.backends.BackendChipletV2 import BackendChipletV2  # noqa: E402

OUTPUT_DIR = Path("experiments/evaluation/mapping_routing_ablation")

# (key, label, colour, hatch) -- order is the plotting order
VARIANTS = [
    ("chipmunq_tradeoff", "Chipmunq (Tradeoff)", "#2A5687", ""),
    ("chipmunq_focus", "Chipmunq (Focus)", "#5E97CC", ""),
    ("chipmunq_basic", "Chipmunq (Basic)", "#A7D9ED", ""),
    ("chipmunq_sabre", "Chipmunq map + SABRE-SWAP (equal budget)", "#F3C98B", "//"),
    ("chipmunq_sabre_default", "Chipmunq map + SABRE-SWAP (default)", "#E6A96B", "\\\\"),
    ("no_partitioning", "- patch-IR partitioning", "#B2D8B2", ".."),
    ("no_patch_contraction", "- patch contraction", "#8CC08C", "xx"),
    ("no_sequencing", "- sequencing", "#C9B7E3", "--"),
    ("no_global_mapping", "- global mapping", "#A68BCB", "++"),
    ("no_defect_aware", "- defect-aware placement", "#D5D5D5", "oo"),
    ("sabre_tradeoff", "SABRE map + Tradeoff routing", "#E68A5C", "//"),
    ("lightsabre", "LightSABRE", "lightcoral", "o"),
]
LABEL = {k: l for k, l, _, _ in VARIANTS}
COLOR = {k: c for k, _, c, _ in VARIANTS}
HATCH = {k: h for k, _, _, h in VARIANTS}
REFERENCE = "chipmunq_tradeoff"

FULL_CONFIG = dict(
    k=3,  # distance_scale; d = 2k + 1
    ps=list(np.logspace(-4, -1, 10)),
    ps_inter=1e-3,
    rfactor=100,  # "high variance" link noise, as in Fig. 9
    n_seeds=4,
    defects={"clean": 0, "defective": 2},  # defective qubits per chiplet
    max_backend_retries=20,
    max_variant_retries=5,
    num_shots=10_000_000,
    max_errors=5_000,
    num_workers=None,
)
QUICK_CONFIG = dict(FULL_CONFIG, k=1, ps=[2e-3, 5e-3], n_seeds=1, defects={"clean": 0, "defective": 1},
                    num_shots=20_000, max_errors=200)

BACKEND_SIZES = {1: ((2, 2, 11, 6), 5), 2: ((2, 2, 15, 8), 7), 3: ((2, 2, 19, 10), 9), 4: ((2, 2, 23, 12), 11)}


def make_backend(k: int, n_defects: int, seed: int, cfg: dict) -> BackendChipletV2:
    size, nic = BACKEND_SIZES[k]
    return BackendChipletV2(
        size=size, n_inter=nic, connectivity="nn", topology="rotated_grid",
        inter_chiplet_noise=cfg["ps_inter"], inter_chiplet_amplification=1,
        inter_chiplet_rfactor=cfg["rfactor"], inter_chiplet_noise_type="random",
        num_defective_qubits=n_defects, rng_seed=seed,
    )


# --------------------------------------------------------------------------------------
# Compilation
# --------------------------------------------------------------------------------------
def _finish(name, routed, route_info, t_map, qc, stim_ref, backend, extra=None):
    stim_c = check_routed(routed, stim_ref, backend)
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
    for router in ("tradeoff", "focus", "basic"):
        def run(_, router=router):
            out, info = route_from_mapping(qc, backend, base, router, ps_inter)
            return _finish(router, out, info, t_base, qc, stim_ref, backend)
        attempt(f"chipmunq_{router}", run)

    budget = max(res[f"chipmunq_{r}"]["routing_s"] for r in ("tradeoff", "focus", "basic") if res[f"chipmunq_{r}"]["ok"])

    def run_sabre_budget(_):
        out, info = route_from_mapping(qc, backend, base, "sabre", ps_inter, seed=seed, budget_s=budget)
        return _finish("sabre", out, info, t_base, qc, stim_ref, backend)
    attempt("chipmunq_sabre", run_sabre_budget)

    def run_sabre_default(_):
        out, info = route_from_mapping(qc, backend, base, "sabre", ps_inter, seed=seed)
        return _finish("sabre", out, info, t_base, qc, stim_ref, backend)
    attempt("chipmunq_sabre_default", run_sabre_default)

    # Mapping ablations, all routed with Tradeoff
    def tradeoff_from(mapping, t_map, extra=None):
        out, info = route_from_mapping(qc, backend, mapping, "tradeoff", ps_inter)
        return _finish("tradeoff", out, info, t_map, qc, stim_ref, backend, extra)

    def run_no_partitioning(r):
        import time
        t0 = time.perf_counter()
        parts = kahypar_grid_partitions(qc, partitions, seed=42 + r)
        t_part = time.perf_counter() - t0
        m, t = chipmunq_mapping(qc, backend, parts)
        return tradeoff_from(m, t + t_part)
    attempt("no_partitioning", run_no_partitioning, cfg["max_variant_retries"])

    def run_no_contraction(r):
        import time
        t0 = time.perf_counter()
        m = intra_chiplet_sabre_mapping(base, qc, backend, seed=seed + r)
        return tradeoff_from(m, t_base + time.perf_counter() - t0)
    attempt("no_patch_contraction", run_no_contraction, cfg["max_variant_retries"])

    def run_no_sequencing(r):
        m, t = chipmunq_mapping(qc, backend, partitions, mapper_cls=RandomSequenceMapper,
                                mapper_kwargs={"seed": 1000 * seed + r})
        return tradeoff_from(m, t)
    attempt("no_sequencing", run_no_sequencing, cfg["max_variant_retries"])

    def run_no_global(r):
        import time
        t0 = time.perf_counter()
        for s in range(100):  # skip the identity permutation
            m = permute_chiplets(base, backend, seed=1000 * seed + 100 * r + s)
            if m != base:
                break
        m, moved = repair_defects(m, backend)
        return tradeoff_from(m, t_base + time.perf_counter() - t0, {"repaired_qubits": moved})
    attempt("no_global_mapping", run_no_global, cfg["max_variant_retries"])

    if n_defects > 0:
        def run_no_defect(_):
            import time
            m, t = chipmunq_mapping(qc, backend, partitions, mapper_cls=DefectBlindMapper)
            t0 = time.perf_counter()
            m, moved = repair_defects(m, backend)
            return tradeoff_from(m, t + time.perf_counter() - t0, {"repaired_qubits": moved})
        attempt("no_defect_aware", run_no_defect)

    def run_sabre_tradeoff(_):
        m, t = sabre_mapping(qc, backend, seed=seed)
        return tradeoff_from(m, t)
    attempt("sabre_tradeoff", run_sabre_tradeoff)

    def run_lightsabre(_):
        out, info = lightsabre(qc, backend, seed=seed)
        return _finish("sabre", out, info, info["mapping_s"], qc, stim_ref, backend)
    attempt("lightsabre", run_lightsabre)

    res["_budget_s"] = budget
    return res


def compile_all(cfg: dict) -> dict:
    with quiet():
        circuit, partitions = get_tqec_cnot_rotated(distance_scale=cfg["k"], n1=1, n2=0)
    qc = StimCodeCircuit(stim_circuit=circuit).qc
    results = {"config": {k: v for k, v in cfg.items()}, "instances": {}}
    for bkind, n_def in cfg["defects"].items():
        for s in range(cfg["n_seeds"]):
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
            print(f"[{bkind} #{s}] backend seed {bseed} ({len(retries)} rejected)")
            inst = compile_instance(qc, circuit, partitions, backend, n_def, cfg, seed=s)
            inst["_backend_seed"] = bseed
            inst["_rejected_backends"] = retries
            inst["_remote"] = symmetric_remote_noise(backend)
            results["instances"][(bkind, s)] = inst
    return results


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
                   variants=("chipmunq_tradeoff", "chipmunq_basic", "chipmunq_sabre", "sabre_tradeoff", "lightsabre")):
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
    "chipmunq_focus": "Focus",
    "chipmunq_basic": "Basic",
    "chipmunq_sabre": "SABRE\n(equal)",
    "chipmunq_sabre_default": "SABRE\n(Qiskit)",
    "no_partitioning": "w/o\nPart.",
    "no_patch_contraction": "w/o\nContr.",
    "no_sequencing": "w/o\nSeq.",
    "no_global_mapping": "w/o\nGlobal",
    "no_defect_aware": "w/o\nDefect",
    "sabre_tradeoff": "SABRE\n+Trade.",
    "lightsabre": "Light-\nSABRE",
}
# Routing: other routers on the Chipmunq mapping. Mapping: one mapping stage removed (Tradeoff routing).
# Baselines: SABRE mapping (+ Tradeoff routing) and LightSABRE.
RATIO_GROUPS = [
    ("Routing", ["chipmunq_focus", "chipmunq_basic", "chipmunq_sabre", "chipmunq_sabre_default"]),
    ("Mapping", ["no_partitioning", "no_patch_contraction", "no_sequencing", "no_global_mapping",
                 "no_defect_aware"]),
    ("Baselines", ["sabre_tradeoff", "lightsabre"]),
]
GROUP_GAP = 0.4                       # extra space between groups [bar slots]


def plot_ler_ratio(results, stats, filename):
    """Paper figure a): LER of each variant relative to full Chipmunq (Tradeoff), per backend."""
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
        ax.bar(xs + (j - (len(bkinds) - 1) / 2) * w, vals, w, color=[COLOR[v] for v in order],
               edgecolor="black", linewidth=0.4, hatch="" if j == 0 else "////", label=b)

    ax.axhline(1, color="black", linestyle="--", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[v] for v in order], linespacing=0.95)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(xs[0] - 0.55, xs[-1] + 0.55)
    ax.set_ylabel("Relative LER")  # LER_variant / LER_Chipmunq(Tradeoff), geometric mean over p (caption)
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
    ax.set_ylim(lo, hi * 2.5)              # headroom for the group names and the legend (log axis)

    # Backend legend (hatching = backend; colour = variant), below the first group name
    ax.legend(handles=[Patch(facecolor="white", edgecolor="black", linewidth=0.4,
                             hatch="" if j == 0 else "////", label=f"{b} backend")
                       for j, b in enumerate(bkinds)],
              loc="upper left", bbox_to_anchor=(0.0, 0.86), ncols=1, frameon=False, handlelength=1.2,
              handletextpad=0.35, labelspacing=0.2, borderaxespad=0.2)
    _title(fig, ax, "a) Contribution of each stage")
    fig.savefig(filename, format="pdf")
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
            f = lambda k: np.mean([r[k] for r in ok]) if ok else np.nan  # noqa: E731
            ratio = ler_ratio(t, b, v, REFERENCE, ps) if t else np.nan
            lines.append(f"| {LABEL[v]} | {len(ok)}/{len(recs)} | {f('mapping_s') * 1e3:.0f} | "
                         f"{f('routing_s') * 1e3:.0f} | {f('trials'):.0f} | {f('2q_overhead'):.0f} | "
                         f"{f('depth_overhead'):.0f} | {f('inter_chiplet_2q'):.0f} | {ratio:.2f} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
def run_mapping_routing_ablation(reproduce: bool = True, quick: bool = False) -> None:
    cfg = QUICK_CONFIG if quick else FULL_CONFIG
    out = OUTPUT_DIR / ("quick" if quick else "")
    out.mkdir(parents=True, exist_ok=True)
    tag = f"k{cfg['k']}_pinter{cfg['ps_inter']}"
    if reproduce:
        results = compile_all(cfg)
        with open(out / f"compile_{tag}.pkl", "wb") as f:
            pickle.dump(results, f)
        stats = simulate(results, cfg)
        with open(out / f"ler_{tag}.pkl", "wb") as f:
            pickle.dump(stats, f)
    with open(out / f"compile_{tag}.pkl", "rb") as f:
        results = pickle.load(f)
    with open(out / f"ler_{tag}.pkl", "rb") as f:
        stats = pickle.load(f)

    plot_compile_metrics(results, out / f"ablation_compile_{tag}.pdf")
    plot_ler_cross(results, stats, out / f"ablation_ler_cross_{tag}.pdf")
    plot_ler_ratio(results, stats, out / f"ablation_ler_ratio_{tag}.pdf")
    table = summary_table(results, stats)
    (out / f"ablation_summary_{tag}.md").write_text(table)
    print(table)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="d=3, 1 seed, 2 error rates, few shots")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    try:
        run_mapping_routing_ablation(reproduce=not a.plot_only, quick=a.quick)
    except Exception:
        traceback.print_exc()
        raise