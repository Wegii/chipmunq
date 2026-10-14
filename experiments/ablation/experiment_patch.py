"""Partition preservation: do the compilers keep the given patches together as compact, separate regions?

For every circuit stored by experiment_statistics (CIRCUIT_DIR: one file per benchmark, distance and method), the
qubits' physical positions are clustered *without* knowing the given partitions, and the found clusters are
compared with the given data patches:

  1. Positions: the method's initial placement, then the placement after replaying its SWAPs up to evenly spaced
     checkpoints of the compiled circuit (N_CHECKPOINTS; the same time grid for every method, since some
     compilers drop the Stim TICK barriers).
  2. Distances: hop distance between the occupied physical qubits in the backend's coupling graph (an
     inter-chiplet link counts as one hop, gaps between chiplets do not distort the distances).
  3. Clusters: agglomerative clustering (average linkage) on these distances, cut at k = number of data patches.
  4. Score: Adjusted Rand Index (ARI) between the found clusters and the given patches. 1 = every patch is its own
     compact region, ~0 = no better than a random placement.

Only qubits the circuit acts on and only data patches are scored: lattice-surgery pipes / merge strips
(partition types ending in "_ancilla") are small and sit between patches by design, so a spatial clustering
would absorb them into a neighbour even for a perfect placement.

Reported per (benchmark, distance, method): ARI of the initial placement, mean and minimum over the checkpoints.
A random placement of the same qubits on the same occupied positions gives the chance level.

Run from the repo root (after experiment_statistics stored its circuits):
    python experiments/scalability_exps/partition_preservation.py [--distances 3 5 7] [--jobs N]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import sys
from pathlib import Path

sys.path.append(os.path.join(os.getcwd(), "."))

import numpy as np

import experiments.scalability_exps.experiment_statistics as S

N_CHECKPOINTS = 20                       # time points after the initial placement
RANDOM_SEEDS = 5                         # random placements for the chance level
OUT_DIR = Path("experiments/evaluation/ablation")
OUT_PKL = OUT_DIR / "partition_preservation.pkl"
OUT_MD = OUT_DIR / "partition_preservation.md"
OUT_PDF = OUT_DIR / "partition_preservation.pdf"
OUT_LEGEND = OUT_DIR / "partition_preservation_legend.pdf"
NO_RESULT_BAR = 0.15  # height of the hollow bar of a method without a compiled circuit (T/O or N/A)


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------
def used_virtual_qubits(qc) -> set[int]:
    """Virtual qubits the (uncompiled) circuit acts on with real operations."""
    return {qc.find_bit(q).index for ins in qc.data if ins.operation.name not in S.NON_OPS for q in ins.qubits}


def patch_labels(partitions: list[dict], used: set[int]) -> dict[int, int]:
    """virtual qubit -> data-patch id (pipes / merge strips, i.e. *_ancilla partitions, are left out)."""
    labels = {}
    for i, p in enumerate(partitions):
        if str(p.get("type", "")).endswith("_ancilla"):
            continue
        for q in p["indices"]:
            if int(q) in used:
                labels[int(q)] = i
    return labels


def initial_layout(circ) -> list[int]:
    """virtual qubit index -> physical qubit of the method's initial placement."""
    if circ.layout is not None:
        return [int(p) for p in circ.layout.initial_index_layout(filter_ancillas=True)]
    md = circ.metadata or {}
    for key in ("noise_adaptive_layout", "olsq2_initial_layout"):
        if key in md:
            return [int(p) for p in md[key]]
    return list(range(circ.num_qubits))  # no placement recorded: trivial layout


# --------------------------------------------------------------------------------------
# Clustering and score
# --------------------------------------------------------------------------------------
def hop_distances(backend, positions: list[int]) -> np.ndarray:
    """Pairwise hop distances between ``positions`` in the backend's (undirected) coupling graph."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import shortest_path

    n = backend.num_qubits_total if hasattr(backend, "num_qubits_total") else backend.num_qubits
    edges = np.array(list(backend.coupling_map.get_edges()))
    adj = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    d = shortest_path(adj, directed=False, unweighted=True, indices=positions)
    return d[:, positions]


def spatial_ari(dist: np.ndarray, truth: list[int], k: int) -> float:
    """ARI between the given patch labels and an average-linkage clustering of ``dist`` cut at k clusters."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics import adjusted_rand_score

    d = np.where(np.isfinite(dist), dist, dist[np.isfinite(dist)].max() * 10 + 1)  # disconnected: far apart
    np.fill_diagonal(d, 0)
    found = fcluster(linkage(squareform(d, checks=False), method="average"), t=k, criterion="maxclust")
    return float(adjusted_rand_score(truth, found))


def score_circuit(job) -> tuple:
    """ARI over time for one compiled circuit. Returns (key, ks, method, {"ari": [...], "random": [...]})."""
    key, ks, method, labels, partitions = job
    circ = S.load_circuit(S.circuit_path(key, ks, method))
    backend, _ = S.make_backend(key, ks, partitions)
    virt = sorted(labels)
    truth = [labels[v] for v in virt]
    k = len(set(truth))
    lay = initial_layout(circ)
    phys_of = {v: lay[v] for v in virt}
    where = {p: v for v, p in enumerate(lay)}  # physical -> virtual (all placed qubits)

    def ari_now() -> float:
        pos = [phys_of[v] for v in virt]
        return spatial_ari(hop_distances(backend, pos), truth, k)

    data = circ.data
    marks = set(np.linspace(0, len(data), N_CHECKPOINTS + 1, dtype=int)[1:].tolist())
    aris = [ari_now()]  # initial placement
    for i, ins in enumerate(data, 1):
        if ins.operation.name == "swap":  # the qubits on physical a and b exchange places
            a, b = (circ.find_bit(q).index for q in ins.qubits)
            va, vb = where.pop(a, None), where.pop(b, None)
            if va is not None:
                where[b] = va
                if va in phys_of:
                    phys_of[va] = b
            if vb is not None:
                where[a] = vb
                if vb in phys_of:
                    phys_of[vb] = a
        if i in marks:
            aris.append(ari_now())

    # Chance level: the same qubits randomly assigned to the occupied initial positions
    pos0 = [lay[v] for v in virt]
    dist0 = hop_distances(backend, pos0)
    rng = np.random.default_rng(0)
    rand = [spatial_ari(dist0, list(rng.permutation(truth)), k) for _ in range(RANDOM_SEEDS)]
    return key, ks, method, {"ari": aris, "random": rand, "k": k, "n": len(virt)}


# --------------------------------------------------------------------------------------
# Driver, table, plot
# --------------------------------------------------------------------------------------
def run(ks_list: list[int], jobs: int) -> dict:
    todo, missing = [], {}
    stats_results = S._load_results()  # status (ok / timeout / error) of every method run
    for key in S.BENCHMARKS:
        kss = ks_list[:1] if key in S.DISTANCE_INDEPENDENT else ks_list  # fixed-distance codes: one run
        for ks in kss:
            methods = [m for m in S.RUN_TOOLS if S.circuit_path(key, ks, m).exists()]
            if not methods or not S.circuit_path(key, ks, "ideal").exists():
                continue
            qc = S.load_circuit(S.circuit_path(key, ks, "ideal"))
            _, partitions = S.generate_circuit(key, ks)
            labels = patch_labels(partitions, used_virtual_qubits(qc))
            if len(set(labels.values())) < 2:
                continue  # a single patch has nothing to separate
            todo += [(key, ks, m, labels, partitions) for m in methods]
            for m in S.RUN_TOOLS:  # no compiled circuit: timed out (T/O) or never run / failed (N/A)
                if m not in methods:
                    missing[(key, ks, m)] = {"status": S._no_result_label(stats_results, m, key, ks) or "N/A"}
    print(f"Scoring {len(todo)} compiled circuits on {jobs} worker(s)", flush=True)
    results = {}
    if jobs <= 1:
        outs = map(score_circuit, todo)
    else:
        pool = mp.get_context("spawn").Pool(min(jobs, len(todo)))
        outs = pool.imap_unordered(score_circuit, todo)
    for key, ks, method, r in outs:
        results[(key, ks, method)] = r
        print(f"  {S._title(key)} d={S.DISTANCE_INDEPENDENT.get(key, S.distance(ks))} {S.TOOL_STYLE[method][0]}: "
              f"ARI initial {r['ari'][0]:.2f}, mean {np.mean(r['ari']):.2f}, min {min(r['ari']):.2f} "
              f"(random {np.mean(r['random']):.2f}; {r['k']} patches, {r['n']} qubits)", flush=True)
    if jobs > 1:
        pool.close()
    for (key, ks, m), r in missing.items():
        results[(key, ks, m)] = r
        print(f"  {S._title(key)} d={S.DISTANCE_INDEPENDENT.get(key, S.distance(ks))} {S.TOOL_STYLE[m][0]}: "
              f"{r['status']} (no compiled circuit)", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PKL, "wb") as f:
        pickle.dump(results, f)
    return results


def add_missing(results: dict) -> dict:
    """Add an entry for every method without a compiled circuit for a scored (benchmark, distance), with its
    status (T/O or N/A) from the statistics results. Runs on every plot, so --plot-only also shows the
    timed-out methods, even with score files written before missing methods were recorded."""
    stats_results = S._load_results()
    for key, ks in {(k, ks) for (k, ks, _) in results}:
        for m in S.RUN_TOOLS:
            if (key, ks, m) not in results:
                results[(key, ks, m)] = {"status": S._no_result_label(stats_results, m, key, ks) or "N/A"}
    return results


def table(results: dict) -> str:
    lines = ["| benchmark | d | method | ARI initial | ARI mean | ARI min | random |", "|---|---|---|---|---|---|---|"]
    for (key, ks, m), r in sorted(results.items(), key=lambda kv: (S.BENCHMARKS.index(kv[0][0]), kv[0][1],
                                                                    S.RUN_TOOLS.index(kv[0][2]))):
        head = f"| {S._title(key)} | {S.DISTANCE_INDEPENDENT.get(key, S.distance(ks))} | {S.TOOL_STYLE[m][0]} | "
        if "ari" not in r:
            lines.append(head + f"{r['status']} | | | |")
            continue
        lines.append(head + f"{r['ari'][0]:.2f} | {np.mean(r['ari']):.2f} | {min(r['ari']):.2f} | "
                     f"{np.mean(r['random']):.2f} |")
    return "\n".join(lines)


def plot(results: dict, filename: Path = OUT_PDF, title: str = "b) Patch preservation") -> None:
    """One panel of a four-across row (same size, fonts and margins as the panels of experiment_statistics).
    One bar per method and benchmark in the method's base colour: mean ARI over time (mean over the distances
    if several were run), with the minimum over time as a dashed tick; a method without a compiled circuit
    (T/O or N/A) gets a short hollow, dashed bar as in experiment_statistics. Chance level: dotted line.
    The legend goes to its own file (save_legend). The centred title must stay within ~1.3 in at 7 pt."""
    import matplotlib.pyplot as plt

    keys = [k for k in S.BENCHMARKS if any(r[0] == k for r in results)]
    methods = [m for m in S.RUN_TOOLS if any(r[2] == m for r in results)]
    S._panel_fonts()
    fig = plt.figure(figsize=(S.PANEL_W, S.PANEL_H))
    ax = fig.add_subplot(111)
    width = 0.8 / len(methods)
    for j, m in enumerate(methods):
        _, color, hatch = S.TOOL_STYLE[m]
        for i, key in enumerate(keys):
            x = i + (j - (len(methods) - 1) / 2) * width
            runs = [r for (k, _, mm), r in results.items() if k == key and mm == m]
            scored = [r for r in runs if "ari" in r]
            if scored:
                ax.bar(x, max(np.mean([np.mean(r["ari"]) for r in scored]), 0), width, color=color, hatch=hatch,
                       edgecolor="black", linewidth=0.4, zorder=2)
                ax.plot([x - width / 2, x + width / 2], [np.mean([min(r["ari"]) for r in scored])] * 2,
                        color="black", linewidth=0.6, linestyle="--", zorder=3)
            elif runs:  # no compiled circuit (T/O or N/A): hollow, dashed, hatched
                ax.bar(x, NO_RESULT_BAR, width, color="white", hatch="xxx", edgecolor=color, linewidth=0.8,
                       linestyle="--", zorder=2)
    rand = [np.mean(r["random"]) for r in results.values() if "random" in r]
    chance = float(np.mean(rand)) if rand else 0.0
    ax.axhline(chance, color="grey", linestyle=":", linewidth=0.8, zorder=1)
    ax.set_ylim(min(0, chance - 0.05), 1.05)
    ax.set_xticks(range(len(keys)), [S.PANEL_TICKS.get(k, k) for k in keys], fontsize=S.FONT_PT - 1.5,
                  linespacing=0.95)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Patch agreement (ARI)")  # Adjusted Rand Index (caption)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    S._panel_title(ax, title, better="Higher is better ↑")
    fig.subplots_adjust(**S.PANEL_MARGINS)
    fig.savefig(filename, format="pdf")
    plt.close(fig)
    save_legend(methods)


def save_legend(methods, filename: Path = OUT_LEGEND) -> None:
    """Legend of the panel: the methods in one row (explain the hollow timeout bars, the dashed minimum tick and
    the dotted chance line in the caption)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    S._panel_fonts()
    handles = [Patch(facecolor=S.TOOL_STYLE[m][1], hatch=S.TOOL_STYLE[m][2], edgecolor="black", linewidth=0.4,
                     label=S.TOOL_STYLE[m][0]) for m in methods]
    fig = plt.figure(figsize=(S.TEXT_WIDTH_IN, 0.3))
    fig.legend(handles=handles, loc="center", frameon=False, ncols=len(handles), columnspacing=1.0,
               handlelength=1.4, handletextpad=0.35)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, bbox_inches="tight", format="pdf")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--distances", nargs="+", type=int, default=[S.distance(k) for k in S.DISTANCE_SCALES])
    ap.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--plot-only", action="store_true", help="replot the saved scores")
    ap.add_argument("--title", default="b) Patch preservation",
                    help="panel title incl. its letter in the four-panel row (keep it short, ~1.3 in at 7 pt)")
    a = ap.parse_args()
    if a.plot_only:
        with open(OUT_PKL, "rb") as f:
            res = pickle.load(f)
    else:
        res = run([(d - 1) // 2 for d in a.distances], a.jobs)
    res = add_missing(res)
    md = table(res)
    OUT_MD.write_text(md)
    print(md)
    plot(res, title=a.title)