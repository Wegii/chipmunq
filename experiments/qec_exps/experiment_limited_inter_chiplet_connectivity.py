"""LER row of the paper (Fig. 8): distributed lattice-surgery CNOT on chiplet backends.

    a) Compilation effects   LER of the ideal (uncompiled) circuit, Chipmunq and LightSABRE, d = 5 ... 13
    b) Connectivity vs. LER  Chipmunq, d = 7, n_inter in {8, 6, 4, 2, 1}
    c) Relative LER          b) relative to full connectivity (n_inter = 8)
    d) Higher distances      Chipmunq, d = 9 ... 15, three connectivity levels per distance:
                             full (2k + 3 links, the maximum of the backend), ~d/2 (k + 1 links), single (1 link)
    Legend for a-d: ler_row_legend.pdf (one legend, full text width)

All panels are drawn at their printed size (four across the text width, Fig. 10 style) and share one encoding:
marker = code distance, colour = compilation method (a) / connectivity (b-d; lighter = fewer links), line style
= connectivity level in d). Points without observed errors are not shown.

Diagnostics (not in the paper): Chipmunq / LightSABRE layouts of a) (backend_mapping/), and the mapping and
routing of every d) configuration (inter_chiplet_<p>_high_d_layouts.png).

Compilations run in their own process with a wall-clock limit of COMPILE_TIMEOUT_S; a configuration that hits
the limit is skipped (not simulated). Sampling: at most MAX_SHOTS shots and MAX_ERRORS logical errors per task
(panel a, d) / run_sinter_simulation (panels b, c).

Run from the repo root:
    python experiments/qec_exps/experiment_distributed_lattice_surgery.py                    # a-d
    python experiments/qec_exps/experiment_distributed_lattice_surgery.py --part a           # only a)
    python experiments/qec_exps/experiment_distributed_lattice_surgery.py --part connectivity  # only b), c)
    python experiments/qec_exps/experiment_distributed_lattice_surgery.py --part high        # only d)
    python experiments/qec_exps/experiment_distributed_lattice_surgery.py --part layouts     # d) mappings only
    python experiments/qec_exps/experiment_distributed_lattice_surgery.py --plot-only [--part ...]
    python experiments/qec_exps/experiment_distributed_lattice_surgery.py --quick --part a   # smoke test of a)
"""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.getcwd(), "."))

from stim import Circuit as StimCircuit
from experiments.exp_utils.circuit_generator import get_tqec_cnot_rotated
from glue.qiskit_qec.stim_code_circuit import StimCodeCircuit
from experiments.exp_utils.simulation_utils import *
from experiments.exp_utils.transpilation_utils import *
from experiments.exp_utils.utils import *
from qeccm.backends.backend_utils import plot_circuit_layout_utilization, plot_circuit_layout, generate_coordinates
from experiments.exp_utils.circuit_noise import get_noise_model

# Plotting
import argparse
import multiprocessing
import pickle
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from pathlib import Path

# ######################################################################################
# Figure style of the LER row (shared by all four panels)
# ######################################################################################
# --------------------------------------------------------------------------------------
# Panel geometry: drawn at printed size -- include at natural width (or width=0.24\textwidth), no scaling
# --------------------------------------------------------------------------------------
TEXT_WIDTH_IN = 7.0                   # full text width of the paper (two-column IEEE/ACM: ~7.0 in)
N_PANELS = 4
PANEL_W = TEXT_WIDTH_IN / N_PANELS    # 1.75 in
PANEL_H = 1.6
FONT_PT = 7
MARGINS = dict(left=0.27, right=0.92, top=0.80, bottom=0.24)  # identical for all panels -> axes line up

# --------------------------------------------------------------------------------------
# Encoding shared by all panels
# --------------------------------------------------------------------------------------
DIST_MARKERS = {3: "+", 5: "x", 7: "o", 9: "s", 11: "^", 13: "D", 15: "v"}
METHOD_STYLE = {"default": ("Ideal", "#000000"), "compiled": ("Chipmunq", "#3B6FA8"),
                "sabre": ("LightSABRE", "#C85E59")}
# b), c): n_inter -> colour (lighter = fewer links)
NINTER_COLORS = {1: "#B7D1EC", 2: "#8FB7E1", 4: "#5E97CC", 6: "#3B6FA8", 8: "#2A5687"}
# d): connectivity level -> (colour on the same gradient, line style, legend label)
LEVEL_STYLE = {"full": ("#1B3A63", "-", "max links"),
               "half": ("#5E97CC", "--", r"$\approx d/2$ links"),
               "single": ("#B7D1EC", ":", "single link")}
# Code distances shown in the multi-distance panels (only plotting; all simulated distances stay in the
# result files). None = all simulated distances. Override with --distances-a / --distances-d.
PLOT_DISTANCES_A = (5, 9, 13)      # a) Compilation effects
PLOT_DISTANCES_D = (9, 13)         # d) Higher distances
CONNECTIVITY_DISTANCE = 7          # b), c)


def legend_distances() -> list[int]:
    """Distances listed in the row legend: those actually plotted in a-d."""
    ds = set(PLOT_DISTANCES_A or [2 * k + 1 for k in DISTANCE_SCALES]) | {CONNECTIVITY_DISTANCE}
    ds |= set(PLOT_DISTANCES_D or [2 * k + 1 for k in HIGH_KS])
    return sorted(ds)


def _shown(ds, selection) -> list[str]:
    """Simulated distances (as strings) restricted to the selection (None = all)."""
    return [d for d in ds if selection is None or int(d) in selection]


def dist_marker(d) -> str:
    return DIST_MARKERS.get(int(d), "P")


def fonts() -> None:
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
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "lines.markeredgewidth": 0.7,
        "errorbar.capsize": 1.5,
    })


def panel():
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    fig.subplots_adjust(**MARGINS)
    return fig, ax


def title(ax, text: str, better: str = "Lower is better ↓") -> None:
    """Panel title and the "better" hint, both centred on the plot rectangle (axes), not on the figure."""
    ax.text(0.5, 1.03, text, transform=ax.transAxes, fontweight="bold", ha="center", va="bottom")
    ax.text(0.5, 1.16, better, transform=ax.transAxes, fontweight="bold", color=plot_lib_color,
            ha="center", va="bottom")


def log_axes(ax, xlim=(1e-4, 1e-2), ylim=(5e-8, 2e0), ylabel="LER") -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Physical error rate")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)


def line(ax, xs, ys, color, d, ls="-", **kw):
    """One LER curve: colour = method / connectivity, hollow marker = code distance."""
    return ax.plot(xs, ys, linestyle=ls, color=color, marker=dist_marker(d), markerfacecolor="none",
                   markeredgecolor=color, **kw)


def save_row_legend(filename: str) -> None:
    """One legend for panels a-d, full text width (built from the style constants only, no data)."""
    fonts()
    grey = "#606060"
    # Row 1: compilation methods (a) and code distances (all panels); row 2: connectivity (b-d)
    top = [Line2D([], [], color=c, label=l) for l, c in METHOD_STYLE.values()]
    top += [Line2D([], [], color=grey, marker=DIST_MARKERS[d], markerfacecolor="none", markeredgecolor=grey,
                   linestyle="none", label=f"d={d}") for d in legend_distances()]
    bottom = [Line2D([], [], color=c, label=r"$n_{inter}$" + f" = {n}") for n, c in NINTER_COLORS.items()]
    bottom += [Line2D([], [], color=c, linestyle=ls, label=l) for c, ls, l in LEVEL_STYLE.values()]
    ncols = max(len(top), len(bottom))
    blank = lambda: Line2D([], [], linestyle="none", label=" ")  # noqa: E731
    top += [blank() for _ in range(ncols - len(top))]
    bottom += [blank() for _ in range(ncols - len(bottom))]
    h = [x for pair in zip(top, bottom) for x in pair]  # matplotlib fills legends column by column
    fig = plt.figure(figsize=(TEXT_WIDTH_IN, 0.4))
    fig.legend(handles=h, loc="center", frameon=False, ncols=ncols, columnspacing=0.8,
               handlelength=1.8, handletextpad=0.35)
    fig.savefig(filename, bbox_inches="tight", format="pdf")
    plt.close(fig)


# ######################################################################################
# a) Compilation effects: ideal vs. Chipmunq vs. LightSABRE
# ######################################################################################
# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
DISTANCE_SCALES = [2, 3, 4, 5, 6]  # d = 2k + 1 -> 5, 7, 9, 11, 13
MAX_SHOTS = 10_000_000              # per (method, d, p)
MAX_ERRORS = 1000                   # per (method, d, p)
COMPILE_TIMEOUT_S = 1000           # wall-clock limit per compilation (Chipmunq Basic and LightSABRE)
OUTPUT_DIR = Path("experiments/evaluation/qec_evaluation")



def get_backend(p_icc: float, amp_icc: float, d: int = -1):
    """Backend for distance scale d (code distance 2d + 1); each patch fits on one chiplet.
    Chiplet size (11 + 4(d-1)) x (6 + 2(d-1)) with 2d + 3 inter-chiplet links, as for d = 1..4 before."""
    chiplet_size = (6, 6, 11 + 4 * (d - 1), 6 + 2 * (d - 1))
    nic = 2 * d + 3
    backend = BackendChipletV2(
        size=chiplet_size,
        n_inter=nic,
        connectivity="nn",
        topology="rotated_grid",
        inter_chiplet_noise=p_icc,
        inter_chiplet_amplification=amp_icc,
        inter_chiplet_noise_type="constant",
        num_defective_qubits=0,
    )
    return nic, backend


def compile_circuit(t: str, distance_scale: int, p_icc: float, amp_icc: float):
    """Compile the CNOT with Chipmunq ("compiled") or LightSABRE ("sabre"). Module level so it can run in a
    killable child process. Returns (Stim circuit as string, compiled QuantumCircuit)."""
    _, backend = get_backend(p_icc=p_icc, amp_icc=amp_icc, d=distance_scale)
    circuit, partitions = get_tqec_cnot_rotated(distance_scale=distance_scale, n1=1, n2=0)
    if t == "compiled":
        _, custom_circuit, _, _ = transpile_stim_circuit(
            circuit,
            backend,
            pre_defined_partitions=partitions,
            routing_type="cost",
            routing_alpha=0,
            routing_beta=0,
        )
    elif t == "sabre":
        custom_circuit = sabre_transpilation(StimCodeCircuit(stim_circuit=circuit).qc, backend)
    else:
        raise ValueError(t)
    return str(get_stim_circuits_with_detectors(custom_circuit)[0][0]), custom_circuit


def _rates(stats):
    """{method: {d: {p: LER}}} (only configurations that were simulated)."""
    error_rates = defaultdict(lambda: defaultdict(dict))
    for s in stats:
        n = s.shots - s.discards
        if n > 0:
            md = s.json_metadata
            error_rates[str(md["run_name"])][str(md["d"])][md["p"]] = s.errors / n
    return error_rates


def plot_evaluation(stats, filename, inter_chiplet_noise):
    """a) LER of every method and distance (panel of the four-panel row), and the LightSABRE / Chipmunq
    ratio as a separate panel in the same style. Points without observed errors are not shown."""
    error_rates = _rates(stats)
    ps = sorted({s.json_metadata["p"] for s in stats})
    d_values = _shown(sorted({str(s.json_metadata["d"]) for s in stats}, key=int), PLOT_DISTANCES_A)
    fonts()

    # ---------------- a) LER of every method and distance ----------------
    fig, ax = panel()
    ax.plot(ps, ps, linestyle="--", linewidth=0.8, color="#000000B3")
    for t, (_, color) in METHOD_STYLE.items():
        for d in d_values:
            pts = [(p, error_rates[t][d][p]) for p in ps if error_rates[t][d].get(p, 0) > 0]
            if pts:  # nothing to draw if the compilation timed out or no errors were observed
                xs, ys = zip(*pts)
                line(ax, xs, ys, color, d)
    log_axes(ax, ylim=(5e-9, 2e0))
    title(ax, "a) Compilation effects")
    fig.savefig(filename + ".pdf", format="pdf")
    plt.close(fig)

    # ---------------- LER_LightSABRE / LER_Chipmunq ----------------
    fig, ax = panel()
    for d in d_values:
        sab, chip, ideal = error_rates["sabre"][d], error_rates["compiled"][d], error_rates["default"][d]
        ps_both = [p for p in ps if sab.get(p, 0) > 0 and chip.get(p, 0) > 0]
        if not ps_both:
            continue
        ratio = [sab[p] / chip[p] for p in ps_both]
        avg = [r for p, r in zip(ps_both, ratio) if p < 1e-2]
        avg_ideal = [sab[p] / ideal[p] for p in ps_both if p < 1e-2 and ideal.get(p, 0) > 0]
        print(f"Relative increase for compiled {d} for icc {inter_chiplet_noise}: {np.mean(avg) if avg else np.nan}")
        print(f"Relative increase for sabre {d} for icc {inter_chiplet_noise}: "
              f"{np.mean(avg_ideal) if avg_ideal else np.nan}")
        line(ax, ps_both, ratio, METHOD_STYLE["compiled"][1], d, ls="--")
    log_axes(ax, ylim=None, ylabel=r"$LER_{LightSABRE} / LER_{Chipmunq}$")
    title(ax, "b) LER comparison")
    fig.savefig(filename + "_relative.pdf", format="pdf")
    plt.close(fig)

    # One legend for the whole LER row (a-d), identical to the one of the connectivity experiment
    save_row_legend(str(Path(filename).parent / "ler_row_legend.pdf"))


def run_exp_distributed_lattice_surgery(reproduce: bool = False, quick: bool = False) -> None:
    # Physical noise level
    ps = list(np.logspace(-4, -2, 6)) if not quick else [1e-3, 5e-3]  # 1e-4 ... 1e-2, 6 points

    # Inter-chiplet noise levels
    inter_chiplet_noise = [1e-3]

    # Compilation methods
    # - default: No compilation
    # - compiled: Our method (Basic routing)
    # - sabre: LightSABRE method
    ts = ["default", "compiled", "sabre"]

    # Code size of surface code: d = 2k + 1
    ks = DISTANCE_SCALES if not quick else [1, 2]
    timeout = COMPILE_TIMEOUT_S if not quick else 120
    output_dir = OUTPUT_DIR / ("quick" if quick else "")

    for ps_inter in inter_chiplet_noise:
        if reproduce:
            # ---- compile every (method, k) once; Chipmunq and LightSABRE under the time limit ----
            circuits, status = {}, {}
            for k in ks:
                circuit, _ = get_tqec_cnot_rotated(distance_scale=k, n1=1, n2=0)
                circuits[("default", k)] = get_stim_circuits_with_detectors(StimCodeCircuit(circuit).qc)[0][0]
                status[("default", k)] = dict(status="ok", runtime_s=0.0)
                for t in ("compiled", "sabre"):
                    res, runtime, st = run_with_timeout(compile_circuit, t, k, ps_inter, 1, timeout=timeout)
                    status[(t, k)] = dict(status=st, runtime_s=runtime)
                    print(f"[d={2 * k + 1}] {t}: {st} after {runtime:.0f} s", flush=True)
                    if st != "ok":
                        continue  # timed out / failed: not simulated
                    stim_str, custom_circuit = res
                    circuits[(t, k)] = StimCircuit(stim_str)
                    _, backend = get_backend(p_icc=ps_inter, amp_icc=1, d=k)
                    for plot_fn, prefix in ((plot_circuit_layout, "layout"),
                                            (plot_circuit_layout_utilization, "mapping")):
                        try:
                            plot_fn(custom_circuit, backend,
                                    filename=str(output_dir / "backend_mapping" / f"{prefix}_{t}_{k}.png"))
                        except Exception as e:  # visualisation only
                            print(f"{prefix} plot skipped: {e}")

            def _get_sinter_task():
                for (t, k), circ in circuits.items():
                    remote = None if t == "default" else get_backend(d=k, p_icc=ps_inter, amp_icc=1)[1].inter_chiplet_connections
                    for p in ps:
                        yield sinter.Task(
                            circuit=get_noise_model("modsi1000", None, p, None, remote=remote).noisy_circuit(circ),
                            json_metadata={"d": 2 * k + 1, "r": 2 * k + 1, "p": p, "run_name": t},
                        )

            # Run simulation (at most MAX_SHOTS shots / MAX_ERRORS errors per task)
            stats = sinter.collect(
                num_workers=max(1, multiprocessing.cpu_count() // 2),
                tasks=_get_sinter_task(),
                max_shots=MAX_SHOTS,
                max_errors=MAX_ERRORS,
                decoders=["pymatching"],
                print_progress=True,
                hint_num_tasks=len(circuits) * len(ps),
                count_observable_error_combos=True,
            )

            # Save simulation results and compile status (timeouts)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / f"single_cnot_rotated_{ps_inter}.pkl", "wb") as f:
                pickle.dump(stats, f)
            with open(output_dir / f"compile_status_{ps_inter}.pkl", "wb") as f:
                pickle.dump(status, f)

        # Load simulation results
        with open(output_dir / f"single_cnot_rotated_{ps_inter}.pkl", "rb") as f:
            stats = pickle.load(f)
        status_file = output_dir / f"compile_status_{ps_inter}.pkl"
        if status_file.exists():
            with open(status_file, "rb") as f:
                for (t, k), st in pickle.load(f).items():
                    if st["status"] != "ok":
                        print(f"d={2 * k + 1} {t}: {st['status']} (not simulated)")

        # Plot statistics
        plot_evaluation(
            stats,
            filename=str(output_dir / f"single_cnot_rotated_{ps_inter}"),
            inter_chiplet_noise=ps_inter,
        )


# ######################################################################################
# b) - d) Limited inter-chiplet connectivity
# ######################################################################################
INTER_OUTPUT_DIR = Path("experiments/evaluation/inter_chiplet")

# --------------------------------------------------------------------------------------
# Higher code distances (panel d)
# --------------------------------------------------------------------------------------
HIGH_KS = [4, 5, 6, 7]                   # d = 2k + 1 -> 9, 11, 13, 15
HIGH_PS = list(np.logspace(-4, -2, 10))  # 1e-4 ... 1e-2, 10 points
HIGH_MAX_SHOTS = 10_000_000
HIGH_MAX_ERRORS = 1000


def connectivity_levels(k: int) -> dict[str, int]:
    """Inter-chiplet links per chiplet edge for distance scale k (d = 2k + 1): the backend's maximum, a level
    below the code distance (~d/2), and a single link."""
    return {"full": 2 * k + 3, "half": k + 1, "single": 1}


def _points(rates: dict, ps) -> tuple:
    """(xs, ys) of the points with at least one observed error (others are not shown on the log axis)."""
    pts = [(p, rates[p]) for p in ps if rates.get(p, 0) > 0]
    return tuple(zip(*pts)) if pts else ((), ())


# --------------------------------------------------------------------------------------
# b) LER vs. physical error rate for d = 7
# --------------------------------------------------------------------------------------
def plot_connectivity(stats, filename, inter_chiplet_noise, num_inter):
    error_rates = _rates(stats)
    ps = sorted({s.json_metadata["p"] for s in stats})
    fonts()
    fig, ax = panel()
    ax.plot(ps, ps, linestyle="--", linewidth=0.8, color="#000000B3")
    for n in sorted(num_inter):
        xs, ys = _points(error_rates[str(n)]["7"], ps)
        if xs:
            line(ax, xs, ys, NINTER_COLORS[n], 7)
    log_axes(ax)
    title(ax, "b) Connectivity vs. LER")
    fig.savefig(filename, format="pdf")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# c) LER relative to full connectivity for d = 7
# --------------------------------------------------------------------------------------
def plot_connectivity_relative(stats, filename, inter_chiplet_noise):
    error_rates = _rates(stats)
    ps = sorted({s.json_metadata["p"] for s in stats})
    fonts()
    fig, ax = panel()
    full = error_rates["8"]["7"]
    for n in (1, 2, 4, 6):
        red = error_rates[str(n)]["7"]
        ratio = {p: red[p] / full[p] for p in ps if red.get(p, 0) > 0 and full.get(p, 0) > 0}
        xs, ys = _points(ratio, ps)
        if xs:
            line(ax, xs, ys, NINTER_COLORS[n], 7)
    log_axes(ax, ylim=(0.9, 130), ylabel=r"$LER_{Reduced} / LER_{Full}$")
    title(ax, "c) Relative LER")
    fig.savefig(filename, format="pdf")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# d) higher code distances
# --------------------------------------------------------------------------------------
def plot_high_distance(stats, filename):
    """LER vs. physical error rate: marker = distance, colour and line style = connectivity level."""
    error_rates = _rates(stats)
    ps = sorted({s.json_metadata["p"] for s in stats})
    ds = [int(d) for d in _shown(sorted({str(s.json_metadata["d"]) for s in stats}, key=int), PLOT_DISTANCES_D)]
    fonts()
    fig, ax = panel()
    ax.plot(ps, ps, linestyle="--", linewidth=0.8, color="#000000B3")
    for d in ds:
        for level, (color, ls, _) in LEVEL_STYLE.items():
            xs, ys = _points(error_rates[level][str(d)], ps)
            if xs:
                line(ax, xs, ys, color, d, ls=ls)
    log_axes(ax)
    title(ax, "d) Higher distances")
    fig.savefig(filename, format="pdf")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Diagnostic: mapping and routing of every panel-d configuration
# --------------------------------------------------------------------------------------
PATCH_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def plot_high_d_layouts(layouts: dict, ps_inter: float, filename: str) -> None:
    """Rows = code distance, columns = connectivity level. Qubits at their initial position (colour = patch),
    couplers coloured by the number of SWAPs routed over them, inter-chiplet links dashed purple; each panel
    shows only the chiplets the circuit uses."""
    from matplotlib.colors import LogNorm

    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    ks = sorted({k for k, _ in layouts["configs"]})
    levels = list(LEVEL_STYLE)
    vmax = max([max(c["swaps"].values()) for c in layouts["configs"].values() if c and c["swaps"]] or [1])
    norm, cmap = LogNorm(1, max(vmax, 2)), plt.cm.inferno_r
    fig, axes = plt.subplots(len(ks), len(levels), figsize=(3.4 * len(levels), 3.0 * len(ks)), squeeze=False)
    for i, k in enumerate(ks):
        parts, used = layouts["partitions"][k], layouts["used"][k]
        patch_of = {int(q): j for j, p in enumerate(parts) for q in p["indices"]}
        for j, level in enumerate(levels):
            ax = axes[i][j]
            ax.set_axis_off()
            info = layouts["configs"].get((k, level))
            n = connectivity_levels(k)[level]
            if not info:
                ax.text(0.5, 0.5, "T/O", transform=ax.transAxes, ha="center", va="center", fontsize=14)
                ax.set_title(f"d={2 * k + 1}, n_inter={n}", fontsize=8)
                continue
            backend = get_backend_k(n, ps_inter, k)
            xy = np.asarray(generate_coordinates(backend), float)
            chip = backend.node_to_chiplet
            lay = info["layout"]
            phys_used = {lay[v] for v in used if v < len(lay)}
            chips = {chip[q] for q in phys_used} | {chip[q] for e in info["swaps"] for q in e}
            shown = [q for q in range(len(xy)) if chip[q] in chips]
            edges = {tuple(sorted(map(int, e))) for e in backend.coupling_map.get_edges()}
            for a, b in edges:
                if chip[a] in chips and chip[b] in chips and chip[a] != chip[b]:
                    ax.plot(*xy[[a, b]].T, color="#7B2CBF", ls="--", lw=0.8, zorder=1)
            ax.scatter(*xy[shown].T, s=2, color="#D8D8D8", zorder=2)
            for (a, b), cnt in info["swaps"].items():
                ax.plot(*xy[[a, b]].T, color=cmap(norm(cnt)), lw=0.6 + 1.8 * norm(cnt), zorder=3)
            vq = [v for v in used if v < len(lay)]
            cols = [PATCH_COLORS[patch_of.get(v, 0) % len(PATCH_COLORS)] for v in vq]
            ax.scatter(*xy[[lay[v] for v in vq]].T, s=5, c=cols, edgecolors="none", zorder=4)
            ax.set_title(f"d={2 * k + 1}, n_inter={n} ({level})\n{sum(info['swaps'].values())} SWAPs, "
                         f"{info['link2q']} two-qubit gates on links", fontsize=8)
            ax.set_aspect("equal")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.5, pad=0.01,
                 label="SWAPs on coupler")
    fig.legend(handles=[Line2D([], [], marker="o", ls="none", color=PATCH_COLORS[j], label=f"patch {j}")
                        for j in range(len(layouts["partitions"][ks[0]]))]
               + [Line2D([], [], color="#7B2CBF", ls="--", label="inter-chiplet link")],
               loc="lower center", ncols=8, frameon=False)
    fig.suptitle("Panel d): initial placement (colour = patch) and routing (SWAPs per coupler)", fontweight="bold")
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def get_backend_k(n_icc: int, p_icc: float, k: int) -> BackendChipletV2:
    """Backend for distance scale k: chiplet (11 + 4(k-1)) x (6 + 2(k-1)), one patch per chiplet."""
    return BackendChipletV2(
        size=(6, 6, 11 + 4 * (k - 1), 6 + 2 * (k - 1)),
        n_inter=n_icc,
        connectivity="nn",
        topology="rotated_grid",
        inter_chiplet_noise=p_icc,
        inter_chiplet_amplification=1,
        inter_chiplet_noise_type="constant",
    )


def compile_high_d(k: int, n_icc: int, p_icc: float, circuit_str: str, partitions):
    """Chipmunq compilation (same routing as panels b, c). Module level: runs in a killable child process.
    The tqec circuit is generated once by the parent (concurrent tqec generation writes the same temp files).
    Returns (Stim circuit as string, mapping/routing summary for the layout plot)."""
    backend = get_backend_k(n_icc, p_icc, k)
    _, custom_circuit, _, _ = transpile_stim_circuit(
        StimCircuit(circuit_str), backend, pre_defined_partitions=partitions,
        routing_type="default", routing_alpha=1.0, routing_beta=1.0,
    )
    chip = backend.node_to_chiplet
    swaps, link2q = Counter(), 0
    for ins in custom_circuit.data:
        if ins.operation.num_qubits != 2 or ins.operation.name == "barrier":
            continue
        a, b = (custom_circuit.find_bit(q).index for q in ins.qubits)
        link2q += chip[a] != chip[b]
        if ins.operation.name == "swap":
            swaps[tuple(sorted((a, b)))] += 1
    info = dict(layout=list(custom_circuit.layout.initial_index_layout(filter_ancillas=True)), swaps=dict(swaps),
                link2q=link2q)
    return str(get_stim_circuits_with_detectors(custom_circuit)[0][0]), info


def _used_virtual_qubits(circuit: StimCircuit) -> list[int]:
    """Qubits acted on by gates (the patch specification also lists unused indices)."""
    skip = {"TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"}
    return sorted({t.value for ins in circuit.flattened() if ins.name not in skip
                   for t in ins.targets_copy() if t.is_qubit_target})


def compile_high_distance(ps_inter: float, output_dir: Path) -> dict:
    """Compile every panel-d configuration concurrently (each in its own process with a COMPILE_TIMEOUT_S
    limit), save the mapping/routing summary and plot it. Returns {(k, level, n): Stim circuit}."""
    sources, used = {}, {}
    for k in HIGH_KS:
        circuit, partitions = get_tqec_cnot_rotated(distance_scale=k, n1=1, n2=0)
        sources[k] = (str(circuit), partitions)
        used[k] = _used_virtual_qubits(circuit)
    jobs = [(k, level, n) for k in HIGH_KS for level, n in connectivity_levels(k).items()]
    caps = ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    saved = {v: os.environ.get(v) for v in caps}
    os.environ.update({v: str(max(1, (os.cpu_count() or 1) // len(jobs))) for v in caps})
    circuits, configs = {}, {}

    def _job(job):
        k, level, n = job
        return job, run_with_timeout(compile_high_d, k, n, ps_inter, *sources[k], timeout=COMPILE_TIMEOUT_S)

    try:
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            for (k, level, n), (res, runtime, st) in (f.result() for f in as_completed(
                    [pool.submit(_job, j) for j in jobs])):
                print(f"[d={2 * k + 1}, n_inter={n} ({level})] {st} after {runtime:.0f} s", flush=True)
                configs[(k, level)] = None
                if st == "ok":
                    stim_str, info = res
                    circuits[(k, level, n)] = StimCircuit(stim_str)
                    configs[(k, level)] = info
    finally:
        for v, old in saved.items():
            if old is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old
    layouts = dict(configs=configs, partitions={k: sources[k][1] for k in HIGH_KS}, used=used)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"inter_chiplet_{ps_inter}_high_d_layouts.pkl", "wb") as f:
        pickle.dump(layouts, f)
    plot_high_d_layouts(layouts, ps_inter, str(output_dir / f"inter_chiplet_{ps_inter}_high_d_layouts.png"))
    return circuits


def run_high_distance(ps_inter: float, output_dir: Path) -> list:
    """Compile and simulate panel d)."""
    circuits = compile_high_distance(ps_inter, output_dir)

    def _tasks():
        for (k, level, n), circ in sorted(circuits.items()):
            remote = get_backend_k(n, ps_inter, k).inter_chiplet_connections
            for p in HIGH_PS:
                yield sinter.Task(
                    circuit=get_noise_model("modsi1000", None, p, None, remote=remote).noisy_circuit(circ),
                    json_metadata={"d": 2 * k + 1, "r": 2 * k + 1, "p": p, "run_name": level, "n_inter": n,
                                   "p_inter": ps_inter},
                )

    stats = sinter.collect(
        num_workers=multiprocessing.cpu_count(),
        tasks=_tasks(),
        max_shots=HIGH_MAX_SHOTS,
        max_errors=HIGH_MAX_ERRORS,
        decoders=["pymatching"],
        print_progress=True,
        hint_num_tasks=len(circuits) * len(HIGH_PS),
    )
    with open(output_dir / f"inter_chiplet_{ps_inter}_high_d.pkl", "wb") as f:
        pickle.dump(stats, f)
    return stats


def run_exp_distributed_inter_chiplet(reproduce: bool = False, part: str = "all") -> None:
    """Panels b) - d). part: "all" (b-d), "low" (b, c), "high" (d), "layouts" (d mappings only)."""

    # Number of inter_chiplet_connections
    num_inter_chiplet_connections = [8, 6, 4, 2, 1]

    # Physical noise level
    ps = list(np.logspace(-4, -1, 10))

    # Inter-chiplet noise level
    inter_chiplet_noise = [1e-3]  # [1e-4, 1e-3, 1e-2]

    # Transpilation
    ts = [str(i) for i in num_inter_chiplet_connections]

    # Code size of surface code
    ks = [3]

    # Routing methods
    routing_types = ["default"]  # ["cost", "default"]

    output_dir = INTER_OUTPUT_DIR
    for ps_inter in inter_chiplet_noise:
        if part in ("all", "low"):
            if reproduce:
                transpiled_circuits = {}

                def get_circuit(n_icc, p_icc, amp_icc, t: str, routing_type: str, k: int = 1) -> StimCircuit:
                    if (n_icc, p_icc, amp_icc, routing_type, k) in transpiled_circuits:
                        # Circuit does not need to be transpiled again
                        print("Utilizing existing backend")
                        return transpiled_circuits[(n_icc, p_icc, amp_icc, routing_type, k)]
                    else:
                        circuit, partitions = get_tqec_cnot_rotated(distance_scale=k, n1=1, n2=0)

                        backend = get_backend(n_icc=n_icc, p_icc=p_icc, amp_icc=amp_icc, k=k)

                        # Transpile circuit to backend
                        _, custom_circuit, _, _ = transpile_stim_circuit(
                            circuit,
                            backend,
                            pre_defined_partitions=partitions,
                            routing_type=routing_type,  # "cost",
                            routing_alpha=1.0,
                            routing_beta=1.0,
                        )
                        # Convert circuit to stim
                        custom_circuit_stim = get_stim_circuits_with_detectors(custom_circuit)[0][0]
                        # Add circuit to dictionary, in order to not transpile this circuit configuration again
                        transpiled_circuits[(n_icc, p_icc, amp_icc, routing_type, k)] = custom_circuit_stim

                        plot_circuit_layout(
                            custom_circuit,
                            backend,
                            filename=f"experiments/evaluation/inter_chiplet/backend_mapping/layout_{n_icc}_{k}.png",
                        )

                        plot_circuit_layout_utilization(
                            custom_circuit,
                            backend,
                            filename=f"experiments/evaluation/inter_chiplet/backend_mapping/mapping_{n_icc}_{k}.png",
                        )

                        return custom_circuit_stim

                def get_backend(n_icc: int, p_icc: float, amp_icc: float, t: str = "", k: int = 1) -> BackendChipletV2:
                    if t == "default":
                        return None
                    else:
                        if k == 1:
                            chiplet_size = (6, 6, 11, 6)
                        elif k == 2:
                            chiplet_size = (6, 6, 15, 8)
                        elif k == 3:
                            chiplet_size = (6, 6, 19, 10)

                        return BackendChipletV2(
                            size=chiplet_size,  # (2, 2, 15, 8),
                            n_inter=n_icc,
                            connectivity="nn",
                            topology="rotated_grid",
                            inter_chiplet_noise=p_icc,
                            inter_chiplet_amplification=amp_icc,
                            inter_chiplet_noise_type="constant",
                        )

                def _get_sinter_task():
                    # Construct sinter task for multiple code distances and noise levels
                    yield from (
                        sinter.Task(
                            circuit=circuit,
                            json_metadata={"d": 2 * k + 1, "r": 2 * k + 1, "p": p, "run_name": t, "p_inter": p_icc},
                        )
                        for circuit, k, p, t, rt, p_icc in (
                            (
                                get_noise_model(
                                    "modsi1000",
                                    None,
                                    p,
                                    None,
                                    remote=(
                                        None
                                        if t == "default"
                                        else get_backend(int(t), ps_inter, 1, t, k).inter_chiplet_connections
                                    ),
                                ).noisy_circuit(
                                    get_circuit(-1 if t == "default" else int(t), ps_inter, 1, t, routing_type=rt, k=k)
                                ),
                                k,
                                p,
                                t,
                                rt,
                                ps_inter,
                            )
                            for t in ts
                            for rt in routing_types
                            for k in ks
                            for p in ps
                        )
                    )

                # Run simulation
                stats = run_sinter_simulation(_get_sinter_task, ks, ps)

                # Save simulation results
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_dir / f"inter_chiplet_{ps_inter}_sweep.pkl", "wb") as f:
                    pickle.dump(stats, f)

            # Load simulation results
            with open(output_dir / f"inter_chiplet_{ps_inter}_sweep.pkl", "rb") as f:
                stats = pickle.load(f)

            # b) LER vs. physical error rate
            plot_connectivity(
                stats,
                filename=str(output_dir / f"inter_chiplet_{ps_inter}.pdf"),
                inter_chiplet_noise=ps_inter,
                num_inter=num_inter_chiplet_connections,
            )

            # c) Relative to full inter-chiplet connectivity
            plot_connectivity_relative(
                stats,
                filename=str(output_dir / f"inter_chiplet_{ps_inter}_difference.pdf"),
                inter_chiplet_noise=ps_inter,
            )

        if part in ("all", "high"):
            # d) Higher code distances
            if reproduce:
                stats_high = run_high_distance(ps_inter, output_dir)
            else:
                with open(output_dir / f"inter_chiplet_{ps_inter}_high_d.pkl", "rb") as f:
                    stats_high = pickle.load(f)
                layouts_file = output_dir / f"inter_chiplet_{ps_inter}_high_d_layouts.pkl"
                if layouts_file.exists():
                    with open(layouts_file, "rb") as f:
                        plot_high_d_layouts(pickle.load(f), ps_inter,
                                            str(output_dir / f"inter_chiplet_{ps_inter}_high_d_layouts.png"))
            plot_high_distance(stats_high, filename=str(output_dir / f"inter_chiplet_{ps_inter}_high_d.pdf"))

        if part == "layouts":
            # Mapping and routing of the panel-d configurations only (compiles, no simulation)
            compile_high_distance(ps_inter, output_dir)

    # One legend for the whole LER row (a-d); depends on the style constants only
    output_dir.mkdir(parents=True, exist_ok=True)
    save_row_legend(str(output_dir / "ler_row_legend.pdf"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--part", choices=["all", "a", "connectivity", "high", "layouts"], default="all",
                    help="a = compilation effects; connectivity = b, c (d = 7); high = d (HIGH_KS); "
                         "layouts = d) mapping/routing only (compiles, no simulation)")
    ap.add_argument("--quick", action="store_true", help="panel a): d=3,5, two error rates, 120 s compile limit")
    ap.add_argument("--distances-a", type=int, nargs="+", default=None,
                    help=f"code distances shown in a) (default {PLOT_DISTANCES_A}; 0 = all simulated)")
    ap.add_argument("--distances-d", type=int, nargs="+", default=None,
                    help=f"code distances shown in d) (default {PLOT_DISTANCES_D}; 0 = all simulated)")
    a = ap.parse_args()
    if a.distances_a:
        PLOT_DISTANCES_A = None if a.distances_a == [0] else tuple(a.distances_a)
    if a.distances_d:
        PLOT_DISTANCES_D = None if a.distances_d == [0] else tuple(a.distances_d)
    if a.part in ("all", "a"):
        run_exp_distributed_lattice_surgery(reproduce=not a.plot_only, quick=a.quick)
    if a.part != "a":
        inter_part = {"all": "all", "connectivity": "low", "high": "high", "layouts": "layouts"}[a.part]
        run_exp_distributed_inter_chiplet(reproduce=not a.plot_only, part=inter_part)