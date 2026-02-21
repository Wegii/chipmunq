import matplotlib.pyplot as plt
import numpy as np


def plot_compilation_comparison(tqec_et, tqec_topologiq_et, el_loom_et, filename: str = ""):
    
    section_titles = ["tqec", "tqec + topologiq", "el-loom"]

    compiled = [tqec_et[0], tqec_topologiq_et[0], el_loom_et[0]]
    compiled_mapped = [tqec_et[1], tqec_topologiq_et[1], el_loom_et[1]]

    # Extract values
    x = np.arange(3)
    width = 0.35

    # Pastel colors
    pastel_blue = "#5c79bd"
    pastel_orange = "#ef8d38"

    # Create depth statistics
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.bar(x, compiled_mapped, width,
        label="Compilation+Mapping", color=pastel_orange,
        hatch="o", edgecolor="black")

    ax.bar(x, compiled, width,
           label="Compilation", color=pastel_blue,
           hatch="/", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(section_titles)
    ax.set_xlabel("Compilation Framework")
    ax.set_ylabel("Execution Time [s]")
    ax.legend()

    # Add annotation
    ax.text(-0.025, 1.05, "Lower is better ↓",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left")

    fig.tight_layout()
    fig.savefig(f"{filename}", dpi=300)
    plt.show()


def perform_surgery_comparison():
    # TODO: Show time taken for compiling 10 logical qubits
    # TODO: Show time taken for mapping the generated circuit

    # TODO: Create timing for different surgery frameworks

    tqec_et = [10, 15]
    tqec_topologiq_et = [20, 25]
    el_loom_et = [15, 20]

    plot_compilation_comparison(tqec_et,
                                tqec_topologiq_et,
                                el_loom_et,
                                "experiments/evaluation/scalability/surgery_compilation_comparison.png")

if __name__ == "__main__":
    perform_surgery_comparison()