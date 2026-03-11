import matplotlib.pyplot as plt

# Figure sizes and font
WIDTH_FIGSIZE = 6
HEIGHT_FIGSIZE = 2.2
FONTSIZE = 12
BAR_WIDTH = 0.2  # constant bar width for consistency with size plot
group_spacing = 0.4

plot_lib_color = "#5c79bd"

# Matplotlib rcParams settings
tex_fonts = {
    # Use LaTeX to write all text
    # "text.usetex": True,
    "font.family": "serif",
    # Font sizes
    "axes.labelsize": FONTSIZE,
    "font.size": FONTSIZE,
    "legend.fontsize": FONTSIZE - 2,
    "xtick.labelsize": FONTSIZE - 1,
    "ytick.labelsize": FONTSIZE - 1,
    "axes.titlesize": 10,
    # Line and marker styles
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "lines.markeredgewidth": 1.5,
    "lines.markeredgecolor": "black",
    # Error bar cap size
    "errorbar.capsize": 3,
}

plt.rcParams.update(tex_fonts)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
