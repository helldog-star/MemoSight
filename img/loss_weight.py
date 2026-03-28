import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.interpolate import make_interp_spline

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.facecolor": "#F5F3EE",
    "figure.facecolor": "#F5F3EE",
    "grid.color": "white",
    "grid.linewidth": 1.2,
    "axes.linewidth": 0,
    "xtick.bottom": False,
    "ytick.left": False,
})

BLUE  = "#2E5FA3"
CORAL = "#D4622A"

# annotation customization
SHOW_POINT_LABELS = True
LABEL_DECIMALS = 2
LABEL_Y_OFFSET = 0.22
LABEL_FONT_SIZE = 9
LABEL_FONT_WEIGHT = "semibold"
LABEL_COLOR_07 = BLUE
LABEL_COLOR_05 = CORAL

labels = ["c=2", "c=4", "c=6", "c=8"]
x      = np.array([0, 1, 2, 3], dtype=float)
acc_07 = np.array([68.41, 65.52, 66.19, 64.71])
acc_05 = np.array([66.82, 65.47, 63.29, 63.15])

# smooth spline interpolation
x_smooth = np.linspace(x.min(), x.max(), 300)
spl_07 = make_interp_spline(x, acc_07, k=3)
spl_05 = make_interp_spline(x, acc_05, k=3)
y_smooth_07 = spl_07(x_smooth)
y_smooth_05 = spl_05(x_smooth)

fig, ax = plt.subplots(figsize=(7.5, 3.8))

ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)

# smooth lines
ax.plot(x_smooth, y_smooth_07, color=BLUE, linewidth=2.2, zorder=3)
ax.plot(x_smooth, y_smooth_05, color=CORAL, linewidth=2.2, zorder=3,
        linestyle=(0, (6, 3)))

# dots on actual data points
ax.scatter(x, acc_07, color=BLUE,  s=55, zorder=4)
ax.scatter(x, acc_05, color=CORAL, s=55, zorder=4)

if SHOW_POINT_LABELS:
    # annotate each point with value text; tweak globals above for custom style
    for xi, yi in zip(x, acc_07):
        ax.annotate(
            f"{yi:.{LABEL_DECIMALS}f}%",
            xy=(xi, yi),
            xytext=(0, LABEL_Y_OFFSET * 72),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=LABEL_FONT_SIZE,
            fontweight=LABEL_FONT_WEIGHT,
            color=LABEL_COLOR_07,
            zorder=5,
        )
    for xi, yi in zip(x, acc_05):
        ax.annotate(
            f"{yi:.{LABEL_DECIMALS}f}%",
            xy=(xi, yi),
            xytext=(0, -LABEL_Y_OFFSET * 72),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=LABEL_FONT_SIZE,
            fontweight=LABEL_FONT_WEIGHT,
            color=LABEL_COLOR_05,
            zorder=5,
        )

# legend proxy lines
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=BLUE,  linewidth=2.2, marker="o", markersize=6,
           label="LM=0.7, MTP=0.3"),
    Line2D([0], [0], color=CORAL, linewidth=2.2, marker="o", markersize=6,
           linestyle=(0, (6, 3)), label="LM=0.5, MTP=0.5"),
]
ax.legend(handles=legend_elements, fontsize=10, frameon=False,
          loc="upper center", bbox_to_anchor=(0.45, 1.18),
          ncol=2, handlelength=2.4, handletextpad=0.5, columnspacing=1.5)

ax.set_title("Average accuracy (%)", fontsize=11, color="#555", pad=8, loc="center")
ax.set_ylim(62.0, 70.5)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
ax.tick_params(axis="y", labelsize=10, colors="#555", length=0)
ax.tick_params(axis="x", labelsize=10, colors="#555", length=0)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlim(-0.4, len(labels) - 0.6)

for sp in ax.spines.values():
    sp.set_visible(False)

plt.tight_layout()
plt.savefig("loss_weight_smooth.pdf", bbox_inches="tight", dpi=300)
plt.savefig("loss_weight_smooth.png", bbox_inches="tight", dpi=300)
print("Saved.")