import json
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 17,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
})

# -----------------------------
# Load designs data
# -----------------------------
designs_file = Path("flowboost_data/designs.json")
with open(designs_file, "r") as f:
    data = json.load(f)

designs = data["designs"]

# -----------------------------
# Extract relevant data
# -----------------------------
design_data = []

for design in designs:
    params = design.get("parameters", {})
    objectives = design.get("objectives", {})

    aoa = params.get("angleOfAttack")
    speed = params.get("speed")
    LD = objectives.get("L/D", {}).get("value")
    gen_index = design.get("generation_index", "0")

    if aoa is not None and speed is not None and LD is not None:
        design_data.append({
            "gen_index": gen_index,
            "angleOfAttack": aoa,
            "speed": speed,
            "L/D": LD
        })

# Sort by numeric generation index
def gen_to_float(g):
    try:
        return float(g)
    except ValueError:
        return 0.0

design_data.sort(key=lambda x: gen_to_float(x["gen_index"]))

# -----------------------------
# Prepare arrays
# -----------------------------
aoas = [d["angleOfAttack"] for d in design_data]
speeds = [d["speed"] for d in design_data]
LDs = [d["L/D"] for d in design_data]

iterations = list(range(1, len(LDs) + 1))

# Cumulative best (maximize LD)
cumulative_best = []
current_best = LDs[0]
for v in LDs:
    current_best = max(current_best, v)
    cumulative_best.append(current_best)

# -----------------------------
# Fixed global axis limits (NO jitter)
# -----------------------------
aoa_min, aoa_max = min(aoas), max(aoas)
spd_min, spd_max = min(speeds), max(speeds)
LD_min, LD_max = min(LDs), max(LDs)

aoa_pad = 0.1 * (aoa_max - aoa_min if aoa_max != aoa_min else 1.0)
spd_pad = 0.1 * (spd_max - spd_min if spd_max != spd_min else 1.0)
LD_pad = 0.1 * (LD_max - LD_min if LD_max != LD_min else 1.0)

AOA_LIM = (aoa_min - aoa_pad, aoa_max + aoa_pad)
SPD_LIM = (spd_min - spd_pad, spd_max + spd_pad)
LD_LIM = (LD_min - LD_pad, LD_max + LD_pad)

# -----------------------------
# Figure & axes
# -----------------------------
fig = plt.figure(figsize=(12, 18))

ax1 = fig.add_subplot(3, 1, 1)
ax4 = fig.add_subplot(3, 1, (2, 3), projection="3d")

plt.subplots_adjust(hspace=0.35)

# Top plot lines
line1, = ax1.plot([], [], "o-", alpha=0.6, label="Individual runs")
line2, = ax1.plot([], [], "r-", lw=2, label="Best so far")

# -----------------------------
# Top plot formatting
# -----------------------------
ax1.set_xlim(0, len(LDs) + 1)
lmin, lmax = min(LDs), max(LDs)
ax1.set_ylim(
    lmin * 1.1 if lmin < 0 else lmin * 0.9,
    lmax * 1.1 if lmax > 0 else lmax * 0.9
)

ax1.set_xlabel("Iteration", fontsize=14)
ax1.set_ylabel("L/D", fontsize=14)
ax1.set_title("Optimization Progress: LD vs Iteration", fontsize=16)
ax1.legend(loc="lower right",fontsize=12)
ax1.grid(alpha=0.3)
ax1.tick_params(labelsize=12)

# Sobol shading
num_sobol = min(4, len(LDs))
ax1.axvspan(0, num_sobol, alpha=0.15, color="green", label="Sobol")

stats_text = ax1.text(
    0.02, 0.98, "",
    transform=ax1.transAxes,
    va="top",
    fontsize=15,
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
)

# -----------------------------
# Helper function to draw transparent planes
# -----------------------------
def draw_best_planes(ax, aoa_best, speed_best):
    # Plane at AoA = aoa_best (speed × LD)
    speed_grid, LD_grid = np.meshgrid(
        np.linspace(*SPD_LIM, 20),
        np.linspace(*LD_LIM, 20)
    )
    ax.plot_surface(
        np.full_like(speed_grid, aoa_best),
        speed_grid,
        LD_grid,
        alpha=0.15,
        color="red",
        linewidth=0,
        zorder=0
    )

    # Plane at speed = speed_best (AoA × LD)
    aoa_grid, LD_grid2 = np.meshgrid(
        np.linspace(*AOA_LIM, 20),
        np.linspace(*LD_LIM, 20)
    )
    ax.plot_surface(
        aoa_grid,
        np.full_like(aoa_grid, speed_best),
        LD_grid2,
        alpha=0.15,
        color="blue",
        linewidth=0,
        zorder=0
    )

# -----------------------------
# Animation function
# -----------------------------
def animate(frame):
    k = frame + 1

    x = iterations[:k]
    y = LDs[:k]
    best = cumulative_best[:k]

    # Update top plot
    line1.set_data(x, y)
    line2.set_data(x, best)

    best_idx = y.index(max(y))
    aoa_best = aoas[best_idx]
    speed_best = speeds[best_idx]

    # --- 3D plot ---
    ax4.clear()
    ax4.set_xlim(*AOA_LIM)
    ax4.set_ylim(*SPD_LIM)
    ax4.set_zlim(*LD_LIM)

    # Draw transparent planes
    draw_best_planes(ax4, aoa_best, speed_best)

    # Scatter points
    ax4.scatter(
        aoas[:k],
        speeds[:k],
        y,
        c=y,
        cmap="viridis",
        s=80,
        edgecolors="black",
        alpha=0.85
    )

    # Best point
    ax4.scatter(
        [aoa_best],
        [speed_best],
        [y[best_idx]],
        c="red",
        s=150,
        marker="*",
        label=f"Best LD: {max(y):.3f}"
    )

    ax4.set_xlabel("Angle of Attack [deg]", fontsize=17, labelpad=12)
    ax4.set_ylabel("Speed", fontsize=17, labelpad=12)
    ax4.set_zlabel("LD", fontsize=17, labelpad=12)
    ax4.set_title("3D Design Space: AoA × Speed × LD", fontsize=20, pad=20)
    ax4.tick_params(labelsize=15)
    ax4.legend(loc="upper left", fontsize=14)

    phase = "Sobol sampling" if k <= num_sobol else "Bayesian Optimization"
    stats_text.set_text(
        f"Iteration: {k}/{len(LDs)} ({phase})\n"
        f"Current LD: {y[-1]:.4f}\n"
        f"Best LD: {best[-1]:.4f}"
    )

    return line1, line2, stats_text

# -----------------------------
# Create & save animation
# -----------------------------
anim = animation.FuncAnimation(
    fig,
    animate,
    frames=len(LDs),
    interval=500,
    blit=False
)

plt.tight_layout()

print("Saving animation...")
writer = PillowWriter(fps=2)
anim.save("optimization_progress.gif", writer=writer, dpi=150)

plt.savefig("optimization_progress.png", dpi=300, bbox_inches="tight")

print(f"Frames: {len(LDs)}")
print(f"Final best LD: {cumulative_best[-1]:.4f}")