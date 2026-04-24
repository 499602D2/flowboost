import json
import itertools
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable

# --- Load ---
data_path = Path(__file__).parent / "flowboost_data/designs.json"
with open(data_path) as f:
    data = json.load(f)

designs = sorted(data["designs"], key=lambda d: d["created_at"])
output_dir = Path(__file__).parent / "figures"
output_dir.mkdir(exist_ok=True)

param_keys = list({k for d in designs for k in d.get("parameters", {})})
obj_keys = list({k for d in designs for k in d.get("objectives", {})})
con_keys = list({k for d in designs for k in d.get("constraints", {})})

CMAP = plt.cm.viridis
ALPHA_FEAS = 1.0
ALPHA_INFEAS = 0.2
VMIN = 0
VMAX = len(designs) - 1
NORM = mcolors.Normalize(vmin=VMIN, vmax=VMAX)


def get_param(d, key):
    return d.get("parameters", {}).get(key)


def get_obj(d, key):
    return d.get("objectives", {}).get(key, {}).get("value")


def get_con(d, key):
    return d.get("constraints", {}).get(key, {}).get("value")


def feasible(d):
    for key, con in d.get("constraints", {}).items():
        v = con.get("value")
        if v is None:
            continue
        if con.get("gte") is not None and v < con["gte"]:
            return False
        if con.get("lte") is not None and v > con["lte"]:
            return False
    return True


def is_dominated(vals, i, minimizes):
    for j, other in enumerate(vals):
        if j == i:
            continue
        dominates = True
        strictly = False
        for k, (vi, vj) in enumerate(zip(vals[i], other)):
            if minimizes[k]:
                if vj > vi:
                    dominates = False
                    break
                if vj < vi:
                    strictly = True
            else:
                if vj < vi:
                    dominates = False
                    break
                if vj > vi:
                    strictly = True
        if dominates and strictly:
            return True
    return False


def add_colorbar(fig, ax):
    sm = ScalarMappable(cmap=CMAP, norm=NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Design index (older → newer)")


def scatter_with_feasibility(ax, xs, ys, indices, feas_flags):
    f_x = [x for x, f in zip(xs, feas_flags) if f]
    f_y = [y for y, f in zip(ys, feas_flags) if f]
    f_i = [i for i, f in zip(indices, feas_flags) if f]
    if_x = [x for x, f in zip(xs, feas_flags) if not f]
    if_y = [y for y, f in zip(ys, feas_flags) if not f]
    if_i = [i for i, f in zip(indices, feas_flags) if not f]
    if if_x:
        ax.scatter(
            if_x,
            if_y,
            c=if_i,
            cmap=CMAP,
            norm=NORM,
            alpha=ALPHA_INFEAS,
            zorder=2,
            edgecolors="none",
        )
    if f_x:
        ax.scatter(
            f_x,
            f_y,
            c=f_i,
            cmap=CMAP,
            norm=NORM,
            alpha=ALPHA_FEAS,
            zorder=3,
            edgecolors="none",
        )


# ===========================================================================
# STATIC FIGURES
# ===========================================================================

for obj_key in obj_keys:
    for param_key in param_keys:
        valid = [
            (i, d)
            for i, d in enumerate(designs)
            if get_param(d, param_key) is not None and get_obj(d, obj_key) is not None
        ]
        indices = [i for i, _ in valid]
        xs = [get_param(d, param_key) for _, d in valid]
        ys = [get_obj(d, obj_key) for _, d in valid]
        feas = [feasible(d) for _, d in valid]

        fig, ax = plt.subplots()
        scatter_with_feasibility(ax, xs, ys, indices, feas)
        add_colorbar(fig, ax)
        ax.set_xlabel(param_key)
        ax.set_ylabel(obj_key)
        ax.set_title(f"{obj_key} vs {param_key}")
        ax.grid(True)
        fname = output_dir / f"obj_{obj_key}_vs_{param_key}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")

for con_key in con_keys:
    for param_key in param_keys:
        valid = [
            (i, d)
            for i, d in enumerate(designs)
            if get_param(d, param_key) is not None and get_con(d, con_key) is not None
        ]
        indices = [i for i, _ in valid]
        xs = [get_param(d, param_key) for _, d in valid]
        ys = [get_con(d, con_key) for _, d in valid]
        feas = [feasible(d) for _, d in valid]

        gte = next(
            (
                d.get("constraints", {}).get(con_key, {}).get("gte")
                for d in designs
                if d.get("constraints", {}).get(con_key, {}).get("gte") is not None
            ),
            None,
        )
        lte = next(
            (
                d.get("constraints", {}).get(con_key, {}).get("lte")
                for d in designs
                if d.get("constraints", {}).get(con_key, {}).get("lte") is not None
            ),
            None,
        )

        fig, ax = plt.subplots()
        scatter_with_feasibility(ax, xs, ys, indices, feas)
        add_colorbar(fig, ax)
        handles = []
        if gte is not None:
            ax.axhline(gte, color="blue", linestyle="--")
            handles.append(
                Line2D([0], [0], color="blue", linestyle="--", label=f"gte={gte}")
            )
        if lte is not None:
            ax.axhline(lte, color="orange", linestyle="--")
            handles.append(
                Line2D([0], [0], color="orange", linestyle="--", label=f"lte={lte}")
            )
        ax.set_xlabel(param_key)
        ax.set_ylabel(con_key)
        ax.set_title(f"{con_key} vs {param_key}")
        ax.grid(True)
        if handles:
            ax.legend(handles=handles, loc="upper right")
        fname = output_dir / f"con_{con_key}_vs_{param_key}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")

# --- Pareto front static ---
feasible_designs = [d for d in designs if feasible(d)]

for obj_a, obj_b in itertools.combinations(obj_keys, 2):
    valid_feas = [
        (designs.index(d), d)
        for d in feasible_designs
        if get_obj(d, obj_a) is not None and get_obj(d, obj_b) is not None
    ]
    feas_vals = [(get_obj(d, obj_a), get_obj(d, obj_b)) for _, d in valid_feas]

    min_a = designs[0].get("objectives", {}).get(obj_a, {}).get("minimize", True)
    min_b = designs[0].get("objectives", {}).get(obj_b, {}).get("minimize", True)
    minimizes = [min_a, min_b]

    pareto_mask = [
        not is_dominated(feas_vals, i, minimizes) for i in range(len(feas_vals))
    ]

    fig, ax = plt.subplots()
    inf_designs = [
        (designs.index(d), d)
        for d in designs
        if not feasible(d)
        and get_obj(d, obj_a) is not None
        and get_obj(d, obj_b) is not None
    ]
    if inf_designs:
        ax.scatter(
            [get_obj(d, obj_a) for _, d in inf_designs],
            [get_obj(d, obj_b) for _, d in inf_designs],
            c=[i for i, _ in inf_designs],
            cmap=CMAP,
            norm=NORM,
            alpha=ALPHA_INFEAS,
            zorder=2,
            edgecolors="none",
        )
    non_pareto = [(idx, d) for (idx, d), p in zip(valid_feas, pareto_mask) if not p]
    if non_pareto:
        ax.scatter(
            [get_obj(d, obj_a) for _, d in non_pareto],
            [get_obj(d, obj_b) for _, d in non_pareto],
            c=[i for i, _ in non_pareto],
            cmap=CMAP,
            norm=NORM,
            alpha=ALPHA_FEAS,
            zorder=3,
            edgecolors="none",
        )
    pareto_pts = [(idx, d) for (idx, d), p in zip(valid_feas, pareto_mask) if p]
    if pareto_pts:
        px = [get_obj(d, obj_a) for _, d in pareto_pts]
        py = [get_obj(d, obj_b) for _, d in pareto_pts]
        ax.scatter(
            px,
            py,
            c=[i for i, _ in pareto_pts],
            cmap=CMAP,
            norm=NORM,
            alpha=ALPHA_FEAS,
            zorder=4,
            edgecolors="black",
            linewidths=1.2,
        )
        sp = sorted(zip(px, py))
        ax.plot(
            [p[0] for p in sp], [p[1] for p in sp], c="gold", linestyle="--", zorder=3
        )
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="grey",
                    markeredgecolor="black",
                    linestyle="None",
                    label="Pareto front",
                )
            ],
            loc="upper right",
        )
    add_colorbar(fig, ax)
    ax.set_xlabel(f"{obj_a} ({'min' if min_a else 'max'})")
    ax.set_ylabel(f"{obj_b} ({'min' if min_b else 'max'})")
    ax.set_title(f"Pareto Front: {obj_a} vs {obj_b}")
    ax.grid(True)
    fname = output_dir / f"pareto_{obj_a}_vs_{obj_b}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")

# ===========================================================================
# ANIMATIONS
# ===========================================================================


def make_scatter_animation(
    all_xs, all_ys, all_indices, all_feas, xlabel, ylabel, title, fname, hlines=None
):
    pad_x = (max(all_xs) - min(all_xs)) * 0.1 or 1
    pad_y = (max(all_ys) - min(all_ys)) * 0.1 or 1

    fig, ax = plt.subplots()
    ax.set_xlim(min(all_xs) - pad_x, max(all_xs) + pad_x)
    ax.set_ylim(min(all_ys) - pad_y, max(all_ys) + pad_y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

    if hlines:
        handles = []
        for hy, hc, hl in hlines:
            ax.axhline(hy, color=hc, linestyle="--")
            handles.append(Line2D([0], [0], color=hc, linestyle="--", label=hl))
        ax.legend(handles=handles, loc="upper right")

    # Use c=[] + vmin/vmax to avoid UserWarning about missing color data
    scat_inf = ax.scatter(
        [],
        [],
        c=[],
        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
        alpha=ALPHA_INFEAS,
        zorder=2,
        edgecolors="none",
    )
    scat_feas = ax.scatter(
        [],
        [],
        c=[],
        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
        alpha=ALPHA_FEAS,
        zorder=3,
        edgecolors="none",
    )
    step_text = ax.text(
        0.02, 0.03, "", transform=ax.transAxes, verticalalignment="bottom", fontsize=9
    )
    add_colorbar(fig, ax)

    def init():
        scat_inf.set_offsets(np.empty((0, 2)))
        scat_inf.set_array(np.array([], dtype=float))
        scat_feas.set_offsets(np.empty((0, 2)))
        scat_feas.set_array(np.array([], dtype=float))
        step_text.set_text("")
        return scat_inf, scat_feas, step_text

    def update(frame):
        n = frame + 1
        xs_n = all_xs[:n]
        ys_n = all_ys[:n]
        idx_n = all_indices[:n]
        feas_n = all_feas[:n]

        if_pts = [(x, y, i) for x, y, i, f in zip(xs_n, ys_n, idx_n, feas_n) if not f]
        f_pts = [(x, y, i) for x, y, i, f in zip(xs_n, ys_n, idx_n, feas_n) if f]

        if if_pts:
            scat_inf.set_offsets(np.array([(x, y) for x, y, _ in if_pts]))
            scat_inf.set_array(np.array([i for _, _, i in if_pts], dtype=float))
        else:
            scat_inf.set_offsets(np.empty((0, 2)))
            scat_inf.set_array(np.array([], dtype=float))

        if f_pts:
            scat_feas.set_offsets(np.array([(x, y) for x, y, _ in f_pts]))
            scat_feas.set_array(np.array([i for _, _, i in f_pts], dtype=float))
        else:
            scat_feas.set_offsets(np.empty((0, 2)))
            scat_feas.set_array(np.array([], dtype=float))

        step_text.set_text(f"Designs: {n}/{len(all_xs)}")
        return scat_inf, scat_feas, step_text

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(all_xs),
        init_func=init,
        blit=True,
        interval=600,
        repeat_delay=1500,
    )
    ani.save(fname, writer="pillow", dpi=120)
    plt.close(fig)
    print(f"Saved {fname}")


# Animate objectives vs parameters
for obj_key in obj_keys:
    for param_key in param_keys:
        valid = [
            (i, d)
            for i, d in enumerate(designs)
            if get_param(d, param_key) is not None and get_obj(d, obj_key) is not None
        ]
        indices = [i for i, _ in valid]
        xs = [get_param(d, param_key) for _, d in valid]
        ys = [get_obj(d, obj_key) for _, d in valid]
        feas = [feasible(d) for _, d in valid]
        fname = output_dir / f"obj_{obj_key}_vs_{param_key}.gif"
        make_scatter_animation(
            xs,
            ys,
            indices,
            feas,
            xlabel=param_key,
            ylabel=obj_key,
            title=f"{obj_key} vs {param_key}",
            fname=fname,
        )

# Animate constraints vs parameters
for con_key in con_keys:
    for param_key in param_keys:
        valid = [
            (i, d)
            for i, d in enumerate(designs)
            if get_param(d, param_key) is not None and get_con(d, con_key) is not None
        ]
        indices = [i for i, _ in valid]
        xs = [get_param(d, param_key) for _, d in valid]
        ys = [get_con(d, con_key) for _, d in valid]
        feas = [feasible(d) for _, d in valid]

        gte = next(
            (
                d.get("constraints", {}).get(con_key, {}).get("gte")
                for d in designs
                if d.get("constraints", {}).get(con_key, {}).get("gte") is not None
            ),
            None,
        )
        lte = next(
            (
                d.get("constraints", {}).get(con_key, {}).get("lte")
                for d in designs
                if d.get("constraints", {}).get(con_key, {}).get("lte") is not None
            ),
            None,
        )
        hlines = []
        if gte is not None:
            hlines.append((gte, "blue", f"gte={gte}"))
        if lte is not None:
            hlines.append((lte, "orange", f"lte={lte}"))

        fname = output_dir / f"con_{con_key}_vs_{param_key}.gif"
        make_scatter_animation(
            xs,
            ys,
            indices,
            feas,
            xlabel=param_key,
            ylabel=con_key,
            title=f"{con_key} vs {param_key}",
            fname=fname,
            hlines=hlines or None,
        )

# Animate Pareto front
for obj_a, obj_b in itertools.combinations(obj_keys, 2):
    min_a = designs[0].get("objectives", {}).get(obj_a, {}).get("minimize", True)
    min_b = designs[0].get("objectives", {}).get(obj_b, {}).get("minimize", True)
    minimizes = [min_a, min_b]

    valid = [
        (i, d, feasible(d))
        for i, d in enumerate(designs)
        if get_obj(d, obj_a) is not None and get_obj(d, obj_b) is not None
    ]
    all_xs = [get_obj(d, obj_a) for _, d, _ in valid]
    all_ys = [get_obj(d, obj_b) for _, d, _ in valid]
    all_idx = [i for i, _, _ in valid]
    all_feas = [f for _, _, f in valid]

    pad_x = (max(all_xs) - min(all_xs)) * 0.1 or 1
    pad_y = (max(all_ys) - min(all_ys)) * 0.1 or 1

    fig, ax = plt.subplots()
    ax.set_xlim(min(all_xs) - pad_x, max(all_xs) + pad_x)
    ax.set_ylim(min(all_ys) - pad_y, max(all_ys) + pad_y)
    ax.set_xlabel(f"{obj_a} ({'min' if min_a else 'max'})")
    ax.set_ylabel(f"{obj_b} ({'min' if min_b else 'max'})")
    ax.set_title(f"Pareto Front: {obj_a} vs {obj_b}")
    ax.grid(True)

    scat_inf = ax.scatter(
        [],
        [],
        c=[],
        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
        alpha=ALPHA_INFEAS,
        zorder=2,
        edgecolors="none",
    )
    scat_feas = ax.scatter(
        [],
        [],
        c=[],
        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
        alpha=ALPHA_FEAS,
        zorder=3,
        edgecolors="none",
    )
    scat_pareto = ax.scatter(
        [],
        [],
        c=[],
        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
        alpha=ALPHA_FEAS,
        zorder=4,
        edgecolors="black",
        linewidths=1.2,
    )
    (pareto_line,) = ax.plot([], [], c="gold", linestyle="--", zorder=3)
    step_text = ax.text(
        0.02, 0.03, "", transform=ax.transAxes, verticalalignment="bottom", fontsize=9
    )
    add_colorbar(fig, ax)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="grey",
                markeredgecolor="black",
                linestyle="None",
                label="Pareto front",
            )
        ],
        loc="upper right",
    )

    def make_pareto_update(all_xs, all_ys, all_idx, all_feas, minimizes):
        def update(frame):
            n = frame + 1
            xs_n = all_xs[:n]
            ys_n = all_ys[:n]
            idx_n = all_idx[:n]
            feas_n = all_feas[:n]

            inf_data = [
                (x, y, i) for x, y, i, f in zip(xs_n, ys_n, idx_n, feas_n) if not f
            ]
            feas_data = [
                (x, y, i) for x, y, i, f in zip(xs_n, ys_n, idx_n, feas_n) if f
            ]

            if inf_data:
                scat_inf.set_offsets(np.array([(x, y) for x, y, _ in inf_data]))
                scat_inf.set_array(np.array([i for _, _, i in inf_data], dtype=float))
            else:
                scat_inf.set_offsets(np.empty((0, 2)))
                scat_inf.set_array(np.array([], dtype=float))

            feas_vals = [(x, y) for x, y, _ in feas_data]
            if feas_vals:
                pmask = [
                    not is_dominated(feas_vals, i, minimizes)
                    for i in range(len(feas_vals))
                ]
                non_p = [(x, y, i) for (x, y, i), m in zip(feas_data, pmask) if not m]
                par_p = [(x, y, i) for (x, y, i), m in zip(feas_data, pmask) if m]

                if non_p:
                    scat_feas.set_offsets(np.array([(x, y) for x, y, _ in non_p]))
                    scat_feas.set_array(np.array([i for _, _, i in non_p], dtype=float))
                else:
                    scat_feas.set_offsets(np.empty((0, 2)))
                    scat_feas.set_array(np.array([], dtype=float))

                if par_p:
                    scat_pareto.set_offsets(np.array([(x, y) for x, y, _ in par_p]))
                    scat_pareto.set_array(
                        np.array([i for _, _, i in par_p], dtype=float)
                    )
                    sp = sorted([(x, y) for x, y, _ in par_p])
                    pareto_line.set_data([p[0] for p in sp], [p[1] for p in sp])
                else:
                    scat_pareto.set_offsets(np.empty((0, 2)))
                    scat_pareto.set_array(np.array([], dtype=float))
                    pareto_line.set_data([], [])
            else:
                for s in (scat_feas, scat_pareto):
                    s.set_offsets(np.empty((0, 2)))
                    s.set_array(np.array([], dtype=float))
                pareto_line.set_data([], [])

            step_text.set_text(f"Designs: {n}/{len(all_xs)}")
            return scat_inf, scat_feas, scat_pareto, pareto_line, step_text

        return update

    ani = animation.FuncAnimation(
        fig,
        make_pareto_update(all_xs, all_ys, all_idx, all_feas, minimizes),
        frames=len(valid),
        interval=600,
        repeat_delay=1500,
        blit=True,
    )
    fname = output_dir / f"pareto_{obj_a}_vs_{obj_b}.gif"
    ani.save(fname, writer="pillow", dpi=120)
    plt.close(fig)
    print(f"Saved {fname}")

print("Done.")

# ===========================================================================
# BEST DESIGN SUMMARY
# ===========================================================================
print("\n--- Best Feasible Designs ---")
feasible_all = [d for d in designs if feasible(d)]

if not feasible_all:
    print("No feasible designs found.")
else:
    for obj_key in obj_keys:
        valid = [d for d in feasible_all if get_obj(d, obj_key) is not None]
        if not valid:
            continue
        minimize = valid[0].get("objectives", {}).get(obj_key, {}).get("minimize", True)
        best = (
            min(valid, key=lambda d: get_obj(d, obj_key))
            if minimize
            else max(valid, key=lambda d: get_obj(d, obj_key))
        )

        print(f"\nBest for '{obj_key}' ({'minimize' if minimize else 'maximize'}):")
        print(f"  Name       : {best['name']}")
        print(f"  Created at : {best['created_at']}")
        print("  Parameters :")
        for k, v in best.get("parameters", {}).items():
            print(f"    {k}: {v}")
        print("  Objectives :")
        for k, v in best.get("objectives", {}).items():
            print(f"    {k}: {v.get('value')}")
        print("  Constraints:")
        for k, v in best.get("constraints", {}).items():
            print(
                f"    {k}: {v.get('value')}  (gte={v.get('gte')}, lte={v.get('lte')})"
            )
