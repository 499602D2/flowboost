"""
Plot results from the pitzDaily optimization trial.

Reads case data from the working directory and generates:
  1. Search space scatter (k vs epsilon, colored by inlet pressure)
  2. Pressure vs k, grouped by epsilon
  3. Pressure vs epsilon, grouped by k
  4. Pressure field contour for the best case (requires pyvista)

Usage:
    python plot_results.py [data_dir]

    data_dir defaults to ./flowboost_data
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from flowboost.openfoam.case import Case
from flowboost.openfoam.data import Data
from flowboost.openfoam.interface import run_command
from flowboost.openfoam.runtime import get_runtime


def collect_results(workdir: Path) -> list[dict]:
    """Read k, epsilon, and inlet pressure from all completed cases."""
    case_dirs = sorted(workdir.glob("cases_*/job_*"))
    if not case_dirs:
        case_dirs = sorted(workdir.glob("case_*"))

    case_dirs = [d for d in case_dirs if (d / "log.foamRun").exists()]
    if not case_dirs:
        print("No completed cases found.")
        return []

    # Find common parent for a single mount
    mount_root = case_dirs[0].parent
    for d in case_dirs:
        while not d.is_relative_to(mount_root):
            mount_root = mount_root.parent

    results = []
    with get_runtime().container(mount_root):
        for case_dir in case_dirs:
            d = Data(case_dir)
            fo = "patchAverage(patch=inlet,fields=(pU))"
            df = d.simple_function_object_reader(fo)
            if df is None or len(df) == 0:
                continue

            case = Case(case_dir)
            k = float(
                str(case.dictionary("0/k").entry("boundaryField/inlet/value").value)
            )
            eps = float(
                str(
                    case.dictionary("0/epsilon")
                    .entry("boundaryField/inlet/value")
                    .value
                )
            )
            p = float(df["areaAverage(p)"][-1])
            results.append({"k": k, "epsilon": eps, "p": p, "path": case_dir})

    print(f"Collected {len(results)} results")
    return results


def plot_search_space(results: list[dict], out: Path):
    k = np.array([r["k"] for r in results])
    eps = np.array([r["epsilon"] for r in results])
    p = np.array([r["p"] for r in results])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sc = axes[0].scatter(
        k, eps, c=p, cmap="RdYlBu_r", s=80, edgecolors="k", linewidth=0.5
    )
    axes[0].set_xlabel("Inlet k [m²/s²]")
    axes[0].set_ylabel("Inlet ε [m²/s³]")
    axes[0].set_title("Search Space — Inlet Pressure")
    axes[0].set_yscale("log")
    fig.colorbar(sc, ax=axes[0], label="Avg. Inlet Pressure [Pa]")

    for ev in sorted(set(eps)):
        mask = eps == ev
        idx = np.argsort(k[mask])
        axes[1].plot(
            k[mask][idx], p[mask][idx], "o-", label=f"ε={ev:.2f}", markersize=5
        )
    axes[1].set_xlabel("Inlet k [m²/s²]")
    axes[1].set_ylabel("Avg. Inlet Pressure [Pa]")
    axes[1].set_title("Pressure vs. k (by ε)")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)

    for kv in sorted(set(k)):
        mask = k == kv
        idx = np.argsort(eps[mask])
        axes[2].plot(
            eps[mask][idx], p[mask][idx], "s-", label=f"k={kv:.3f}", markersize=5
        )
    axes[2].set_xlabel("Inlet ε [m²/s³]")
    axes[2].set_ylabel("Avg. Inlet Pressure [Pa]")
    axes[2].set_title("Pressure vs. ε (by k)")
    axes[2].set_xscale("log")
    axes[2].legend(fontsize=7, ncol=2)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved search space plot to {out}")
    plt.show()


def plot_pressure_field(case_dir: Path, out: Path):
    """Convert the case to VTK via foamToVTK and plot the pressure field."""
    try:
        import pyvista as pv
    except ImportError:
        print("pyvista not installed — skipping pressure field plot")
        print("  Install with: uv add flowboost[viz]")
        return

    case_dir = case_dir.resolve()

    # Convert to VTK inside Docker
    with get_runtime().container(case_dir):
        run_command(["foamToVTK"], cwd=case_dir)

    # Find the last timestep VTK file (highest index)
    vtk_files = sorted(case_dir.glob("VTK/work_*.vtk"))
    if not vtk_files:
        print("No VTK files found after foamToVTK")
        return

    mesh = pv.read(vtk_files[-1])

    # Slice at z=0 for 2D view
    slc = mesh.slice(normal="z", origin=(0, 0, 0))
    centers = slc.cell_centers()
    x, y = centers.points[:, 0], centers.points[:, 1]
    p = slc.cell_data["p"]

    fig, ax = plt.subplots(figsize=(14, 4))
    tc = ax.tricontourf(x, y, p, levels=50, cmap="RdYlBu_r")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Pressure field — {case_dir.name}")
    fig.colorbar(tc, ax=ax, label="p [Pa]")
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved pressure field plot to {out}")
    plt.show()


if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("flowboost_data")
    results = collect_results(data_dir)
    if not results:
        sys.exit(1)

    plot_search_space(results, data_dir / "results.png")

    # Plot pressure field for the best case (lowest inlet pressure)
    best = min(results, key=lambda r: r["p"])
    print(f"Best case: {best['path'].name} (p={best['p']:.4f})")
    plot_pressure_field(best["path"], data_dir / "pressure_field.png")
