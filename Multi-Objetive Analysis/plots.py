# plot_fronts_and_parallel_metrics_combined.py
# ------------------------------------------------------------
# Outputs per K:
#   (A) fronts_k{K}_point_to_point.png
#   (B) parallel_metrics_k{K}.png                (classic parallel coordinates)
#   (C) combined_k{K}_fronts_plus_parallel.png   (fronts + parallel)
#
# Parallel plot:
# - vertical axes = metrics
# - one polyline per method
# - metrics normalized to [0,1] within each K across methods
# - oriented so "higher is better"
# - baseline highlighted (black, dotted, thicker)
# ------------------------------------------------------------

from __future__ import annotations

import os
import csv
import math
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt


# -----------------------------
# PATHS
# -----------------------------
ROOT = r"G:\Mi unidad\Paper2\Seeds\Results\Global"

BUDGET_DIRS: Dict[int, str] = {
    5:  os.path.join(ROOT, "5_mse_Pareto_Archives"),
    10: os.path.join(ROOT, "10_mse_Pareto_Archives"),
    15: os.path.join(ROOT, "15_mse_Pareto_Archives"),
}

PQ20_CSV = os.path.join(ROOT, "20_mse_Pareto_Archives", "pq_baseline", "pareto_global.csv")

OUT_DIR = os.path.join(ROOT, "comparative_metrics", "plots", "fronts_plus_parallel_metrics")
os.makedirs(OUT_DIR, exist_ok=True)

# Metrics CSVs produced by your metrics runner ("brain")
METRICS_DIR = os.path.join(ROOT, "comparative_metrics")
METRICS_CSV_BY_K = {
    5:  os.path.join(METRICS_DIR, "metrics_k5.csv"),
    10: os.path.join(METRICS_DIR, "metrics_k10.csv"),
    15: os.path.join(METRICS_DIR, "metrics_k15.csv"),
}

# Representative MO metrics for parallel coordinates
# (csv_key, axis_label, direction) direction: "up" or "down"
PARALLEL_METRICS = [
    ("igd_plus",   "IGD+",       "down"),
    ("hv_2d_norm", "HV (norm)",  "up"),
    ("epsilon",    "ε-add",      "down"),
    ("hausdorff",  "Hausdorff",  "down"),
    ("r2",         "R2",         "down"),
]

BASELINE_METHOD_NAME = "baseline_pq20"  # adjust to your CSV label if needed


# -----------------------------
# CSV I/O
# -----------------------------
def read_csv_rows(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_points_minspace(path: str) -> List[Tuple[float, float]]:
    """
    Return points in MINIMIZATION space:
      x = -PSNR
      y = Params
    Expects CSV columns: PSNR, Params
    """
    rows = read_csv_rows(path)
    pts: List[Tuple[float, float]] = []
    for i, r in enumerate(rows):
        try:
            x = -float(r["PSNR"])
            y = float(r["Params"])
            pts.append((x, y))
        except KeyError as e:
            raise KeyError(f"{path}: missing column {e} (row {i}). Expected 'PSNR' and 'Params'.")
        except ValueError as e:
            raise ValueError(f"{path}: invalid numeric value on row {i}: {e}")
    return pts


def discover_method_csvs(budget_dir: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not os.path.isdir(budget_dir):
        return out
    for m in sorted(os.listdir(budget_dir)):
        p = os.path.join(budget_dir, m, "pareto_global.csv")
        if os.path.isfile(p):
            out[m] = p
    return out


def sort_points_for_polyline(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Sort by x then y for stable point-to-point polyline rendering."""
    return sorted(pts, key=lambda t: (t[0], t[1]))


# -----------------------------
# Numeric helpers
# -----------------------------
def safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def finite_minmax(values: List[float]) -> Tuple[float, float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return (float("nan"), float("nan"))
    return (min(finite), max(finite))


def norm01(v: float, vmin: float, vmax: float) -> float:
    if not (math.isfinite(v) and math.isfinite(vmin) and math.isfinite(vmax)):
        return float("nan")
    if vmax <= vmin:
        return 0.5
    return (v - vmin) / (vmax - vmin)


def orient_to_higher_better(n01: float, direction: str) -> float:
    if not math.isfinite(n01):
        return float("nan")
    if direction == "up":
        return n01
    if direction == "down":
        return 1.0 - n01
    raise ValueError(f"Unknown direction: {direction}")


# -----------------------------
# Metrics normalization for a given K
# -----------------------------
def load_and_score_metrics(metrics_csv: str) -> Tuple[List[str], List[str], Dict[str, List[float]]]:
    """
    Returns:
      methods: list of method names (sorted, baseline last)
      axis_labels: list of metric axis labels (same order as PARALLEL_METRICS)
      scores_by_method: dict method -> list of scores in [0,1] where higher is better
    """
    if not os.path.isfile(metrics_csv):
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")

    rows = read_csv_rows(metrics_csv)
    if not rows:
        raise ValueError(f"Metrics CSV is empty: {metrics_csv}")

    metric_keys = [k for (k, _, _) in PARALLEL_METRICS]
    axis_labels = [lab for (_, lab, _) in PARALLEL_METRICS]

    # min/max per metric across all methods in this K
    mins: Dict[str, float] = {}
    maxs: Dict[str, float] = {}
    for key in metric_keys:
        vals = [safe_float(r.get(key)) for r in rows]
        vmin, vmax = finite_minmax(vals)
        mins[key], maxs[key] = vmin, vmax

    # Build method -> scores
    scores_by_method: Dict[str, List[float]] = {}
    for r in rows:
        m = str(r.get("method", "")).strip()
        if not m:
            continue

        svec: List[float] = []
        for (key, _, direction) in PARALLEL_METRICS:
            v = safe_float(r.get(key))
            n01 = norm01(v, mins[key], maxs[key])
            s = orient_to_higher_better(n01, direction)
            svec.append(s)

        scores_by_method[m] = svec

    # Sort methods: baseline last to plot on top
    methods = sorted([m for m in scores_by_method.keys() if m != BASELINE_METHOD_NAME]) + (
        [BASELINE_METHOD_NAME] if BASELINE_METHOD_NAME in scores_by_method else []
    )

    return methods, axis_labels, scores_by_method


# -----------------------------
# Plot A: Pareto fronts point-to-point
# -----------------------------
def plot_fronts(ax, K: int, pq_points: List[Tuple[float, float]], fronts_by_method: Dict[str, List[Tuple[float, float]]]) -> None:
    # Other archivers first
    for method in sorted(fronts_by_method.keys()):
        pts = fronts_by_method[method]
        if not pts:
            continue
        pts_sorted = sort_points_for_polyline(pts)
        x = [p[0] for p in pts_sorted]
        y = [p[1] for p in pts_sorted]
        ax.scatter(x, y, s=18, alpha=0.85, zorder=2)
        ax.plot(x, y, linestyle="-", linewidth=1.4, label=method, zorder=2)

    # Baseline last
    pq_sorted = sort_points_for_polyline(pq_points)
    pq_x = [p[0] for p in pq_sorted]
    pq_y = [p[1] for p in pq_sorted]
    ax.scatter(
        pq_x, pq_y,
        s=60,
        facecolors="none",
        edgecolors="black",
        linewidths=1.8,
        label="baseline",
        zorder=5,
    )
    ax.plot(
        pq_x, pq_y,
        linestyle=":",
        linewidth=2.5,
        color="black",
        label="baseline",
        zorder=4,
    )

    ax.set_xlabel("Inverted PSNR")
    ax.set_ylabel("Parameters")

    ax.grid(True, linewidth=0.3)


def save_single_front_plot(K: int, pq_points, fronts_by_method, out_path: str) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_fronts(ax, K, pq_points, fronts_by_method)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Plot B: Classic parallel coordinates of metrics
# -----------------------------
def plot_parallel_metrics_classic(
    ax,
    K: int,
    methods: List[str],
    axis_labels: List[str],
    scores_by_method: Dict[str, List[float]],
) -> None:
    """
    Classic parallel coordinates:
      - vertical axes at x = 0..M-1
      - y in [0,1] (already normalized & oriented: higher is better)
      - one polyline per method
      - baseline highlighted
    """
    m = len(axis_labels)
    xs = list(range(m))

    # Draw vertical axes
    for x in xs:
        ax.plot([x, x], [0.0, 1.0], linewidth=1.0)

    # Draw polylines: others first
    for method in methods:
        if method == BASELINE_METHOD_NAME:
            continue

        ys = scores_by_method.get(method, [])
        if len(ys) != m:
            continue
        if not any(math.isfinite(v) for v in ys):
            continue

        ys_plot = [v if math.isfinite(v) else 0.0 for v in ys]
        ax.plot(xs, ys_plot, linewidth=1.0, alpha=0.55)

    # Baseline last
    if BASELINE_METHOD_NAME in scores_by_method:
        ys = scores_by_method[BASELINE_METHOD_NAME]
        if len(ys) == m and any(math.isfinite(v) for v in ys):
            ys_plot = [v if math.isfinite(v) else 0.0 for v in ys]
            ax.plot(
                xs, ys_plot,
                linewidth=2.8,
                linestyle=":",
                color="black",
                label="baseline",
                zorder=5,
            )

    ax.set_xlim(-0.2, m - 0.8)
    ax.set_ylim(0.0, 1.0)

    ax.set_xticks(xs)
    ax.set_xticklabels(axis_labels, rotation=25, ha="right")
    ax.set_ylabel("Normalized score (higher is better)")


    ax.grid(True, linewidth=0.3)


def save_single_parallel_plot(K: int, methods, axis_labels, scores_by_method, out_path: str) -> None:
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    plot_parallel_metrics_classic(ax, K, methods, axis_labels, scores_by_method)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left", fontsize=9, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Plot C: Combined
# -----------------------------
def save_combined_plot(K: int, pq_points, fronts_by_method, methods, axis_labels, scores_by_method, out_path: str) -> None:
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    plot_fronts(ax1, K, pq_points, fronts_by_method)
    plot_parallel_metrics_classic(ax2, K, methods, axis_labels, scores_by_method)

    # Legend only on the fronts panel (keeps combined cleaner)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    pq_points = load_points_minspace(PQ20_CSV)
    if not pq_points:
        raise ValueError("PQ baseline is empty.")

    for K, budget_dir in sorted(BUDGET_DIRS.items()):
        # Load fronts
        paths = discover_method_csvs(budget_dir)
        if not paths:
            print(f"[WARN] No method CSVs found for K={K} at {budget_dir}")
            continue
        fronts = {m: load_points_minspace(p) for m, p in paths.items()}

        # Load and score metrics for parallel plot
        metrics_csv = METRICS_CSV_BY_K.get(K)
        if not metrics_csv or not os.path.isfile(metrics_csv):
            print(f"[WARN] Metrics CSV missing for K={K}: {metrics_csv}")
            continue

        methods, axis_labels, scores_by_method = load_and_score_metrics(metrics_csv)

        # (A) Fronts plot
        out_front = os.path.join(OUT_DIR, f"fronts_k{K}_point_to_point.png")
        save_single_front_plot(K, pq_points, fronts, out_front)
        print(f"[DONE] {out_front}")

        # (B) Parallel coordinates plot
        out_par = os.path.join(OUT_DIR, f"parallel_metrics_k{K}.png")
        save_single_parallel_plot(K, methods, axis_labels, scores_by_method, out_par)
        print(f"[DONE] {out_par}")

        # (C) Combined plot
        out_combo = os.path.join(OUT_DIR, f"combined_k{K}_fronts_plus_parallel.png")
        save_combined_plot(K, pq_points, fronts, methods, axis_labels, scores_by_method, out_combo)
        print(f"[DONE] {out_combo}")


if __name__ == "__main__":
    main()
