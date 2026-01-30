from __future__ import annotations

import os
import math
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from io_fronts import (
    read_front_csv,
    rows_to_min_points,
    discover_csv_paths,
    Point2D,
    summarize_points,
)
from report import evaluate_methods, write_metrics_csv, pivot_metric_over_k


# -------------------------
# Config
# -------------------------

ROOT = r"G:\Mi unidad\Paper2\Seeds\Results\Global"

BUDGET_DIRS = {
    5:  os.path.join(ROOT, "5_mse_Pareto_Archives"),
    10: os.path.join(ROOT, "10_mse_Pareto_Archives"),
    15: os.path.join(ROOT, "15_mse_Pareto_Archives"),
    20: os.path.join(ROOT, "20_mse_Pareto_Archives"),  # needed for baseline normalization
}

REF_PQ_CSV = os.path.join(ROOT, "20_mse_Pareto_Archives", "pq_baseline", "pareto_global.csv")

OUT_DIR = os.path.join(ROOT, "comparative_metrics")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

BASELINE_METHOD = "baseline_pq20"
BASELINE_K = 20


# -------------------------
# IO
# -------------------------

def load_fronts_from_dir(budget_dir: str) -> Dict[str, List[Point2D]]:
    paths = discover_csv_paths(budget_dir)
    fronts: Dict[str, List[Point2D]] = {}
    for method, csv_path in paths.items():
        rows = read_front_csv(csv_path)
        fronts[method] = rows_to_min_points(rows, path_hint=csv_path)  # f2 = -PSNR_N
    return fronts


# -------------------------
# Plotting
# -------------------------

def plot_curves(series: Dict[str, Dict[int, float]], ks: List[int], y_label: str, title: str, out_path: str) -> None:
    plt.figure()
    for method, vals_by_k in sorted(series.items()):
        ys = [vals_by_k.get(k, float("nan")) for k in ks]
        plt.plot(ks, ys, marker="o", label=method)

    plt.xlabel("K (archive size)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------------
# HV reference point (robust)
# -------------------------

def compute_hv_ref_point(
    reference: List[Point2D],
    fronts_by_k: Dict[int, Dict[str, List[Point2D]]],
    *,
    margin_frac: float = 0.05,
    min_abs_margin: float = 1e-6,
) -> Tuple[float, float]:
    """
    Compute a robust 2D HV reference point in minimization space.

    ref = (max_f1 + margin, max_f2 + margin)
    margin = max(margin_frac * range, min_abs_margin)

    Ensures ref is (very likely) dominated by all points.
    """
    pooled: List[Point2D] = []
    pooled.extend(reference)
    for fronts_by_method in fronts_by_k.values():
        for pts in fronts_by_method.values():
            pooled.extend(pts)

    if not pooled:
        raise ValueError("compute_hv_ref_point: no points available")

    min_f1, max_f1, min_f2, max_f2 = summarize_points(pooled)
    range_f1 = max(0.0, max_f1 - min_f1)
    range_f2 = max(0.0, max_f2 - min_f2)

    margin_f1 = max(margin_frac * range_f1, min_abs_margin)
    margin_f2 = max(margin_frac * range_f2, min_abs_margin)

    return (max_f1 + margin_f1, max_f2 + margin_f2)


def assert_ref_dominates_all(ref: Tuple[float, float], points: List[Point2D], *, label: str) -> None:
    """
    In minimization, ref must be >= every point componentwise.
    """
    if not points:
        return
    r1, r2 = ref
    offenders = [p for p in points if (p.f1 > r1 or p.f2 > r2)]
    if offenders:
        sample = offenders[:5]
        raise ValueError(
            f"HV ref point {ref} is not dominated by all points in {label}. "
            f"Example offenders: {sample} (showing {len(sample)}/{len(offenders)})"
        )


# -------------------------
# HV normalization (baseline = 1)
# -------------------------

def find_metric_value(rows: List[dict], *, method_name: str, metric_key: str) -> float:
    for r in rows:
        if r.get("method") == method_name:
            return float(r.get(metric_key))
    raise ValueError(f"Metric '{metric_key}' for method '{method_name}' not found.")


def add_hv_normalized(
    metrics_by_k: Dict[int, List[dict]],
    *,
    baseline_method: str = BASELINE_METHOD,
    baseline_k: int = BASELINE_K,
    hv_key: str = "hv_2d",
    out_key: str = "hv_2d_norm",
) -> None:
    if baseline_k not in metrics_by_k:
        raise ValueError(
            f"Baseline K={baseline_k} not present in metrics_by_k. "
            f"Include K=20 evaluation to normalize HV to baseline=1."
        )

    hv_base = find_metric_value(metrics_by_k[baseline_k], method_name=baseline_method, metric_key=hv_key)
    if not math.isfinite(hv_base) or hv_base <= 0.0:
        raise ValueError(f"Invalid baseline HV ({hv_base}) for '{baseline_method}' at K={baseline_k}.")

    for K, rows in metrics_by_k.items():
        for r in rows:
            hv = float(r.get(hv_key, float("nan")))
            r[out_key] = (hv / hv_base) if math.isfinite(hv) else float("nan")


# -------------------------
# Main
# -------------------------

def main() -> None:
    # 1) Reference set (PQ=20)
    ref_rows = read_front_csv(REF_PQ_CSV)
    reference = rows_to_min_points(ref_rows, path_hint=REF_PQ_CSV)

    # 2) Load all fronts first (needed to derive robust HV ref point)
    fronts_by_k: Dict[int, Dict[str, List[Point2D]]] = {}
    for K in sorted(BUDGET_DIRS.keys()):
        budget_dir = BUDGET_DIRS[K]
        if not os.path.isdir(budget_dir):
            print(f"[WARN] Missing budget directory (skipping K={K}): {budget_dir}")
            continue
        fronts_by_k[K] = load_fronts_from_dir(budget_dir)

    if not fronts_by_k:
        raise RuntimeError("No budget directories found. Nothing to evaluate.")

    if BASELINE_K not in fronts_by_k:
        raise RuntimeError(
            f"Baseline K={BASELINE_K} directory is missing. "
            f"Needed to set hv_2d_norm baseline=1. Check BUDGET_DIRS."
        )

    # 3) Compute robust HV reference point (minimization space)
    hv_ref_point = compute_hv_ref_point(reference, fronts_by_k, margin_frac=0.05)

    # Validate dominance against pooled points
    pooled: List[Point2D] = []
    pooled.extend(reference)
    for fronts_by_method in fronts_by_k.values():
        for pts in fronts_by_method.values():
            pooled.extend(pts)
    assert_ref_dominates_all(hv_ref_point, pooled, label="pooled(reference + all fronts)")
    print(f"[INFO] Using HV reference point (min space): {hv_ref_point}")

    # 4) Evaluate per K (include baseline method in every K)
    metrics_by_k: Dict[int, List[dict]] = {}
    for K in sorted(fronts_by_k.keys()):
        fronts_by_method = fronts_by_k[K]
        metrics = evaluate_methods(
            reference=reference,
            fronts_by_method=fronts_by_method,
            hv_ref_point=hv_ref_point,
            include_baseline=True,
            baseline_name=BASELINE_METHOD,
        )
        metrics_by_k[K] = metrics

    # 5) Add HV normalization (baseline@K=20 = 1.0)
    add_hv_normalized(metrics_by_k, baseline_method=BASELINE_METHOD, baseline_k=BASELINE_K)

    # 6) Write per-K CSVs (after normalization so column exists everywhere)
    for K in sorted(metrics_by_k.keys()):
        out_csv = os.path.join(OUT_DIR, f"metrics_k{K}.csv")
        write_metrics_csv(out_csv, metrics_by_k[K])
        print(f"[DONE] Metrics for K={K} written to {out_csv}")

    # 7) Plots
    ks_eval = sorted(metrics_by_k.keys())

    igd_series = pivot_metric_over_k(metrics_by_k, "igd_plus")
    hv_norm_series = pivot_metric_over_k(metrics_by_k, "hv_2d_norm")

    plot_curves(
        series=igd_series,
        ks=ks_eval,
        y_label="IGD+ (lower is better)",
        title="IGD+ vs K (Reference: PQ=20, best-known)",
        out_path=os.path.join(PLOTS_DIR, "igd_plus_vs_k.png"),
    )

    plot_curves(
        series=hv_norm_series,
        ks=ks_eval,
        y_label="Hypervolume (normalized; baseline@K=20 = 1.0)",
        title="Normalized HV vs K (baseline=1.0)",
        out_path=os.path.join(PLOTS_DIR, "hv_norm_vs_k.png"),
    )

    print(f"[DONE] Plots written under: {PLOTS_DIR}")


if __name__ == "__main__":
    main()

