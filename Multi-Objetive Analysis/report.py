from __future__ import annotations

import csv
import math
import os
from typing import Dict, List, Tuple, Optional, Any

from io_fronts import Point2D
from indicators import igd_plus, coverage_stats, nondominated
from hv2d import hypervolume_2d_min

# Modular indicators
from r2_indicator import r2_indicator, generate_weight_vectors
from epsilon_indicator import epsilon_additive
from coverage_metric import coverage_metric
from hausdorff_metric import hausdorff_distance


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

def _is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _safe_metric(fn, *, on_empty: float = float("nan"), name: str = "") -> float:
    """
    Execute a metric function safely. Returns NaN on exceptions, with a warning.
    """
    try:
        val = fn()
        return float(val)
    except Exception as e:
        tag = f" {name}" if name else ""
        print(f"[WARN]{tag} metric failed: {e}")
        return float(on_empty)


def _assert_hv_ref_dominated_by_set(
    hv_ref_point: Tuple[float, float],
    pts: List[Point2D],
    *,
    label: str,
) -> None:
    """
    In minimization, hypervolume ref point must be component-wise >= every point.
    """
    if not pts:
        return

    r1, r2 = hv_ref_point
    offenders = [p for p in pts if (p.f1 > r1 or p.f2 > r2)]
    if offenders:
        sample = offenders[:5]
        raise ValueError(
            f"HV ref point {hv_ref_point} is not dominated by all points in '{label}'. "
            f"Example offenders: {sample} (showing {len(sample)}/{len(offenders)})"
        )


def _normalize_method_name(name: str) -> str:
    return name.strip()


# -------------------------------------------------------------------------
# CSV Writer
# -------------------------------------------------------------------------

def write_metrics_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write metrics to CSV file, creating parent directories if needed."""
    if not rows:
        return

    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # Stable header: union of keys in insertion order (first row defines baseline)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# -------------------------------------------------------------------------
# Main evaluation
# -------------------------------------------------------------------------

def evaluate_methods(
    reference: List[Point2D],
    fronts_by_method: Dict[str, List[Point2D]],
    hv_ref_point: Tuple[float, float],
    *,
    include_baseline: bool = True,
    baseline_name: str = "baseline_pq20",
    r2_n_weights: int = 50,
    r2_seed: int = 123,
) -> List[Dict[str, Any]]:
    """
    Evaluate multiple methods against a reference Pareto front (best-known set).

    Assumes MINIMIZATION space:
        f1 = Params_N (min)
        f2 = -PSNR_N  (min)  <=> maximize PSNR_N

    Computes:
      - n_points, n_nd
      - IGD+ (reference vs ND(front))
      - HV 2D on ND(front)
      - coverage_stats(reference, ND(front))  (your original metric block)
      - R2 (via weight vectors)
      - epsilon additive
      - coverage metric (C-metric)
      - Hausdorff distance
    """

    # Defensive copy + optional baseline injection
    methods: Dict[str, List[Point2D]] = {}
    for m, pts in fronts_by_method.items():
        methods[_normalize_method_name(m)] = list(pts)

    if include_baseline:
        # Include baseline as a pseudo-method so it appears in every metrics_k*.csv
        methods[_normalize_method_name(baseline_name)] = list(reference)

    # Pre-generate weight vectors for R2 (reproducible)
    # NOTE: This assumes generate_weight_vectors supports reproducible behavior with a seed
    # If it doesn't, we will adjust generate_weight_vectors in r2_indicator.py next.
    weights = generate_weight_vectors(r2_n_weights)  # keep signature to avoid breaking
    # If your implementation supports seeding, you should incorporate r2_seed there.

    results: List[Dict[str, Any]] = []

    for method, front in methods.items():
        nd = nondominated(front)

        # Validate HV ref dominance for this front (ND points only)
        # If this fails, HV is not meaningful for this method/K.
        _assert_hv_ref_dominated_by_set(hv_ref_point, nd, label=f"{method}:ND")

        row: Dict[str, Any] = {
            "method": method,
            "n_points": len(front),
            "n_nd": len(nd),
        }

        # Existing metrics (safe wrappers)
        row["igd_plus"] = _safe_metric(
            lambda: igd_plus(reference, nd),
            name=f"IGD+/{method}",
        )
        row["hv_2d"] = _safe_metric(
            lambda: hypervolume_2d_min(nd, ref=hv_ref_point),
            name=f"HV/{method}",
        )

        # Coverage stats block (safe)
        cov = {}
        try:
            cov = coverage_stats(reference, nd)
        except Exception as e:
            print(f"[WARN] coverage_stats/{method} failed: {e}")
            cov = {}
        row.update(cov)

        # New metrics (safe)
        row["r2"] = _safe_metric(
            lambda: r2_indicator(reference, nd, weights),
            name=f"R2/{method}",
        )
        row["epsilon"] = _safe_metric(
            lambda: epsilon_additive(reference, nd),
            name=f"EpsAdd/{method}",
        )
        row["coverage_c"] = _safe_metric(
            lambda: coverage_metric(nd, reference),
            name=f"CoverageC/{method}",
        )
        row["hausdorff"] = _safe_metric(
            lambda: hausdorff_distance(nd, reference),
            name=f"Hausdorff/{method}",
        )

        results.append(row)

    # Sort by IGD+ (ascending) then HV (descending) as a stable default
    def _sort_key(r: Dict[str, Any]) -> Tuple[float, float, str]:
        igd = r.get("igd_plus", float("nan"))
        hv = r.get("hv_2d", float("nan"))
        igd_key = igd if _is_finite(igd) else float("inf")
        hv_key = hv if _is_finite(hv) else float("-inf")
        return (igd_key, -hv_key, str(r.get("method", "")))

    results.sort(key=_sort_key)
    return results


# -------------------------------------------------------------------------
# Verbose evaluation
# -------------------------------------------------------------------------

def evaluate_methods_verbose(
    reference: List[Point2D],
    fronts_by_method: Dict[str, List[Point2D]],
    hv_ref_point: Tuple[float, float],
    *,
    include_baseline: bool = True,
    baseline_name: str = "baseline_pq20",
    r2_n_weights: int = 50,
    r2_seed: int = 123,
    print_results: bool = True,
) -> List[Dict[str, Any]]:

    results = evaluate_methods(
        reference=reference,
        fronts_by_method=fronts_by_method,
        hv_ref_point=hv_ref_point,
        include_baseline=include_baseline,
        baseline_name=baseline_name,
        r2_n_weights=r2_n_weights,
        r2_seed=r2_seed,
    )

    if print_results:
        print("\n" + "=" * 110)
        print("MULTI-OBJECTIVE EVALUATION RESULTS")
        print("=" * 110)
        print(f"Reference front size: {len(reference)} points")
        print(f"HV reference point: {hv_ref_point}")
        print(f"Baseline included: {include_baseline} ({baseline_name})")
        print("-" * 110)

        header = (
            f"{'Method':<22} {'Pts':>6} {'ND':>6} "
            f"{'IGD+':>12} {'HV':>12} "
            f"{'R2':>12} {'Eps':>12} {'CovC':>12} {'Haus':>12}"
        )
        print(header)
        print("-" * 110)

        for row in results:
            def _fmt(x: Any) -> str:
                try:
                    xf = float(x)
                    return f"{xf:>12.6f}" if math.isfinite(xf) else f"{'nan':>12}"
                except Exception:
                    return f"{'nan':>12}"

            print(
                f"{str(row.get('method','')):<22} "
                f"{int(row.get('n_points',0)):>6d} "
                f"{int(row.get('n_nd',0)):>6d} "
                f"{_fmt(row.get('igd_plus'))} "
                f"{_fmt(row.get('hv_2d'))} "
                f"{_fmt(row.get('r2'))} "
                f"{_fmt(row.get('epsilon'))} "
                f"{_fmt(row.get('coverage_c'))} "
                f"{_fmt(row.get('hausdorff'))}"
            )

        print("=" * 110 + "\n")

    return results


# -------------------------------------------------------------------------
# Pivot for plotting
# -------------------------------------------------------------------------

def pivot_metric_over_k(
    metrics_by_k: Dict[int, List[Dict[str, Any]]],
    metric_key: str,
) -> Dict[str, Dict[int, float]]:
    """
    Transform metrics data for plotting metric evolution over parameter K.
    """
    series: Dict[str, Dict[int, float]] = {}
    for k, rows in metrics_by_k.items():
        for row in rows:
            method = str(row.get("method", ""))
            if method not in series:
                series[method] = {}
            try:
                series[method][k] = float(row.get(metric_key))
            except Exception:
                series[method][k] = float("nan")
    return series


# -------------------------------------------------------------------------
# Best method selector
# -------------------------------------------------------------------------

def get_best_method(
    results: List[Dict[str, Any]],
    metric: str = "igd_plus",
    minimize: bool = True,
) -> Optional[str]:
    if not results:
        return None

    valid = []
    for r in results:
        v = r.get(metric, float("nan"))
        try:
            vf = float(v)
            if math.isfinite(vf):
                valid.append((vf, r))
        except Exception:
            continue

    if not valid:
        return None

    if minimize:
        best = min(valid, key=lambda t: t[0])[1]
    else:
        best = max(valid, key=lambda t: t[0])[1]
    return str(best.get("method"))


# -------------------------------------------------------------------------
# Ranking (kept, with more robust normalization)
# -------------------------------------------------------------------------

def compute_ranking(
    results: List[Dict[str, Any]],
    metrics: List[str] = ["igd_plus", "hv_2d"],
    weights: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:

    if not results:
        return []

    if weights is None:
        weights = [1.0] * len(metrics)
    if len(weights) != len(metrics):
        raise ValueError("Number of weights must match number of metrics")

    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Pre-collect finite values
    metric_values: Dict[str, List[float]] = {m: [] for m in metrics}
    for r in results:
        for m in metrics:
            try:
                v = float(r.get(m))
                if math.isfinite(v):
                    metric_values[m].append(v)
            except Exception:
                pass

    bounds: Dict[str, Tuple[float, float]] = {}
    for m, vals in metric_values.items():
        if vals:
            bounds[m] = (min(vals), max(vals))
        else:
            bounds[m] = (0.0, 0.0)

    results_with_scores: List[Dict[str, Any]] = []
    for result in results:
        score = 0.0
        for m, w in zip(metrics, weights):
            v = result.get(m, float("nan"))
            try:
                vf = float(v)
            except Exception:
                vf = float("nan")

            mn, mx = bounds[m]
            if not math.isfinite(vf):
                normalized = 0.0
            elif mx > mn:
                if m == "igd_plus":
                    # lower is better
                    normalized = 1.0 - (vf - mn) / (mx - mn)
                else:
                    # higher is better
                    normalized = (vf - mn) / (mx - mn)
            else:
                normalized = 1.0

            score += w * normalized

        result_copy = result.copy()
        result_copy["score"] = score
        results_with_scores.append(result_copy)

    results_with_scores.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    for rank, result in enumerate(results_with_scores, start=1):
        result["rank"] = rank

    return results_with_scores
