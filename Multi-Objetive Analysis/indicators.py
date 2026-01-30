from __future__ import annotations

from typing import List, Dict
import math

from io_fronts import Point2D


def dominates(a: Point2D, b: Point2D) -> bool:
    """
    True if 'a' dominates 'b' in MINIMIZATION:
      a_i <= b_i for all i and a_j < b_j for at least one j.
    """
    return (a.f1 <= b.f1 and a.f2 <= b.f2) and (a.f1 < b.f1 or a.f2 < b.f2)


def nondominated(points: List[Point2D]) -> List[Point2D]:
    """
    Return the non-dominated subset (MINIMIZATION).
    Complexity: O(n^2). Acceptable for small fronts (K<=20).
    """
    if not points:
        return []

    nd: List[Point2D] = []
    for i, p in enumerate(points):
        is_dominated = False
        for j, q in enumerate(points):
            if j != i and dominates(q, p):
                is_dominated = True
                break
        if not is_dominated:
            nd.append(p)
    return nd


def euclidean(p: Point2D, q: Point2D) -> float:
    """Euclidean distance in objective space."""
    return math.hypot(p.f1 - q.f1, p.f2 - q.f2)


def igd_plus(reference: List[Point2D], approx: List[Point2D]) -> float:
    """
    IGD+ (Inverted Generational Distance Plus) from reference -> approx (MINIMIZATION).

    For each r in reference:
      d^+(r, A) = min_{a in A} sqrt( sum_i max(0, a_i - r_i)^2 )
    IGD+ = average of d^+(r, A) over r in reference.

    If approx is empty, returns +inf (standard practical behavior).
    """
    if not reference:
        raise ValueError("IGD+: reference set is empty.")
    if not approx:
        return float("inf")

    def d_plus(r: Point2D, a: Point2D) -> float:
        d1 = max(0.0, a.f1 - r.f1)
        d2 = max(0.0, a.f2 - r.f2)
        return math.hypot(d1, d2)

    total = 0.0
    for r in reference:
        best = min(d_plus(r, a) for a in approx)
        total += best
    return total / float(len(reference))


def gd_plus(reference: List[Point2D], approx: List[Point2D]) -> float:
    """
    GD+ (Generational Distance Plus) from approx -> reference (MINIMIZATION).

    For each a in approx:
      d^+(a, R) = min_{r in R} sqrt( sum_i max(0, a_i - r_i)^2 )
    GD+ = average over a in approx.
    """
    if not reference:
        raise ValueError("GD+: reference set is empty.")
    if not approx:
        raise ValueError("GD+: approximation set is empty.")

    def d_plus(a: Point2D, r: Point2D) -> float:
        d1 = max(0.0, a.f1 - r.f1)
        d2 = max(0.0, a.f2 - r.f2)
        return math.hypot(d1, d2)

    total = 0.0
    for a in approx:
        best = min(d_plus(a, r) for r in reference)
        total += best
    return total / float(len(approx))


def coverage_stats(reference: List[Point2D], approx: List[Point2D]) -> Dict[str, float]:
    """
    Range-based coverage statistics relative to reference.

    Returns:
      - min/max in each objective for approx
      - coverage ratios wrt reference ranges:
          cov_f1 = (range of approx in f1) / (range of ref in f1), clipped [0,1]
          cov_f2 = (range of approx in f2) / (range of ref in f2), clipped [0,1]

    Note:
      This is NOT Pareto C-metric; it is range coverage.
    """
    if not reference or not approx:
        return {
            "min_f1": float("nan"), "max_f1": float("nan"),
            "min_f2": float("nan"), "max_f2": float("nan"),
            "cov_f1": float("nan"), "cov_f2": float("nan"),
        }

    r_min1 = min(p.f1 for p in reference)
    r_max1 = max(p.f1 for p in reference)
    r_min2 = min(p.f2 for p in reference)
    r_max2 = max(p.f2 for p in reference)

    a_min1 = min(p.f1 for p in approx)
    a_max1 = max(p.f1 for p in approx)
    a_min2 = min(p.f2 for p in approx)
    a_max2 = max(p.f2 for p in approx)

    def ratio(min_r: float, max_r: float, min_a: float, max_a: float) -> float:
        denom = max_r - min_r
        if denom <= 0.0:
            return 0.0
        val = (max_a - min_a) / denom
        return max(0.0, min(1.0, val))

    return {
        "min_f1": a_min1, "max_f1": a_max1,
        "min_f2": a_min2, "max_f2": a_max2,
        "cov_f1": ratio(r_min1, r_max1, a_min1, a_max1),
        "cov_f2": ratio(r_min2, r_max2, a_min2, a_max2),
    }


def spread(points: List[Point2D]) -> float:
    """
    Simple spread/uniformity proxy for a set of points (lower is better; 0 is ideal).

    This is an internal diversity proxy based on distances between consecutive ND solutions
    sorted by f1. It is NOT guaranteed to match the canonical NSGA-II spread if you need
    boundary distances relative to the true PF.
    """
    if len(points) < 2:
        return 0.0

    nd = nondominated(points)
    if len(nd) < 2:
        return 0.0

    nd_sorted = sorted(nd, key=lambda p: (p.f1, p.f2))
    distances = [euclidean(nd_sorted[i], nd_sorted[i + 1]) for i in range(len(nd_sorted) - 1)]
    if not distances:
        return 0.0

    d_mean = sum(distances) / len(distances)
    d_first = distances[0]
    d_last = distances[-1]

    denom = d_first + d_last + len(distances) * d_mean
    if denom <= 0.0:
        return 0.0

    num = d_first + d_last + sum(abs(d - d_mean) for d in distances)
    return num / denom
