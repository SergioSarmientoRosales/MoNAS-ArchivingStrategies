from __future__ import annotations

from typing import List, Tuple
from io_fronts import Point2D
from indicators import nondominated


def hypervolume_2d_min(points: List[Point2D], ref: Tuple[float, float], *, validate_ref: bool = True) -> float:
    """
    Hypervolume in 2D for MINIMIZATION.
    Computes the dominated area bounded by 'ref' (must be worse: larger in both objectives).

    Algorithm:
      1) Filter ND points
      2) Sort by f1 ascending (minimization)
      3) Accumulate disjoint rectangles using a running upper bound on f2 (prev_y)

    Args:
        points: Set of points in minimization space.
        ref: Reference point (rx, ry) that must dominate all ND points (rx > max f1, ry > max f2).
        validate_ref: If True, validates that ref dominates ND(points).

    Returns:
        Hypervolume (float). Returns 0.0 for empty input.
    """
    if not points:
        return 0.0

    rx, ry = ref
    nd = nondominated(points)
    if not nd:
        return 0.0

    if validate_ref:
        max_f1 = max(p.f1 for p in nd)
        max_f2 = max(p.f2 for p in nd)
        if not (rx > max_f1 and ry > max_f2):
            raise ValueError(
                f"Invalid HV reference point {ref}. Must satisfy rx > max_f1 and ry > max_f2 "
                f"over ND(points). Got max_f1={max_f1}, max_f2={max_f2}."
            )

    # Sort by f1 ascending; f2 tie-breaker for stability
    nd_sorted = sorted(nd, key=lambda p: (p.f1, p.f2))

    hv = 0.0
    prev_y = ry

    for p in nd_sorted:
        width = rx - p.f1
        if width <= 0.0:
            continue

        height = prev_y - p.f2
        if height > 0.0:
            hv += width * height

        # Maintain monotone envelope to avoid double counting
        if p.f2 < prev_y:
            prev_y = p.f2

    return hv
