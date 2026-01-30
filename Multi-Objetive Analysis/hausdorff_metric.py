from __future__ import annotations

import math
from typing import List
from io_fronts import Point2D


def euclidean(p: Point2D, q: Point2D) -> float:
    return math.hypot(p.f1 - q.f1, p.f2 - q.f2)


def directed_hausdorff(A: List[Point2D], B: List[Point2D]) -> float:
    """
    Directed Hausdorff distance h(A,B):
      max_{a in A} min_{b in B} d(a,b)
    """
    if not A and not B:
        return float("nan")
    if not A or not B:
        return float("inf")

    return max(min(euclidean(a, b) for b in B) for a in A)


def hausdorff_distance(A: List[Point2D], B: List[Point2D]) -> float:
    """Symmetric Hausdorff distance H(A,B) = max(h(A,B), h(B,A))."""
    h_ab = directed_hausdorff(A, B)
    h_ba = directed_hausdorff(B, A)

    # If both are NaN (both empty), return NaN
    if math.isnan(h_ab) and math.isnan(h_ba):
        return float("nan")
    # max with NaN propagates poorly; handle explicitly
    if math.isnan(h_ab):
        return h_ba
    if math.isnan(h_ba):
        return h_ab
    return max(h_ab, h_ba)
