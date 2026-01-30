from __future__ import annotations
from typing import List
from io_fronts import Point2D

def dominates(a: Point2D, b: Point2D) -> bool:
    return (a.f1 <= b.f1 and a.f2 <= b.f2) and (a.f1 < b.f1 or a.f2 < b.f2)

def coverage_metric(A: List[Point2D], B: List[Point2D]) -> float:
    if not B:
        return float("nan")
    if not A:
        return 0.0
    dominated = sum(1 for b in B if any(dominates(a, b) for a in A))
    return dominated / len(B)
