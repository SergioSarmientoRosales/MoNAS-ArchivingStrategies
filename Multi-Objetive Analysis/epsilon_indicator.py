from __future__ import annotations

from typing import List
from io_fronts import Point2D


def epsilon_additive(reference: List[Point2D], approx: List[Point2D]) -> float:
    """
    Additive epsilon indicator for MINIMIZATION (reference -> approx).
    Lower is better. 0 indicates that approx weakly dominates reference.

    Returns:
      +inf if approx is empty.
    Raises:
      ValueError if reference is empty.
    """
    if not reference:
        raise ValueError("epsilon_additive: reference set is empty.")
    if not approx:
        return float("inf")

    eps = float("-inf")
    for r in reference:
        eps_r = float("inf")
        for a in approx:
            # max over objectives of (a_i - r_i)
            eps_r = min(eps_r, max(a.f1 - r.f1, a.f2 - r.f2))
        eps = max(eps, eps_r)

    return float(eps)
