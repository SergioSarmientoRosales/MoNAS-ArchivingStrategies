from __future__ import annotations

from typing import List, Tuple
from io_fronts import Point2D


def generate_weight_vectors(n_vectors: int, *, eps: float = 1e-6) -> List[Tuple[float, float]]:
    """
    Generate evenly spaced 2D weight vectors with strictly positive components.

    We avoid 0 weights because they make the ASF ignore an objective.
    """
    if n_vectors <= 0:
        raise ValueError("n_vectors must be >= 1")

    if n_vectors == 1:
        return [(0.5, 0.5)]

    ws: List[Tuple[float, float]] = []
    # Use eps..(1-eps) to avoid 0 and 1 exactly.
    for i in range(n_vectors):
        t = i / (n_vectors - 1)
        w1 = eps + (1.0 - 2.0 * eps) * t
        w2 = 1.0 - w1
        # w2 will also be in [eps, 1-eps]
        ws.append((w1, w2))
    return ws


def r2_indicator(
    reference: List[Point2D],   # kept for API compatibility; not required in this variant
    approx: List[Point2D],
    weights: List[Tuple[float, float]],
    *,
    shift_f2_to_unit: bool = True,
) -> float:
    """
    R2 indicator (2D) for MINIMIZATION using a weighted Tchebycheff ASF.

    IMPORTANT for this project:
      Your pipeline uses Point2D with:
        f1 = Params_N          in [0,1] (minimize)
        f2 = -PSNR_N           in [-1,0] (minimize)

      A direct ASF max(w1*f1, w2*f2) is invalid because f2 is negative, which
      collapses the sensitivity to the second objective.

    Fix:
      Use g2 = 1 + f2, which equals (1 - PSNR_N) in [0,1].
      Then ASF uses (g1, g2) both non-negative.

    Returns:
      Lower is better. Returns +inf if approx is empty.
    """
    if not approx:
        return float("inf")
    if not weights:
        raise ValueError("weights must be non-empty")

    values: List[float] = []
    for w1, w2 in weights:
        # Defensive: ensure weights are positive
        if w1 <= 0.0 or w2 <= 0.0:
            raise ValueError(f"Invalid weight vector (w1,w2)=({w1},{w2}). Use strictly positive weights.")

        best = float("inf")
        for p in approx:
            g1 = p.f1
            if shift_f2_to_unit:
                # f2 = -PSNR_N  =>  1 + f2 = 1 - PSNR_N (in [0,1] if PSNR_N in [0,1])
                g2 = 1.0 + p.f2
            else:
                # Only safe if your f2 is already non-negative minimization (e.g., 1-PSNR_N)
                g2 = p.f2

            # Weighted Tchebycheff ASF
            val = max(w1 * g1, w2 * g2)
            if val < best:
                best = val

        values.append(best)

    return sum(values) / float(len(values))
