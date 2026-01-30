# archivers/r2.py
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, Set
from archivers.records import NormRecord



def pareto_dominates_min(a: Sequence[float], b: Sequence[float]) -> bool:
    """Standard Pareto dominance for minimization: a ≺ b."""
    return all(ai <= bi for ai, bi in zip(a, b)) and any(ai < bi for ai, bi in zip(a, b))


def pareto_filter(points: List[NormRecord], obj_fn) -> List[NormRecord]:
    """Pairwise non-dominated subset."""
    nd: List[NormRecord] = []
    for p in points:
        p_obj = obj_fn(p)
        if any(pareto_dominates_min(obj_fn(a), p_obj) for a in nd):
            continue
        nd = [a for a in nd if not pareto_dominates_min(p_obj, obj_fn(a))]
        nd.append(p)
    return nd


def generate_simplex_weights_2d(n: int) -> List[Tuple[float, float]]:
    """
    Generate n weights on w1+w2=1, w_i>=0.
    Includes endpoints (1,0) and (0,1).
    """
    if n <= 1:
        return [(0.5, 0.5)]
    weights = []
    for i in range(n):
        w1 = i / float(n - 1)
        w2 = 1.0 - w1
        weights.append((w1, w2))
    return weights


def tchebycheff_scalar(
    f: Tuple[float, float],
    w: Tuple[float, float],
    z: Tuple[float, float],
) -> float:
    """
    Weighted Tchebycheff scalarization (minimization):
      g_w(f) = max_i w_i * |f_i - z_i|
    """
    return max(w[0] * abs(f[0] - z[0]), w[1] * abs(f[1] - z[1]))


class R2Archiver:
    """
    R2-based archiver (not from the book).

    Steps:
      1) Merge archive + candidates
      2) Pareto filter (optional but recommended to stabilize)
      3) Compute ideal point z (componentwise min)
      4) For each weight vector w, pick p minimizing g_w(p)
      5) Union the selections, Pareto-filter again (safety)

    Objective space:
      f(p) = ( PSNR_N, Params_N) in [0,1]^2 (minimization)
    """
    name = "r2"

    def __init__(self, n_weights: int = 21):
        if n_weights <= 0:
            raise ValueError("n_weights must be positive.")
        self.n_weights = int(n_weights)
        self.weights = generate_simplex_weights_2d(self.n_weights)
        self._archive: List[NormRecord] = []

    def reset(self) -> None:
        self._archive = []

    @staticmethod
    def _obj_min(nr: NormRecord) -> Tuple[float, float]:
        return (1.0 - nr.psnr_n, nr.params_n)

    def update(self, candidates: List[NormRecord]) -> None:
        # 1) Merge
        merged = list(self._archive) + list(candidates)

        # 2) Pareto filter first (recommended)
        nd_pool = pareto_filter(merged, self._obj_min)
        if not nd_pool:
            self._archive = []
            return

        # 3) Ideal point z (componentwise min)
        objs = [self._obj_min(x) for x in nd_pool]
        z = (min(o[0] for o in objs), min(o[1] for o in objs))

        # 4) Select one best solution per weight vector
        selected_indices: Set[int] = set()

        for w in self.weights:
            best_i = 0
            best_val = float("inf")
            for i, f in enumerate(objs):
                val = tchebycheff_scalar(f, w, z)
                if val < best_val:
                    best_val = val
                    best_i = i
            selected_indices.add(best_i)

        selected = [nd_pool[i] for i in selected_indices]

        # 5) Final Pareto safety pass
        self._archive = pareto_filter(selected, self._obj_min)

    def front(self) -> List[NormRecord]:
        return list(self._archive)

    def state(self) -> Dict[str, Any]:
        return {
            "size": len(self._archive),
            "n_weights": self.n_weights,
        }
