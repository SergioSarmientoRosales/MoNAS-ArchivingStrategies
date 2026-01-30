# archivers/igd_ideal.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import math

from archivers.records import NormRecord



def pareto_dominates_min(a: Sequence[float], b: Sequence[float]) -> bool:
    """Standard Pareto dominance for minimization: a ≺ b."""
    return all(ai <= bi for ai, bi in zip(a, b)) and any(ai < bi for ai, bi in zip(a, b))


def euclidean(u: Sequence[float], v: Sequence[float]) -> float:
    return math.sqrt(sum((ui - vi) ** 2 for ui, vi in zip(u, v)))


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


def igd(reference: List[Tuple[float, float]], archive_objs: List[Tuple[float, float]]) -> float:
    """
    IGD(R, A) = (1/|R|) * sum_{r in R} min_{a in A} ||r - a||
    """
    if not reference:
        raise ValueError("Reference set is empty.")
    if not archive_objs:
        return float("inf")

    total = 0.0
    for r in reference:
        best = float("inf")
        for a in archive_objs:
            d = euclidean(r, a)
            if d < best:
                best = d
        total += best
    return total / float(len(reference))


def sample_segment(a: Tuple[float, float], b: Tuple[float, float], n: int) -> List[Tuple[float, float]]:
    """Uniform samples on the segment from a to b (inclusive)."""
    if n <= 1:
        return [a]
    out: List[Tuple[float, float]] = []
    for i in range(n):
        t = i / float(n - 1)
        out.append((a[0] * (1.0 - t) + b[0] * t, a[1] * (1.0 - t) + b[1] * t))
    return out


class IGDIdealArchiver:
    """
    IGD-based bounded Pareto archiver using a synthetic "ideal Pareto" reference.

    Reference construction (2 objectives):
      - Compute non-dominated pool ND from all available points.
      - Find endpoint e1 with minimum f1 (best PSNR).
      - Find endpoint e2 with minimum f2 (best Params).
      - Build reference set R as n_ref samples on segment [e1, e2].

    Then:
      - Maintain a Pareto archive (non-dominated).
      - If size > max_size: greedily remove points to minimize IGD(R, A).

    Objective space (minimization, normalized):
      f(p) = (PSNR_N, Params_N) in [0,1]^2
    """
    name = "igd_ideal"

    def __init__(self, max_size: int = 20, n_ref: int = 100):
        if max_size <= 0:
            raise ValueError("max_size must be positive.")
        if n_ref <= 2:
            raise ValueError("n_ref must be > 2.")
        self.max_size = int(max_size)
        self.n_ref = int(n_ref)
        self._archive: List[NormRecord] = []
        self._reference: Optional[List[Tuple[float, float]]] = None

    def reset(self) -> None:
        self._archive = []
        self._reference = None

    @staticmethod
    def _obj_min(nr: NormRecord) -> Tuple[float, float]:
        return (1.0 - nr.psnr_n, nr.params_n)

    def _build_reference_from_candidates(self, all_points: List[NormRecord]) -> List[Tuple[float, float]]:
        nd = pareto_filter(all_points, self._obj_min)
        nd_objs = [self._obj_min(x) for x in nd]

        # endpoints in ND set
        e1 = min(nd_objs, key=lambda z: z[0])  # best f1
        e2 = min(nd_objs, key=lambda z: z[1])  # best f2

        # if both objectives minimized by same point, reference collapses -> use small neighborhood
        if abs(e1[0] - e2[0]) < 1e-12 and abs(e1[1] - e2[1]) < 1e-12:
            return [e1]

        return sample_segment(e1, e2, self.n_ref)

    def update(self, candidates: List[NormRecord]) -> None:
        # Build reference once, using the first "global" batch that arrives.
        # In your brain, you pass nrecs (all points), so this is ideal.
        if self._reference is None:
            self._reference = self._build_reference_from_candidates(candidates)

        # Merge + Pareto filter
        merged = list(self._archive) + list(candidates)
        nd = pareto_filter(merged, self._obj_min)

        # If small enough, done
        if len(nd) <= self.max_size:
            self._archive = nd
            return

        # Greedy IGD pruning
        current = nd
        while len(current) > self.max_size:
            objs = [self._obj_min(x) for x in current]

            best_val = float("inf")
            best_remove = -1

            for i in range(len(current)):
                reduced = objs[:i] + objs[i + 1 :]
                val = igd(self._reference, reduced)  # type: ignore[arg-type]
                if val < best_val:
                    best_val = val
                    best_remove = i

            current.pop(best_remove)

        self._archive = current

    def front(self) -> List[NormRecord]:
        return list(self._archive)

    def state(self) -> Dict[str, Any]:
        return {
            "size": len(self._archive),
            "max_size": self.max_size,
            "n_ref": self.n_ref,
            "ref_size": 0 if self._reference is None else len(self._reference),
        }
