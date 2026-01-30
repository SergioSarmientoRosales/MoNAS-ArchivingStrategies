# archivers/hv.py
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
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


def hypervolume_2d(front: List[Tuple[float, float]],
                   ref: Tuple[float, float]) -> float:
    """
    Exact hypervolume in 2D (minimization).
    Assumes front is non-dominated.
    """
    if not front:
        return 0.0

    # sort by f1 ascending
    pts = sorted(front, key=lambda x: x[0])

    hv = 0.0
    prev_f2 = ref[1]

    for f1, f2 in pts:
        hv += max(0.0, ref[0] - f1) * max(0.0, prev_f2 - f2)
        prev_f2 = f2

    return hv


class HVArchiver:
    """
    Hypervolume-based archiver.

    - Maintains a Pareto archive
    - If size exceeds max_size, removes points with smallest HV contribution
    - Exact HV for 2 objectives
    """
    name = "hv"

    def __init__(self, max_size: int = 20, ref: Tuple[float, float] = (1.0, 1.0)):
        if max_size <= 0:
            raise ValueError("max_size must be positive.")
        self.max_size = int(max_size)
        self.ref = (float(ref[0]), float(ref[1]))
        self._archive: List[NormRecord] = []

    def reset(self) -> None:
        self._archive = []

    @staticmethod
    def _obj_min(nr: NormRecord) -> Tuple[float, float]:
        return (1.0 - nr.psnr_n, nr.params_n)

    def update(self, candidates: List[NormRecord]) -> None:
        # 1) Merge
        merged = list(self._archive) + list(candidates)

        # 2) Pareto filter
        nd = pareto_filter(merged, self._obj_min)

        # 3) If small enough, keep all
        if len(nd) <= self.max_size:
            self._archive = nd
            return

        # 4) HV contribution pruning
        current = nd

        while len(current) > self.max_size:
            objs = [self._obj_min(x) for x in current]
            total_hv = hypervolume_2d(objs, self.ref)

            worst_idx = -1
            worst_contrib = float("inf")

            for i in range(len(current)):
                reduced = objs[:i] + objs[i + 1 :]
                hv_reduced = hypervolume_2d(reduced, self.ref)
                contrib = total_hv - hv_reduced
                if contrib < worst_contrib:
                    worst_contrib = contrib
                    worst_idx = i

            current.pop(worst_idx)

        self._archive = current

    def front(self) -> List[NormRecord]:
        return list(self._archive)

    def state(self) -> Dict[str, Any]:
        return {
            "size": len(self._archive),
            "max_size": self.max_size,
            "ref": self.ref,
        }
