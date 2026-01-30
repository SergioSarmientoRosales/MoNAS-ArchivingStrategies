# archivers/entropy.py
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import math

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


def shannon_entropy_from_counts(counts: Dict[Tuple[int, int], int]) -> float:
    """Shannon entropy of a discrete distribution defined by counts."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / float(total)
        if p > 0.0:
            h -= p * math.log(p)
    return h


class EntropyArchiver:
    """
    Entropy-based bounded Pareto archiver.

    Diversity objective:
      Maximize Shannon entropy of occupancy over a 2D grid in objective space.

    Pipeline:
      1) Merge archive + candidates
      2) Pareto-filter -> ND pool
      3) If |ND| <= max_size: keep ND
      4) Else: greedy selection that maximizes entropy increase:
           - start with 2 extremes (best f1 and best f2)
           - iteratively add the point that yields highest entropy of selected set
      5) Final Pareto safety filter

    Objective space (minimization, normalized):
      f(p) = (1 - PSNR_N, Params_N) in [0,1]^2
    """
    name = "entropy"

    def __init__(self, max_size: int = 20, bins_x: int = 50, bins_y: int = 50):
        if max_size <= 0:
            raise ValueError("max_size must be positive.")
        if bins_x <= 0 or bins_y <= 0:
            raise ValueError("bins_x and bins_y must be positive.")
        self.max_size = int(max_size)
        self.bins_x = int(bins_x)
        self.bins_y = int(bins_y)
        self._archive: List[NormRecord] = []

    def reset(self) -> None:
        self._archive = []

    @staticmethod
    def _obj_min(nr: NormRecord) -> Tuple[float, float]:
        return (1.0 - nr.psnr_n, nr.params_n)

    def _cell(self, f1: float, f2: float) -> Tuple[int, int]:
        # clamp to [0,1]
        f1 = 0.0 if f1 < 0.0 else (1.0 if f1 > 1.0 else f1)
        f2 = 0.0 if f2 < 0.0 else (1.0 if f2 > 1.0 else f2)
        ix = min(self.bins_x - 1, int(f1 * self.bins_x))
        iy = min(self.bins_y - 1, int(f2 * self.bins_y))
        return (ix, iy)

    def update(self, candidates: List[NormRecord]) -> None:
        merged = list(self._archive) + list(candidates)

        # 1) Pareto pool
        nd = pareto_filter(merged, self._obj_min)
        if len(nd) <= self.max_size:
            self._archive = nd
            return

        objs = [self._obj_min(x) for x in nd]

        # 2) Seed selection with extremes (helps diversity and stability)
        i_best_f1 = min(range(len(nd)), key=lambda i: objs[i][0])
        i_best_f2 = min(range(len(nd)), key=lambda i: objs[i][1])

        selected_idx: List[int] = []
        used = set()

        for i in (i_best_f1, i_best_f2):
            if i not in used:
                selected_idx.append(i)
                used.add(i)
            if len(selected_idx) >= self.max_size:
                break

        # If both extremes are same point, start with it only
        if not selected_idx:
            selected_idx = [0]
            used.add(0)

        # Current occupancy counts
        counts: Dict[Tuple[int, int], int] = {}
        for i in selected_idx:
            f1, f2 = objs[i]
            c = self._cell(f1, f2)
            counts[c] = counts.get(c, 0) + 1

        # 3) Greedy entropy maximization
        while len(selected_idx) < self.max_size:
            best_i = None
            best_h = -float("inf")

            base_counts = counts  # alias; we will trial-update in a copy

            for i in range(len(nd)):
                if i in used:
                    continue
                f1, f2 = objs[i]
                c = self._cell(f1, f2)

                trial = dict(base_counts)
                trial[c] = trial.get(c, 0) + 1
                h = shannon_entropy_from_counts(trial)

                if h > best_h:
                    best_h = h
                    best_i = i

            if best_i is None:
                break

            selected_idx.append(best_i)
            used.add(best_i)
            f1, f2 = objs[best_i]
            c = self._cell(f1, f2)
            counts[c] = counts.get(c, 0) + 1

        selected = [nd[i] for i in selected_idx]

        # 4) Final Pareto safety pass (usually redundant)
        self._archive = pareto_filter(selected, self._obj_min)

    def front(self) -> List[NormRecord]:
        return list(self._archive)

    def state(self) -> Dict[str, Any]:
        return {
            "size": len(self._archive),
            "max_size": self.max_size,
            "bins_x": self.bins_x,
            "bins_y": self.bins_y,
        }
