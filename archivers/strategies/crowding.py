# archivers/crowding.py
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
from archivers.records import NormRecord


def pareto_dominates_min(a: Sequence[float], b: Sequence[float]) -> bool:
    return all(ai <= bi for ai, bi in zip(a, b)) and any(ai < bi for ai, bi in zip(a, b))


def pareto_filter(points: List[NormRecord], obj_fn) -> List[NormRecord]:
    nd: List[NormRecord] = []
    for p in points:
        p_obj = obj_fn(p)
        if any(pareto_dominates_min(obj_fn(a), p_obj) for a in nd):
            continue
        nd = [a for a in nd if not pareto_dominates_min(p_obj, obj_fn(a))]
        nd.append(p)
    return nd


def crowding_distance(objs: List[Tuple[float, float]]) -> List[float]:
    """
    Crowding distance for 2 objectives (minimization).
    Larger = more isolated.
    """
    n = len(objs)
    if n == 0:
        return []
    if n <= 2:
        return [float("inf")] * n

    dist = [0.0] * n

    for m in range(2):  # each objective
        order = sorted(range(n), key=lambda i: objs[i][m])
        dist[order[0]] = float("inf")
        dist[order[-1]] = float("inf")

        min_val = objs[order[0]][m]
        max_val = objs[order[-1]][m]
        if max_val - min_val < 1e-12:
            continue

        for i in range(1, n - 1):
            prev_v = objs[order[i - 1]][m]
            next_v = objs[order[i + 1]][m]
            dist[order[i]] += (next_v - prev_v) / (max_val - min_val)

    return dist


class CrowdingArchiver:
    """
    Crowding-distance Pareto archiver (NSGA-II style).

    - Guarantees non-dominated archive
    - Maximizes local diversity on the Pareto front
    - No reference set, no epsilon, no clustering
    """
    name = "crowding"

    def __init__(self, max_size: int = 20):
        if max_size <= 0:
            raise ValueError("max_size must be positive.")
        self.max_size = int(max_size)
        self._archive: List[NormRecord] = []

    def reset(self) -> None:
        self._archive = []

    @staticmethod
    def _obj_min(nr: NormRecord) -> Tuple[float, float]:
        return (1.0 - nr.psnr_n, nr.params_n)

    def update(self, candidates: List[NormRecord]) -> None:
        merged = list(self._archive) + list(candidates)

        # Pareto pool
        nd = pareto_filter(merged, self._obj_min)

        if len(nd) <= self.max_size:
            self._archive = nd
            return

        objs = [self._obj_min(x) for x in nd]
        cd = crowding_distance(objs)

        # sort by descending crowding
        ranked = sorted(range(len(nd)), key=lambda i: cd[i], reverse=True)

        selected = [nd[i] for i in ranked[: self.max_size]]

        # safety Pareto pass (usually redundant)
        self._archive = pareto_filter(selected, self._obj_min)

    def front(self) -> List[NormRecord]:
        return list(self._archive)

    def state(self) -> Dict[str, Any]:
        return {
            "size": len(self._archive),
            "max_size": self.max_size,
        }
