from __future__ import annotations

from typing import List, Dict, Any, Callable
from archivers.records import NormRecord

from archivers.pareto import dominates


def _crowding_distance_2d(front: List[NormRecord], f1: Callable[[NormRecord], float], f2: Callable[[NormRecord], float]) -> List[float]:
    n = len(front)
    if n == 0:
        return []
    if n <= 2:
        return [float("inf")] * n

    dist = [0.0] * n

    def compute_for(getv: Callable[[NormRecord], float]) -> None:
        idx = sorted(range(n), key=lambda i: getv(front[i]))
        dist[idx[0]] = float("inf")
        dist[idx[-1]] = float("inf")

        min_val = getv(front[idx[0]])
        max_val = getv(front[idx[-1]])
        denom = max_val - min_val
        if denom <= 0.0:
            return

        for pos in range(1, n - 1):
            i_prev = idx[pos - 1]
            i_curr = idx[pos]
            i_next = idx[pos + 1]
            dist[i_curr] += (getv(front[i_next]) - getv(front[i_prev])) / denom

    compute_for(f1)
    compute_for(f2)
    return dist


def _truncate_by_crowding(front: List[NormRecord], K: int, f1: Callable[[NormRecord], float], f2: Callable[[NormRecord], float]) -> List[NormRecord]:
    if len(front) <= K:
        return list(front)

    cd = _crowding_distance_2d(front, f1=f1, f2=f2)

    # Desc by crowding; stable tie-breaker by (f1,f2)
    idx_sorted = sorted(
        range(len(front)),
        key=lambda i: (cd[i], -f1(front[i]), -f2(front[i])),
        reverse=True,
    )
    keep = set(idx_sorted[:K])
    return [front[i] for i in range(len(front)) if i in keep]


class PQArchiver:
    """
    Pareto ND archive truncated to a strict budget K using crowding distance.
    """

    name = "pqk"

    def __init__(
        self,
        max_size: int,
        *,
        f1: Callable[[NormRecord], float],
        f2: Callable[[NormRecord], float],
    ):
        if max_size <= 0:
            raise ValueError("max_size must be >= 1")
        self.max_size = int(max_size)
        self._f1 = f1
        self._f2 = f2
        self._archive: List[NormRecord] = []

    def reset(self) -> None:
        self._archive = []

    def update(self, candidates: List[NormRecord]) -> None:
        pool = self._archive + list(candidates)
        nd: List[NormRecord] = []

        for x in pool:
            if any(dominates(a, x) for a in nd):
                continue
            nd = [a for a in nd if not dominates(x, a)]
            nd.append(x)

        self._archive = _truncate_by_crowding(nd, self.max_size, f1=self._f1, f2=self._f2)

    def front(self) -> List[NormRecord]:
        return list(self._archive)

    def state(self) -> Dict[str, Any]:
        return {"size": len(self._archive), "max_size": self.max_size}
