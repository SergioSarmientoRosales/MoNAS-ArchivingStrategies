# archivers/kmeans.py
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import math
import random

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


def euclidean(u: Tuple[float, float], v: Tuple[float, float]) -> float:
    return math.sqrt((u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2)


def kmeans_2d(points: List[Tuple[float, float]],
              k: int,
              iters: int = 50,
              seed: int = 0) -> Tuple[List[int], List[Tuple[float, float]]]:
    """
    Minimal k-means for 2D.
    Returns: (labels, centers)
    """
    rng = random.Random(seed)
    n = len(points)
    if k <= 0:
        raise ValueError("k must be positive.")
    if k >= n:
        # each point is its own cluster
        labels = list(range(n))
        centers = list(points)
        return labels, centers

    # init centers by sampling distinct points
    centers = [points[i] for i in rng.sample(range(n), k)]
    labels = [0] * n

    for _ in range(iters):
        changed = False

        # assign step
        for i, p in enumerate(points):
            best_j = 0
            best_d = float("inf")
            for j, c in enumerate(centers):
                d = euclidean(p, c)
                if d < best_d:
                    best_d = d
                    best_j = j
            if labels[i] != best_j:
                labels[i] = best_j
                changed = True

        # recompute step
        sums = [(0.0, 0.0, 0) for _ in range(k)]  # (sx, sy, cnt)
        tmp = list(sums)
        for i, p in enumerate(points):
            j = labels[i]
            sx, sy, cnt = tmp[j]
            tmp[j] = (sx + p[0], sy + p[1], cnt + 1)

        new_centers: List[Tuple[float, float]] = []
        for j in range(k):
            sx, sy, cnt = tmp[j]
            if cnt == 0:
                # empty cluster -> re-seed randomly
                new_centers.append(points[rng.randrange(n)])
            else:
                new_centers.append((sx / cnt, sy / cnt))

        centers = new_centers
        if not changed:
            break

    return labels, centers


class KMeansArchiver:
    """
    KMeans-based Pareto archiver.

    Steps:
      1) Merge archive + candidates
      2) Pareto-filter -> ND pool
      3) If |ND| <= max_size: keep ND
      4) Else: cluster ND in objective space into k=max_size clusters
      5) Choose one representative per cluster:
           - pick the point closest to the cluster centroid
      6) Pareto-filter again (safety)

    Objective space (minimization, normalized):
      f(p) = (1 - PSNR_N, Params_N)
    """
    name = "kmeans"

    def __init__(self, max_size: int = 20, iters: int = 50, seed: int = 0):
        if max_size <= 0:
            raise ValueError("max_size must be positive.")
        self.max_size = int(max_size)
        self.iters = int(iters)
        self.seed = int(seed)
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

        # k-means clustering in objective space
        k = self.max_size
        labels, centers = kmeans_2d(objs, k=k, iters=self.iters, seed=self.seed)

        # pick representative per cluster: closest to centroid
        reps: Dict[int, int] = {}  # cluster -> index in nd
        best_dist: Dict[int, float] = {}

        for i, (f1, f2) in enumerate(objs):
            c = labels[i]
            d = euclidean((f1, f2), centers[c])
            if (c not in reps) or (d < best_dist[c]):
                reps[c] = i
                best_dist[c] = d

        selected = [nd[i] for i in reps.values()]

        # safety Pareto filter (should remain ND, but keep it robust)
        self._archive = pareto_filter(selected, self._obj_min)

        # if Pareto filter reduced size too much (rare), top-up by adding next-closest points
        if len(self._archive) < self.max_size:
            chosen = set(id(x) for x in self._archive)
            # rank all by distance to their centroid
            ranked = sorted(
                range(len(nd)),
                key=lambda i: euclidean(objs[i], centers[labels[i]])
            )
            for i in ranked:
                if len(self._archive) >= self.max_size:
                    break
                cand = nd[i]
                if id(cand) in chosen:
                    continue
                # insert while maintaining non-dominance
                cand_obj = self._obj_min(cand)
                if any(pareto_dominates_min(self._obj_min(a), cand_obj) for a in self._archive):
                    continue
                self._archive = [a for a in self._archive if not pareto_dominates_min(cand_obj, self._obj_min(a))]
                self._archive.append(cand)
                chosen.add(id(cand))

    def front(self) -> List[NormRecord]:
        return list(self._archive)

    def state(self) -> Dict[str, Any]:
        return {
            "size": len(self._archive),
            "max_size": self.max_size,
            "iters": self.iters,
            "seed": self.seed,
        }
