# archivers/tight1.py
from __future__ import annotations

from typing import List, Sequence, Dict, Any
from archivers.records import NormRecord



def pareto_dominates_min(a: Sequence[float], b: Sequence[float]) -> bool:
    """Standard Pareto dominance for minimization: a ≺ b."""
    return all(ai <= bi for ai, bi in zip(a, b)) and any(ai < bi for ai, bi in zip(a, b))


def eps_dominates_theta_min(a: Sequence[float], b: Sequence[float],
                           eps: Sequence[float], theta: float) -> bool:
    """a ≺_{theta*eps} b  (minimization)."""
    shifted = [ai - theta * ei for ai, ei in zip(a, eps)]
    return all(si <= bi for si, bi in zip(shifted, b)) and any(si < bi for si, bi in zip(shifted, b))


def d_inf(u: Sequence[float], v: Sequence[float]) -> float:
    """Chebyshev distance (L_infinity)."""
    return max(abs(ui - vi) for ui, vi in zip(u, v))


class Tight1Archiver:
    """
    ArchiveUpdateTight1(P, A0) (Chapter 7, pseudocode)

    Works in minimization space.
    We use objectives:
      f(p) = ( PSNR_N, Params_N) in [0,1]^2.

    Rejection rule (line 3):
      Reject p if:
        (exists a in A : a ≺ p)
        OR
        (exists a1 in A : a1 ≺_{theta*eps} p AND exists a2 in A : d_inf(F(a2), F(p)) <= delta_hat)

    Then (lines 6-11):
      Remove archive points dominated by p, and insert p.
    """
    name = "tight1"

    def __init__(self, eps=(0.02, 0.02), theta: float = 0.5, delta_hat: float = 0.02):
        self.eps = tuple(float(e) for e in eps)
        self.theta = float(theta)
        self.delta_hat = float(delta_hat)
        self._archive: List[NormRecord] = []

    def reset(self) -> None:
        self._archive = []

    @staticmethod
    def _obj_min(nr: NormRecord) -> List[float]:
        return [1.0 - nr.psnr_n, nr.params_n]

    def update(self, candidates: List[NormRecord]) -> None:
        A = list(self._archive)

        for p in candidates:
            p_obj = self._obj_min(p)

            # Line 3 condition (two-part rejection)
            # Part 1: dominated by some archive point -> reject
            if any(pareto_dominates_min(self._obj_min(a), p_obj) for a in A):
                continue

            # Part 2: eps-dominated AND "too close" to existing archive point -> reject
            eps_dominated = any(
                eps_dominates_theta_min(self._obj_min(a1), p_obj, self.eps, self.theta)
                for a1 in A
            )
            if eps_dominated:
                close_to_archive = any(
                    d_inf(self._obj_min(a2), p_obj) <= self.delta_hat
                    for a2 in A
                )
                if close_to_archive:
                    continue

            # Lines 6–10: remove archive points dominated by p
            A = [a for a in A if not pareto_dominates_min(p_obj, self._obj_min(a))]

            # Line 11: insert p
            A.append(p)

        self._archive = A

    def front(self) -> List[NormRecord]:
        return list(self._archive)

    def state(self) -> Dict[str, Any]:
        return {
            "size": len(self._archive),
            "eps": self.eps,
            "theta": self.theta,
            "delta_hat": self.delta_hat,
        }
