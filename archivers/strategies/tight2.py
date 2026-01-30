# archivers/tight2.py
from __future__ import annotations

from typing import List, Sequence, Dict, Any
from archivers.records import NormRecord



def pareto_dominates_min(a: Sequence[float], b: Sequence[float]) -> bool:
    return all(ai <= bi for ai, bi in zip(a, b)) and any(ai < bi for ai, bi in zip(a, b))


def eps_dominates_theta_min(a: Sequence[float], b: Sequence[float],
                           eps: Sequence[float], theta: float) -> bool:
    shifted = [ai - theta * ei for ai, ei in zip(a, eps)]
    return all(si <= bi for si, bi in zip(shifted, b)) and any(si < bi for si, bi in zip(shifted, b))


def d_inf(u: Sequence[float], v: Sequence[float]) -> float:
    return max(abs(ui - vi) for ui, vi in zip(u, v))


class Tight2Archiver:
    name = "tight2"

    def __init__(self, eps=(0.02, 0.02), theta: float = 0.5,
                 delta: float = 0.01, delta_tilde: float = 0.02):
        self.eps = tuple(float(e) for e in eps)
        self.theta = float(theta)
        self.delta = float(delta)
        self.delta_tilde = float(delta_tilde)
        self._archive: List[NormRecord] = []

    def reset(self) -> None:
        self._archive = []

    @staticmethod
    def _obj_min(nr: NormRecord) -> List[float]:
        # minimization objectives in [0,1]^2
        return [1.0 - nr.psnr_n, nr.params_n]

    def update(self, candidates: List[NormRecord]) -> None:
        A = list(self._archive)

        for p in candidates:
            p_obj = self._obj_min(p)

            # Bootstrap: archive cannot start empty forever
            if not A:
                A.append(p)
                continue

            # E1: exists a1 in A such that a1 eps-theta-dominates p
            e1 = any(
                eps_dominates_theta_min(self._obj_min(a1), p_obj, self.eps, self.theta)
                for a1 in A
            )

            # E2: exists a2 in A such that a2 dominates p AND p is farther than delta_tilde from ALL A
            farther_than_all = all(
                d_inf(self._obj_min(a), p_obj) > self.delta_tilde
                for a in A
            )
            e2 = any(
                pareto_dominates_min(self._obj_min(a2), p_obj)
                for a2 in A
            ) and farther_than_all

            if e1 or e2:
                A.append(p)

            # Replacement rule: if p dominates any a in A, remove those a and ensure p is present
            dominated_any = any(
                pareto_dominates_min(p_obj, self._obj_min(a))
                for a in A
            )
            if dominated_any:
                A = [a for a in A if not pareto_dominates_min(p_obj, self._obj_min(a))]
                A.append(p)

        self._archive = A

    def front(self) -> List[NormRecord]:
        return list(self._archive)

    def state(self) -> Dict[str, Any]:
        return {
            "size": len(self._archive),
            "eps": self.eps,
            "theta": self.theta,
            "delta": self.delta,
            "delta_tilde": self.delta_tilde,
        }

