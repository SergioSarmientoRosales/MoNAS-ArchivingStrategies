# archivers/eps1.py
from __future__ import annotations

from typing import List, Dict, Any, Sequence

from archivers.records import NormRecord



def _leq_all(u: Sequence[float], v: Sequence[float]) -> bool:
    return all(ui <= vi for ui, vi in zip(u, v))


def _lt_any(u: Sequence[float], v: Sequence[float]) -> bool:
    return any(ui < vi for ui, vi in zip(u, v))


def pareto_dominates_min(p: Sequence[float], a: Sequence[float]) -> bool:
    """
    Standard Pareto dominance for minimization:
      p ≺ a  <=>  p <= a componentwise AND p != a
    """
    return _leq_all(p, a) and _lt_any(p, a)


def eps_dominates_theta_min(a: Sequence[float], p: Sequence[float],
                           eps: Sequence[float], theta: float) -> bool:
    """
    ε-dominance (minimization) with safety factor theta:
      a ≺_{theta*eps} p  <=>  a - theta*eps <= p componentwise AND strict in at least one dim.
    """
    shifted = [ai - theta * ei for ai, ei in zip(a, eps)]
    return _leq_all(shifted, p) and _lt_any(shifted, p)


class Eps1Archiver:
    """
    ArchiveUpdateEps1 baseline implementation (epsilon-approx archive):
      For each candidate p:
        1) If exists a in A such that a ε-dominates p, reject p.
        2) Else remove all archive points dominated by p (standard Pareto).
        3) Insert p.

    We operate in MINIMIZATION objective space:
      f(p) = (1 - PSNR_N, Params_N)
    """
    name = "eps1"

    def __init__(self, eps: Sequence[float] = (0.02, 0.02), theta: float = 0.5):
        if len(eps) != 2:
            raise ValueError("eps must be a 2D vector (eps_f1, eps_f2).")
        if eps[0] <= 0 or eps[1] <= 0:
            raise ValueError("eps components must be > 0.")
        if not (0.0 < theta < 1.0):
            raise ValueError("theta must satisfy 0 < theta < 1.")
        self.eps = (float(eps[0]), float(eps[1]))
        self.theta = float(theta)
        self._archive: List[NormRecord] = []

    def reset(self) -> None:
        self._archive = []

    @staticmethod
    def _obj_min(nr: NormRecord) -> List[float]:
        # minimization vector in [0,1]^2
        f1 = 1.0 - nr.psnr_n   # smaller is better (higher PSNR_n)
        f2 = nr.params_n       # smaller is better
        return [f1, f2]

    def update(self, candidates: List[NormRecord]) -> None:
        for p in candidates:
            p_obj = self._obj_min(p)

            # Line 3–5: reject p if ε-dominated by archive
            if any(
                eps_dominates_theta_min(self._obj_min(a), p_obj, self.eps, self.theta)
                for a in self._archive
            ):
                continue

            # Line 6–10: remove archive points dominated by p (standard Pareto)
            self._archive = [
                a for a in self._archive
                if not pareto_dominates_min(p_obj, self._obj_min(a))
            ]

            # Line 11: insert p
            self._archive.append(p)

    def front(self) -> List[NormRecord]:
        # Archive is already consistent with its update logic; return as-is.
        return list(self._archive)

    def state(self) -> Dict[str, Any]:
        return {"size": len(self._archive), "eps": self.eps, "theta": self.theta}
