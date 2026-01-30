# archivers/eps2.py
from __future__ import annotations

from typing import List, Sequence, Dict, Any
from archivers.records import NormRecord



def pareto_dominates_min(p: Sequence[float], a: Sequence[float]) -> bool:
    """Standard Pareto dominance for minimization."""
    return all(pi <= ai for pi, ai in zip(p, a)) and any(pi < ai for pi, ai in zip(p, a))


def eps_dominates_theta_min(
    a: Sequence[float],
    p: Sequence[float],
    eps: Sequence[float],
    theta: float
) -> bool:
    """
    ε-dominance (minimization) with safety factor theta:
      a ≺_{θ·ε} p  ⟺  (a - θ·ε) <= p componentwise AND strict in at least one dim.
    """
    shifted = [ai - theta * ei for ai, ei in zip(a, eps)]
    return all(si <= pi for si, pi in zip(shifted, p)) and any(si < pi for si, pi in zip(shifted, p))


class Eps2Archiver:
    """
    ArchiveUpdateEps2 (Appendix B.3)

    Maintains an ε-Pareto archive where all points are Pareto optimal.
    Operates in minimization space.
    """
    name = "eps2"

    def __init__(self, eps=(0.02, 0.02), theta: float = 0.5):
        self.eps = tuple(float(e) for e in eps)
        self.theta = float(theta)
        self._archive: List[NormRecord] = []

    def reset(self) -> None:
        self._archive = []

    @staticmethod
    def _obj_min(nr: NormRecord) -> List[float]:
        # Convert to minimization objectives
        return [1.0 - nr.psnr_n, nr.params_n]

    def update(self, candidates: List[NormRecord]) -> None:
        A = list(self._archive)

        for p in candidates:
            p_obj = self._obj_min(p)

            # Lines 3–5: if NOT ε-dominated, add p
            if not any(
                eps_dominates_theta_min(self._obj_min(a), p_obj, self.eps, self.theta)
                for a in A
            ):
                A.append(p)

            # Lines 6–10: remove all archive points dominated by p
            A = [
                a for a in A
                if not pareto_dominates_min(p_obj, self._obj_min(a))
            ]

        self._archive = A

    def front(self) -> List[NormRecord]:
        return list(self._archive)

    def state(self) -> Dict[str, Any]:
        return {
            "size": len(self._archive),
            "eps": self.eps,
            "theta": self.theta,
        }
