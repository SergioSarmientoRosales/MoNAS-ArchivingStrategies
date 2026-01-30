from __future__ import annotations
from typing import Any, Dict, List, Sequence, Tuple
from archivers.records import NormRecord



def _leq_all(u: Sequence[float], v: Sequence[float]) -> bool:
    """Check if u <= v componentwise."""
    return all(ui <= vi for ui, vi in zip(u, v))


def _lt_any(u: Sequence[float], v: Sequence[float]) -> bool:
    """Check if u < v in at least one dimension."""
    return any(ui < vi for ui, vi in zip(u, v))


def minus_eps_dominates_min(
    a: Sequence[float],
    p: Sequence[float],
    eps: Sequence[float],
) -> bool:
    """
    "-epsilon" dominance for MINIMIZATION (used for PQ,ε):

        a ≺^-_ε p   ⇔   (a + ε) <= p componentwise
                       AND strict in at least one objective.

    Intuition: p is worse than a by at least ε in every objective.
    """
    shifted = [ai + ei for ai, ei in zip(a, eps)]
    return _leq_all(shifted, p) and _lt_any(shifted, p)


class PQEpsArchiver:
    """
    ArchiveUpdateP_{Q,ε}(P, A₀) — archive update with ε-dominance.

    Rules:
      - Insert p if there does NOT exist a ∈ A such that a ≺^-_ε p
      - Remove any a ∈ A such that p ≺^-_ε a
    """

    name = "pq_eps"

    def __init__(self, eps: Sequence[float] = (0.02, 0.02)):
        if len(eps) != 2:
            raise ValueError("eps must be a 2D vector (eps_f1, eps_f2).")
        if eps[0] <= 0 or eps[1] <= 0:
            raise ValueError("eps components must be > 0.")
        self.eps: Tuple[float, float] = (float(eps[0]), float(eps[1]))
        self._archive: List[NormRecord] = []

    def reset(self) -> None:
        """Clear the archive."""
        self._archive = []

    @staticmethod
    def _obj_min(nr: NormRecord) -> Tuple[float, float]:
        """
        Convert a normalized record to minimization objectives:
          f1 = PSNR_n  (maximize PSNR → minimize 1 - PSNR_n)
          f2 = Params_n    (minimize parameters)
        """
        f1 = 1.0 - nr.psnr_n
        f2 = nr.params_n
        return (f1, f2)

    def update(self, candidates: List[NormRecord]) -> None:
        """Update the archive with a list of candidates."""
        A = list(self._archive)

        for p in candidates:
            p_obj = self._obj_min(p)

            # Acceptance condition:
            # accept if NOT (exists a ∈ A : a ≺^-_ε p)
            is_minus_eps_dominated = any(
                minus_eps_dominates_min(self._obj_min(a), p_obj, self.eps)
                for a in A
            )
            if is_minus_eps_dominated:
                continue

            # Add p
            A.append(p)

            # Remove all a ∈ A such that p ≺^-_ε a
            A = [
                a for a in A
                if not minus_eps_dominates_min(p_obj, self._obj_min(a), self.eps)
            ]

        self._archive = A

    def front(self) -> List[NormRecord]:
        """Return the current archive (approximate front)."""
        return list(self._archive)

    def state(self) -> Dict[str, Any]:
        """Return archive state."""
        return {"size": len(self._archive), "eps": self.eps}