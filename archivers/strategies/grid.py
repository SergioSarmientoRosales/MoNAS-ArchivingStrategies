# archivers/grid_pareto.py
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
from archivers.records import NormRecord



def pareto_dominates_min(a: Sequence[float], b: Sequence[float]) -> bool:
    """Standard Pareto dominance for minimization: a ≺ b."""
    return all(ai <= bi for ai, bi in zip(a, b)) and any(ai < bi for ai, bi in zip(a, b))


class GridParetoArchiver:
    """
    Grid archiver with Pareto guarantee:

    1) Maintain a Pareto archive A (pairwise non-dominated) in minimization space:
         f(p) = (1 - PSNR_N, Params_N) in [0,1]^2
    2) Apply a grid to keep at most one representative per cell among non-dominated points.

    Returned front is ALWAYS pairwise non-dominated.
    """
    name = "grid_pareto"

    def __init__(self, bins_x: int = 30, bins_y: int = 30):
        if bins_x <= 0 or bins_y <= 0:
            raise ValueError("bins_x and bins_y must be positive.")
        self.bins_x = int(bins_x)
        self.bins_y = int(bins_y)
        self._archive: List[NormRecord] = []

    def reset(self) -> None:
        self._archive = []

    @staticmethod
    def _obj_min(nr: NormRecord) -> Tuple[float, float]:
        return (1.0 - nr.psnr_n, nr.params_n)

    def _cell_index(self, f1: float, f2: float) -> Tuple[int, int]:
        # clamp to [0,1]
        f1 = 0.0 if f1 < 0.0 else (1.0 if f1 > 1.0 else f1)
        f2 = 0.0 if f2 < 0.0 else (1.0 if f2 > 1.0 else f2)
        ix = min(self.bins_x - 1, int(f1 * self.bins_x))
        iy = min(self.bins_y - 1, int(f2 * self.bins_y))
        return (ix, iy)

    def _pareto_insert(self, p: NormRecord) -> None:
        """Insert p into self._archive maintaining pairwise non-dominance."""
        p_obj = self._obj_min(p)

        # Reject if dominated by archive
        if any(pareto_dominates_min(self._obj_min(a), p_obj) for a in self._archive):
            return

        # Remove archive points dominated by p
        self._archive = [a for a in self._archive if not pareto_dominates_min(p_obj, self._obj_min(a))]
        self._archive.append(p)

    def _grid_compress(self) -> None:
        """
        Keep at most one point per cell.
        Since input is Pareto (pairwise non-dominated), output remains Pareto.
        Representative choice: prefer smaller f1 (better PSNR), tie-break smaller f2.
        """
        cells: Dict[Tuple[int, int], NormRecord] = {}
        for p in self._archive:
            f1, f2 = self._obj_min(p)
            key = self._cell_index(f1, f2)

            if key not in cells:
                cells[key] = p
            else:
                q = cells[key]
                qf1, qf2 = self._obj_min(q)
                # choose lexicographically best in minimization
                if (f1 < qf1) or (f1 == qf1 and f2 < qf2):
                    cells[key] = p

        self._archive = list(cells.values())

        # Safety: re-apply Pareto filter to remove any accidental dominance (rare but safe)
        filtered: List[NormRecord] = []
        for p in self._archive:
            p_obj = self._obj_min(p)
            if any(pareto_dominates_min(self._obj_min(a), p_obj) for a in filtered):
                continue
            filtered = [a for a in filtered if not pareto_dominates_min(p_obj, self._obj_min(a))]
            filtered.append(p)
        self._archive = filtered

    def update(self, candidates: List[NormRecord]) -> None:
        # Step 1: Pareto maintenance
        for p in candidates:
            self._pareto_insert(p)

        # Step 2: Grid compression
        self._grid_compress()

    def front(self) -> List[NormRecord]:
        return list(self._archive)

    def state(self) -> Dict[str, Any]:
        return {
            "size": len(self._archive),
            "bins_x": self.bins_x,
            "bins_y": self.bins_y,
        }
