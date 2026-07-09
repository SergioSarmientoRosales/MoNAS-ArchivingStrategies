from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from monas_archiving.config import ObjectiveSpec


@dataclass(frozen=True)
class ObjectiveNormalizer:
    """Min-max normalizer that converts all objectives to minimization form."""

    ranges: dict[str, tuple[float, float]]
    objectives: tuple[ObjectiveSpec, ...]

    @classmethod
    def fit(cls, frame: pd.DataFrame, objectives: tuple[ObjectiveSpec, ...]) -> "ObjectiveNormalizer":
        ranges: dict[str, tuple[float, float]] = {}
        for objective in objectives:
            values = frame[objective.column].to_numpy(dtype=float)
            ranges[objective.column] = (float(np.min(values)), float(np.max(values)))
        return cls(ranges=ranges, objectives=objectives)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add normalized objective columns, where 0 is best and 1 is worst."""
        out = frame.copy()
        for objective in self.objectives:
            lo, hi = self.ranges[objective.column]
            values = out[objective.column].to_numpy(dtype=float)
            if hi == lo:
                normalized = np.zeros_like(values, dtype=float)
            elif objective.direction == "minimize":
                normalized = (values - lo) / (hi - lo)
            else:
                normalized = (hi - values) / (hi - lo)
            out[objective.normalized_column] = normalized
        return out
