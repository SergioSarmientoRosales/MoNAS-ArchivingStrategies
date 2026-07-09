from __future__ import annotations

import numpy as np
import pandas as pd

from monas_archiving.config import ObjectiveSpec
from monas_archiving.data import deduplicate_by_key
from monas_archiving.normalization import ObjectiveNormalizer
from monas_archiving.pareto import dominates, nondominated_mask


def test_dominance_minimization() -> None:
    assert dominates(np.array([0.1, 0.2]), np.array([0.2, 0.2]))
    assert not dominates(np.array([0.1, 0.3]), np.array([0.2, 0.2]))


def test_nondominated_mask() -> None:
    points = np.array(
        [
            [0.1, 0.8],
            [0.2, 0.5],
            [0.4, 0.4],
            [0.3, 0.7],
        ]
    )
    assert nondominated_mask(points).tolist() == [True, True, True, False]


def test_normalization_respects_objective_directions() -> None:
    frame = pd.DataFrame(
        {
            "architecture_id": ["a", "b"],
            "psnr": [30.0, 35.0],
            "params": [1000.0, 2000.0],
        }
    )
    objectives = (
        ObjectiveSpec("psnr", "maximize"),
        ObjectiveSpec("params", "minimize"),
    )
    normalized = ObjectiveNormalizer.fit(frame, objectives).transform(frame)
    assert normalized.loc[1, "norm_psnr"] == 0.0
    assert normalized.loc[0, "norm_psnr"] == 1.0
    assert normalized.loc[0, "norm_params"] == 0.0
    assert normalized.loc[1, "norm_params"] == 1.0


def test_deduplicate_keeps_best_normalized_row() -> None:
    frame = pd.DataFrame(
        {
            "architecture_id": ["same", "same", "other"],
            "norm_psnr": [0.4, 0.2, 0.3],
            "norm_params": [0.4, 0.1, 0.3],
        }
    )
    deduped = deduplicate_by_key(frame, "architecture_id", ["norm_psnr", "norm_params"])
    assert len(deduped) == 2
    same = deduped[deduped["architecture_id"] == "same"].iloc[0]
    assert same["norm_psnr"] == 0.2
