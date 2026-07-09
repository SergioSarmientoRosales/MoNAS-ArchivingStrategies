from __future__ import annotations

import numpy as np
import pandas as pd

from monas_archiving.archivers import SUPPORTED_ARCHIVERS, run_archiver
from monas_archiving.indicators import evaluate_indicators


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "architecture_id": [f"a{i}" for i in range(8)],
            "norm_psnr": [0.05, 0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 0.30],
            "norm_params": [0.90, 0.70, 0.55, 0.35, 0.25, 0.16, 0.08, 0.80],
        }
    )


def test_each_archiver_returns_expected_size_and_is_deterministic() -> None:
    frame = _frame()
    for method in SUPPORTED_ARCHIVERS:
        first = run_archiver(frame, ["norm_psnr", "norm_params"], method, 3, "architecture_id", seed=9)
        second = run_archiver(frame, ["norm_psnr", "norm_params"], method, 3, "architecture_id", seed=9)
        assert len(first) == 3, method
        assert first["architecture_id"].tolist() == second["architecture_id"].tolist(), method


def test_indicators_return_finite_values_on_toy_data() -> None:
    reference = np.array([[0.0, 1.0], [0.4, 0.4], [1.0, 0.0]])
    approximation = np.array([[0.1, 0.9], [0.5, 0.3], [0.9, 0.1]])
    values = evaluate_indicators(
        reference,
        approximation,
        ("igd_plus", "hypervolume", "r2", "epsilon", "hausdorff"),
        hv_reference_point=(1.1, 1.1),
    )
    assert set(values) == {"igd_plus", "hypervolume", "r2", "epsilon", "hausdorff"}
    for value in values.values():
        assert np.isfinite(value)
