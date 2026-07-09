from __future__ import annotations

from pathlib import Path

import pandas as pd

from monas_archiving.config import load_config
from monas_archiving.pipeline import run_pipeline


def test_toy_pipeline_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs" / "toy_example.yaml"
    config = load_config(config_path)
    config = type(config)(
        run_name="pytest_toy",
        input_path=config.input_path,
        output_dir=tmp_path,
        objectives=config.objectives,
        architecture_id_column=config.architecture_id_column,
        chromosome_column=config.chromosome_column,
        deduplication_key=config.deduplication_key,
        normalization=config.normalization,
        seed=config.seed,
        archivers=config.archivers,
        truncation_sizes=(3,),
        indicators=config.indicators,
        hv_reference_point=config.hv_reference_point,
        plot=False,
    )

    run_pipeline(config)
    assert (config.run_dir / "solution_cloud.csv").exists()
    assert (config.run_dir / "reference_front.csv").exists()
    metrics_path = config.run_dir / "metrics" / "archive_metrics.csv"
    assert metrics_path.exists()
    metrics = pd.read_csv(metrics_path)
    assert not metrics.empty
