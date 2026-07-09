from __future__ import annotations

from pathlib import Path

import pandas as pd

from monas_archiving.archivers import run_archiver
from monas_archiving.config import PipelineConfig, dump_config, load_config
from monas_archiving.data import (
    deduplicate_by_key,
    deterministic_sort,
    load_solution_csv,
    objective_matrix,
)
from monas_archiving.indicators import INDICATOR_DIRECTIONS, evaluate_indicators
from monas_archiving.normalization import ObjectiveNormalizer
from monas_archiving.pareto import nondominated_frame
from monas_archiving.plotting import plot_metric_bars, plot_objective_scatter


def ensure_run_dirs(config: PipelineConfig) -> None:
    """Create the standard output directory tree."""
    for path in [
        config.run_dir,
        config.run_dir / "archives",
        config.run_dir / "metrics",
        config.run_dir / "figures",
        config.run_dir / "logs",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def build_solution_cloud(config: PipelineConfig) -> pd.DataFrame:
    """Load, validate, normalize, deduplicate, and save the solution cloud."""
    ensure_run_dirs(config)
    dump_config(config, config.run_dir / "config_used.yaml")

    raw = load_solution_csv(
        config.input_path,
        objectives=config.objectives,
        architecture_id_column=config.architecture_id_column,
        chromosome_column=config.chromosome_column,
    )
    raw = deterministic_sort(raw, config.deduplication_key)
    normalizer = ObjectiveNormalizer.fit(raw, config.objectives)
    normalized = normalizer.transform(raw)
    cloud = deduplicate_by_key(
        normalized,
        key_column=config.deduplication_key,
        normalized_objective_columns=config.normalized_objective_columns,
    )
    cloud = deterministic_sort(cloud, config.deduplication_key)
    cloud.to_csv(config.run_dir / "solution_cloud.csv", index=False)
    return cloud


def build_reference_front(config: PipelineConfig) -> pd.DataFrame:
    """Build and save the empirical reference Pareto front."""
    cloud_path = config.run_dir / "solution_cloud.csv"
    cloud = pd.read_csv(cloud_path) if cloud_path.exists() else build_solution_cloud(config)
    reference = nondominated_frame(cloud, config.normalized_objective_columns)
    reference = deterministic_sort(reference, config.deduplication_key)
    reference.to_csv(config.run_dir / "reference_front.csv", index=False)
    return reference


def run_archivers(config: PipelineConfig) -> list[Path]:
    """Run every configured archiver and save one archive per k."""
    cloud_path = config.run_dir / "solution_cloud.csv"
    cloud = pd.read_csv(cloud_path) if cloud_path.exists() else build_solution_cloud(config)
    written: list[Path] = []

    for archiver in config.archivers:
        archiver_params = dict(archiver)
        method = str(archiver_params.pop("name"))
        for k in config.truncation_sizes:
            archive = run_archiver(
                cloud,
                objective_columns=config.normalized_objective_columns,
                method=method,
                k=int(k),
                id_column=config.deduplication_key,
                seed=config.seed,
                **archiver_params,
            )
            archive.insert(0, "archiver", method)
            archive.insert(1, "k", int(k))
            out_path = config.run_dir / "archives" / f"{method}_k{k}.csv"
            archive.to_csv(out_path, index=False)
            written.append(out_path)
    return written


def evaluate_archives(config: PipelineConfig) -> pd.DataFrame:
    """Evaluate saved archives against the reference front."""
    reference_path = config.run_dir / "reference_front.csv"
    if not reference_path.exists():
        build_reference_front(config)
    if not list((config.run_dir / "archives").glob("*.csv")):
        run_archivers(config)

    reference = pd.read_csv(reference_path)
    reference_points = objective_matrix(reference, config.normalized_objective_columns)
    rows: list[dict[str, object]] = []

    for archive_path in sorted((config.run_dir / "archives").glob("*.csv")):
        archive = pd.read_csv(archive_path)
        archiver = str(archive["archiver"].iloc[0]) if "archiver" in archive else archive_path.stem
        k = int(archive["k"].iloc[0]) if "k" in archive else len(archive)
        approximation_points = objective_matrix(archive, config.normalized_objective_columns)
        values = evaluate_indicators(
            reference_points,
            approximation_points,
            config.indicators,
            hv_reference_point=config.hv_reference_point,
        )
        for indicator, value in values.items():
            rows.append(
                {
                    "archiver": archiver,
                    "k": k,
                    "indicator": indicator,
                    "value": value,
                    "direction": INDICATOR_DIRECTIONS[indicator],
                    "archive_file": str(archive_path.relative_to(config.run_dir)),
                }
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(config.run_dir / "metrics" / "archive_metrics.csv", index=False)
    return metrics


def generate_plots(config: PipelineConfig) -> None:
    """Generate lightweight reproducibility figures."""
    if not config.plot:
        return
    solution_path = config.run_dir / "solution_cloud.csv"
    reference_path = config.run_dir / "reference_front.csv"
    metrics_path = config.run_dir / "metrics" / "archive_metrics.csv"

    cloud = pd.read_csv(solution_path) if solution_path.exists() else build_solution_cloud(config)
    reference = pd.read_csv(reference_path) if reference_path.exists() else build_reference_front(config)
    plot_objective_scatter(
        cloud,
        reference,
        config.normalized_objective_columns,
        config.run_dir / "figures" / "solution_cloud_reference_front.png",
    )

    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
    else:
        metrics = evaluate_archives(config)
    plot_metric_bars(metrics, config.run_dir / "figures" / "archive_metrics.png")


def run_pipeline(config: PipelineConfig) -> PipelineConfig:
    """Run the full offline archiving pipeline."""
    build_solution_cloud(config)
    build_reference_front(config)
    run_archivers(config)
    evaluate_archives(config)
    generate_plots(config)
    return config


def run_pipeline_from_config(config_path: str | Path) -> PipelineConfig:
    """Load a YAML config and run the full pipeline."""
    config = load_config(config_path)
    return run_pipeline(config)
