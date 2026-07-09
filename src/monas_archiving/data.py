from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from monas_archiving.config import ObjectiveSpec


def load_solution_csv(
    path: str | Path,
    objectives: tuple[ObjectiveSpec, ...],
    architecture_id_column: str,
    chromosome_column: str | None,
) -> pd.DataFrame:
    """Load and validate a solution-cloud CSV file."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {csv_path}")

    frame = pd.read_csv(csv_path)
    required = {objective.column for objective in objectives}
    required.add(architecture_id_column)
    if chromosome_column:
        required.add(chromosome_column)

    missing = sorted(column for column in required if column not in frame.columns)
    if missing:
        raise ValueError(
            f"Input CSV {csv_path} is missing required columns: {missing}. "
            f"Available columns: {list(frame.columns)}"
        )

    frame = frame.copy()
    frame[architecture_id_column] = frame[architecture_id_column].astype(str)
    if chromosome_column:
        frame[chromosome_column] = frame[chromosome_column].astype(str)

    for objective in objectives:
        frame[objective.column] = pd.to_numeric(frame[objective.column], errors="coerce")

    before = len(frame)
    frame = frame.dropna(subset=[objective.column for objective in objectives])
    dropped = before - len(frame)
    if dropped:
        frame.attrs["dropped_missing_objectives"] = dropped

    if frame.empty:
        raise ValueError("No valid rows remain after dropping missing objective values.")

    return frame.reset_index(drop=True)


def deterministic_sort(frame: pd.DataFrame, key_column: str) -> pd.DataFrame:
    """Sort rows in a stable way for deterministic tie-breaking."""
    sort_columns = [key_column]
    if "model" in frame.columns:
        sort_columns.append("model")
    if "seed" in frame.columns:
        sort_columns.append("seed")
    return frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def deduplicate_by_key(
    frame: pd.DataFrame,
    key_column: str,
    normalized_objective_columns: list[str],
) -> pd.DataFrame:
    """Deduplicate architectures by keeping the best deterministic representative.

    The frame must already contain normalized minimization objectives where lower
    values are better. Duplicates are ranked by the sum of normalized objectives
    and then by each normalized objective lexicographically.
    """
    if key_column not in frame.columns:
        raise ValueError(f"Deduplication key column not found: {key_column}")

    working = frame.copy()
    working["_objective_sum"] = working[normalized_objective_columns].sum(axis=1)
    sort_columns = [key_column, "_objective_sum", *normalized_objective_columns]
    working = working.sort_values(sort_columns, kind="mergesort")
    deduplicated = working.drop_duplicates(subset=[key_column], keep="first")
    return deduplicated.drop(columns=["_objective_sum"]).reset_index(drop=True)


def objective_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Return objective columns as a floating-point matrix."""
    if not columns:
        raise ValueError("At least one objective column is required.")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing objective columns: {missing}")
    return frame[columns].to_numpy(dtype=float)
