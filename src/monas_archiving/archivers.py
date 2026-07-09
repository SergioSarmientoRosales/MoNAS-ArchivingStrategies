from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from monas_archiving.indicators import hypervolume_2d, r2_indicator
from monas_archiving.pareto import nondominated_frame, truncate_by_crowding


SUPPORTED_ARCHIVERS = (
    "pq",
    "hv",
    "r2",
    "crowding",
    "grid",
    "epsilon",
    "tight1",
    "kmeans",
    "entropy",
)


def run_archiver(
    frame: pd.DataFrame,
    objective_columns: list[str],
    method: str,
    k: int,
    id_column: str,
    seed: int = 1,
    **params: Any,
) -> pd.DataFrame:
    """Run one deterministic offline archiving strategy."""
    method = method.lower()
    if method not in SUPPORTED_ARCHIVERS:
        raise ValueError(f"Unsupported archiver {method!r}. Supported: {SUPPORTED_ARCHIVERS}")
    if k <= 0:
        raise ValueError("k must be positive.")

    non_dominated = nondominated_frame(frame, objective_columns)
    non_dominated = non_dominated.sort_values(id_column, kind="mergesort").reset_index(drop=True)
    if len(non_dominated) <= k:
        return non_dominated

    if method == "pq":
        return truncate_by_crowding(non_dominated, objective_columns, k, id_column)
    if method == "crowding":
        return truncate_by_crowding(non_dominated, objective_columns, k, id_column)
    if method == "grid":
        return _grid_archive(non_dominated, objective_columns, k, id_column, bins=int(params.get("bins", 10)))
    if method == "epsilon":
        return _epsilon_archive(non_dominated, objective_columns, k, id_column, eps=float(params.get("eps", 0.05)))
    if method == "tight1":
        return _tight1_archive(non_dominated, objective_columns, k, id_column)
    if method == "kmeans":
        return _kmeans_archive(non_dominated, objective_columns, k, id_column, seed=seed, iters=int(params.get("iters", 50)))
    if method == "entropy":
        return _entropy_archive(non_dominated, objective_columns, k, id_column, bins=int(params.get("bins", 10)))
    if method == "hv":
        return _greedy_indicator_archive(non_dominated, objective_columns, k, id_column, indicator="hv")
    if method == "r2":
        return _greedy_indicator_archive(non_dominated, objective_columns, k, id_column, indicator="r2")
    raise AssertionError("Unreachable archiver branch.")


def _finish_selection(
    frame: pd.DataFrame,
    selected_indices: list[int],
    objective_columns: list[str],
    k: int,
    id_column: str,
) -> pd.DataFrame:
    selected = frame.iloc[selected_indices].copy()
    if len(selected) > k:
        selected = truncate_by_crowding(selected, objective_columns, k, id_column)
    return selected.sort_values(id_column, kind="mergesort").reset_index(drop=True)


def _best_row_per_group(
    frame: pd.DataFrame,
    group_keys: np.ndarray,
    objective_columns: list[str],
    id_column: str,
) -> list[int]:
    working = frame.copy()
    working["_group_key"] = [tuple(row) for row in group_keys]
    working["_objective_sum"] = working[objective_columns].sum(axis=1)
    working["_source_index"] = np.arange(len(working))
    ordered = working.sort_values(
        ["_group_key", "_objective_sum", *objective_columns, id_column],
        kind="mergesort",
    )
    return ordered.drop_duplicates("_group_key", keep="first")["_source_index"].astype(int).tolist()


def _grid_archive(
    frame: pd.DataFrame,
    objective_columns: list[str],
    k: int,
    id_column: str,
    bins: int,
) -> pd.DataFrame:
    points = frame[objective_columns].to_numpy(dtype=float)
    bins = max(1, bins)
    group_keys = np.floor(np.clip(points, 0.0, 1.0) * bins).astype(int)
    selected = _best_row_per_group(frame, group_keys, objective_columns, id_column)
    archive = _finish_selection(frame, selected, objective_columns, k, id_column)
    if len(archive) < k:
        archive = truncate_by_crowding(frame, objective_columns, k, id_column)
    return archive


def _epsilon_archive(
    frame: pd.DataFrame,
    objective_columns: list[str],
    k: int,
    id_column: str,
    eps: float,
) -> pd.DataFrame:
    if eps <= 0:
        raise ValueError("epsilon archive parameter 'eps' must be positive.")
    points = frame[objective_columns].to_numpy(dtype=float)
    group_keys = np.floor(points / eps).astype(int)
    selected = _best_row_per_group(frame, group_keys, objective_columns, id_column)
    archive = _finish_selection(frame, selected, objective_columns, k, id_column)
    if len(archive) < k:
        archive = truncate_by_crowding(frame, objective_columns, k, id_column)
    return archive


def _tight1_archive(
    frame: pd.DataFrame,
    objective_columns: list[str],
    k: int,
    id_column: str,
) -> pd.DataFrame:
    points = frame[objective_columns].to_numpy(dtype=float)
    first_order = np.argsort(points[:, 0], kind="mergesort")
    selected: list[int] = [int(first_order[0]), int(first_order[-1])]
    selected = list(dict.fromkeys(selected))

    while len(selected) < k:
        best_index = None
        best_distance = -1.0
        for index in range(len(frame)):
            if index in selected:
                continue
            distances = np.linalg.norm(points[selected] - points[index], axis=1)
            min_distance = float(np.min(distances))
            tie_breaker = tuple(points[index]) + (str(frame.iloc[index][id_column]),)
            if min_distance > best_distance:
                best_distance = min_distance
                best_index = index
                best_tie = tie_breaker
            elif min_distance == best_distance and best_index is not None and tie_breaker < best_tie:
                best_index = index
                best_tie = tie_breaker
        if best_index is None:
            break
        selected.append(best_index)
    return _finish_selection(frame, selected, objective_columns, k, id_column)


def _kmeans_archive(
    frame: pd.DataFrame,
    objective_columns: list[str],
    k: int,
    id_column: str,
    seed: int,
    iters: int,
) -> pd.DataFrame:
    points = frame[objective_columns].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    if len(points) <= k:
        return frame.reset_index(drop=True)

    first_center = int(np.argmin(points.sum(axis=1)))
    centers = [points[first_center]]
    while len(centers) < k:
        distances = np.min(
            np.stack([np.linalg.norm(points - center, axis=1) for center in centers]),
            axis=0,
        )
        probabilities = distances / distances.sum() if distances.sum() else np.ones(len(points)) / len(points)
        centers.append(points[int(rng.choice(np.arange(len(points)), p=probabilities))])
    centers_array = np.vstack(centers)

    labels = np.zeros(len(points), dtype=int)
    for _ in range(max(1, iters)):
        distances = np.stack([np.linalg.norm(points - center, axis=1) for center in centers_array], axis=1)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(k):
            cluster_points = points[labels == cluster]
            if len(cluster_points):
                centers_array[cluster] = cluster_points.mean(axis=0)

    selected: list[int] = []
    for cluster in range(k):
        cluster_indices = np.where(labels == cluster)[0]
        if not len(cluster_indices):
            continue
        distances = np.linalg.norm(points[cluster_indices] - centers_array[cluster], axis=1)
        best_local = cluster_indices[int(np.argmin(distances))]
        selected.append(int(best_local))

    return _finish_selection(frame, list(dict.fromkeys(selected)), objective_columns, k, id_column)


def _entropy_archive(
    frame: pd.DataFrame,
    objective_columns: list[str],
    k: int,
    id_column: str,
    bins: int,
) -> pd.DataFrame:
    points = frame[objective_columns].to_numpy(dtype=float)
    group_keys = np.floor(np.clip(points, 0.0, 1.0) * max(1, bins)).astype(int)
    cells, counts = np.unique(group_keys, axis=0, return_counts=True)
    count_by_cell = {tuple(cell): count for cell, count in zip(cells, counts)}
    rarity = np.array([1.0 / count_by_cell[tuple(cell)] for cell in group_keys])
    working = frame.copy()
    working["_rarity"] = rarity
    working["_objective_sum"] = working[objective_columns].sum(axis=1)
    working["_source_index"] = np.arange(len(frame))
    ordered = working.sort_values(
        ["_rarity", "_objective_sum", id_column],
        ascending=[False, True, True],
        kind="mergesort",
    )
    selected = ordered.head(k)["_source_index"].astype(int).tolist()
    return _finish_selection(frame, selected, objective_columns, k, id_column)


def _greedy_indicator_archive(
    frame: pd.DataFrame,
    objective_columns: list[str],
    k: int,
    id_column: str,
    indicator: str,
) -> pd.DataFrame:
    points = frame[objective_columns].to_numpy(dtype=float)
    if points.shape[1] != 2:
        return truncate_by_crowding(frame, objective_columns, k, id_column)

    selected: list[int] = []
    remaining = list(range(len(frame)))

    while remaining and len(selected) < k:
        best_index = None
        best_value = -np.inf if indicator == "hv" else np.inf
        for index in remaining:
            candidate = selected + [index]
            candidate_points = points[candidate]
            value = (
                hypervolume_2d(candidate_points)
                if indicator == "hv"
                else r2_indicator(candidate_points)
            )
            is_better = value > best_value if indicator == "hv" else value < best_value
            if is_better:
                best_value = value
                best_index = index
            elif value == best_value and best_index is not None:
                current_key = (points[index].sum(), str(frame.iloc[index][id_column]))
                best_key = (points[best_index].sum(), str(frame.iloc[best_index][id_column]))
                if current_key < best_key:
                    best_index = index
        if best_index is None:
            break
        selected.append(best_index)
        remaining.remove(best_index)

    return _finish_selection(frame, selected, objective_columns, k, id_column)
