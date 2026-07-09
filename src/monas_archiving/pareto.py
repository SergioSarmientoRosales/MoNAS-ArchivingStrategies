from __future__ import annotations

import numpy as np
import pandas as pd


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True when minimization point a Pareto-dominates b."""
    return bool(np.all(a <= b) and np.any(a < b))


def nondominated_mask(points: np.ndarray) -> np.ndarray:
    """Compute a deterministic O(n^2) non-dominated mask for minimization."""
    if points.ndim != 2:
        raise ValueError("points must be a 2D objective matrix.")
    n_points = points.shape[0]
    keep = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        if not keep[i]:
            continue
        for j in range(n_points):
            if i == j:
                continue
            if dominates(points[j], points[i]):
                keep[i] = False
                break
    return keep


def nondominated_frame(frame: pd.DataFrame, objective_columns: list[str]) -> pd.DataFrame:
    """Return the non-dominated subset of a data frame."""
    points = frame[objective_columns].to_numpy(dtype=float)
    return frame.loc[nondominated_mask(points)].reset_index(drop=True)


def crowding_distance(points: np.ndarray) -> np.ndarray:
    """NSGA-II style crowding distance for minimization objectives."""
    if points.ndim != 2:
        raise ValueError("points must be a 2D objective matrix.")
    n_points, n_objectives = points.shape
    distances = np.zeros(n_points, dtype=float)
    if n_points == 0:
        return distances
    if n_points <= 2:
        distances[:] = np.inf
        return distances

    for objective_index in range(n_objectives):
        order = np.argsort(points[:, objective_index], kind="mergesort")
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        lo = points[order[0], objective_index]
        hi = points[order[-1], objective_index]
        span = hi - lo
        if span == 0:
            continue
        for rank in range(1, n_points - 1):
            prev_value = points[order[rank - 1], objective_index]
            next_value = points[order[rank + 1], objective_index]
            distances[order[rank]] += (next_value - prev_value) / span
    return distances


def truncate_by_crowding(
    frame: pd.DataFrame,
    objective_columns: list[str],
    k: int,
    id_column: str,
) -> pd.DataFrame:
    """Keep k diverse rows using crowding distance and deterministic ties."""
    if k <= 0:
        raise ValueError("k must be positive.")
    if len(frame) <= k:
        return frame.reset_index(drop=True)

    points = frame[objective_columns].to_numpy(dtype=float)
    distances = crowding_distance(points)
    working = frame.copy()
    working["_crowding_distance"] = distances
    working["_objective_sum"] = working[objective_columns].sum(axis=1)
    working = working.sort_values(
        ["_crowding_distance", "_objective_sum", id_column],
        ascending=[False, True, True],
        kind="mergesort",
    )
    return working.head(k).drop(columns=["_crowding_distance", "_objective_sum"]).reset_index(drop=True)
