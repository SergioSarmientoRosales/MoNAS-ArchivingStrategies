from __future__ import annotations

import math

import numpy as np

from monas_archiving.pareto import nondominated_mask


INDICATOR_DIRECTIONS = {
    "igd_plus": "lower_is_better",
    "hypervolume": "higher_is_better",
    "r2": "lower_is_better",
    "epsilon": "lower_is_better",
    "hausdorff": "lower_is_better",
}


def _as_points(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2:
        raise ValueError("Indicator inputs must be 2D objective matrices.")
    return array


def igd_plus(reference: np.ndarray, approximation: np.ndarray) -> float:
    """IGD+ for minimization. Lower values are better."""
    ref = _as_points(reference)
    approx = _as_points(approximation)
    if len(ref) == 0:
        raise ValueError("IGD+ reference set is empty.")
    if len(approx) == 0:
        return float("inf")

    distances: list[float] = []
    for r in ref:
        diff = np.maximum(approx - r, 0.0)
        distances.append(float(np.min(np.linalg.norm(diff, axis=1))))
    return float(np.mean(distances))


def hypervolume_2d(points: np.ndarray, reference_point: tuple[float, float] = (1.1, 1.1)) -> float:
    """Exact 2D hypervolume for minimization. Higher values are better."""
    pts = _as_points(points)
    if pts.shape[1] != 2:
        raise ValueError("hypervolume_2d supports exactly two objectives.")
    if len(pts) == 0:
        return 0.0

    nd = pts[nondominated_mask(pts)]
    rx, ry = reference_point
    if np.any(nd[:, 0] >= rx) or np.any(nd[:, 1] >= ry):
        return 0.0

    order = np.argsort(nd[:, 0], kind="mergesort")
    sorted_points = nd[order]
    hv = 0.0
    previous_y = ry
    for x, y in sorted_points:
        width = rx - x
        height = previous_y - y
        if width > 0 and height > 0:
            hv += width * height
        previous_y = min(previous_y, y)
    return float(hv)


def generate_weights(n_weights: int = 101, eps: float = 1e-6) -> np.ndarray:
    """Generate strictly positive 2D weight vectors for R2."""
    if n_weights <= 0:
        raise ValueError("n_weights must be positive.")
    if n_weights == 1:
        return np.array([[0.5, 0.5]], dtype=float)
    values = np.linspace(eps, 1.0 - eps, n_weights)
    return np.column_stack([values, 1.0 - values])


def r2_indicator(points: np.ndarray, weights: np.ndarray | None = None) -> float:
    """R2 indicator with weighted Tchebycheff ASF for normalized minimization."""
    pts = _as_points(points)
    if pts.shape[1] != 2:
        raise ValueError("r2_indicator supports exactly two objectives.")
    if len(pts) == 0:
        return float("inf")
    if weights is None:
        weights = generate_weights(101)

    values = []
    for weight in weights:
        scalarized = np.max(weight * pts, axis=1)
        values.append(float(np.min(scalarized)))
    return float(np.mean(values))


def additive_epsilon(reference: np.ndarray, approximation: np.ndarray) -> float:
    """Additive epsilon indicator for minimization. Lower values are better."""
    ref = _as_points(reference)
    approx = _as_points(approximation)
    if len(ref) == 0:
        raise ValueError("epsilon reference set is empty.")
    if len(approx) == 0:
        return float("inf")

    epsilon = -math.inf
    for r in ref:
        epsilon_for_ref = math.inf
        for a in approx:
            epsilon_for_ref = min(epsilon_for_ref, float(np.max(a - r)))
        epsilon = max(epsilon, epsilon_for_ref)
    return float(epsilon)


def hausdorff_distance(reference: np.ndarray, approximation: np.ndarray) -> float:
    """Symmetric Hausdorff distance. Lower values are better."""
    ref = _as_points(reference)
    approx = _as_points(approximation)
    if len(ref) == 0:
        raise ValueError("Hausdorff reference set is empty.")
    if len(approx) == 0:
        return float("inf")

    def directed(a: np.ndarray, b: np.ndarray) -> float:
        distances = []
        for point in a:
            distances.append(float(np.min(np.linalg.norm(b - point, axis=1))))
        return max(distances)

    return float(max(directed(ref, approx), directed(approx, ref)))


def evaluate_indicators(
    reference: np.ndarray,
    approximation: np.ndarray,
    indicators: tuple[str, ...],
    hv_reference_point: tuple[float, ...] | None = None,
) -> dict[str, float]:
    """Evaluate selected indicators on normalized minimization objectives."""
    results: dict[str, float] = {}
    for indicator in indicators:
        name = indicator.lower()
        if name == "igd_plus":
            results[name] = igd_plus(reference, approximation)
        elif name == "hypervolume":
            ref_point = tuple(hv_reference_point or (1.1, 1.1))
            if len(ref_point) != 2:
                raise ValueError("Only 2D hypervolume is currently supported.")
            results[name] = hypervolume_2d(approximation, reference_point=(ref_point[0], ref_point[1]))
        elif name == "r2":
            results[name] = r2_indicator(approximation)
        elif name == "epsilon":
            results[name] = additive_epsilon(reference, approximation)
        elif name == "hausdorff":
            results[name] = hausdorff_distance(reference, approximation)
        else:
            raise ValueError(f"Unknown indicator: {indicator}")
    return results
