from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable, Tuple


@dataclass(frozen=True)
class Point2D:
    """
    2D point for indicators in MINIMIZATION space.

    Attributes:
        f1: First objective value (to minimize)  -> Params_N
        f2: Second objective value (to minimize) -> -PSNR_N (equivalent to maximize PSNR_N)
        net: Optional network identifier (string)
    """
    f1: float
    f2: float
    net: Optional[str] = None

    def __repr__(self) -> str:
        if self.net:
            return f"Point2D(f1={self.f1:.6f}, f2={self.f2:.6f}, net={self.net})"
        return f"Point2D(f1={self.f1:.6f}, f2={self.f2:.6f})"


def read_front_csv(path: str) -> List[Dict[str, str]]:
    """
    Read a pareto_global.csv produced by the archiving pipeline.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise csv.Error(f"CSV has no header/fieldnames: {path}")

        for row in reader:
            # Keep as raw strings; downstream converts robustly.
            rows.append(row)
    return rows


def _to_float(value: object, *, field: str, row_idx: int, path_hint: str = "") -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception as e:
        hint = f" ({path_hint})" if path_hint else ""
        raise ValueError(f"Row {row_idx}: cannot parse {field}={value!r} as float{hint}: {e}") from e


def rows_to_min_points(
    rows: List[Dict[str, str]],
    *,
    params_field: str = "Params_N",
    psnr_field: str = "PSNR_N",
    net_field: str = "Net",
    path_hint: str = "",
    strict_finite: bool = True,
    warn_out_of_range: bool = True,
    expected_range: Tuple[float, float] = (0.0, 1.0),
    eps: float = 1e-6,
) -> List[Point2D]:
    """
    Convert CSV rows to Point2D objects in minimization space.

    Transform:
        f1 = Params_N
        f2 = -PSNR_N

    Notes:
      - Assumes Params_N and PSNR_N are globally normalized (typically ~[0,1]).
      - Optional warnings if values are outside expected_range (does not stop execution).
    """
    lo, hi = expected_range
    points: List[Point2D] = []

    for i, row in enumerate(rows):
        if params_field not in row:
            raise KeyError(f"Row {i}: Missing required column {params_field!r}{' ('+path_hint+')' if path_hint else ''}")
        if psnr_field not in row:
            raise KeyError(f"Row {i}: Missing required column {psnr_field!r}{' ('+path_hint+')' if path_hint else ''}")

        params_n = _to_float(row.get(params_field), field=params_field, row_idx=i, path_hint=path_hint)
        psnr_n = _to_float(row.get(psnr_field), field=psnr_field, row_idx=i, path_hint=path_hint)

        if strict_finite:
            if not (params_n == params_n and psnr_n == psnr_n):  # NaN check
                raise ValueError(f"Row {i}: NaN detected in {params_field} or {psnr_field}{' ('+path_hint+')' if path_hint else ''}")
            if params_n in (float("inf"), float("-inf")) or psnr_n in (float("inf"), float("-inf")):
                raise ValueError(f"Row {i}: inf detected in {params_field} or {psnr_field}{' ('+path_hint+')' if path_hint else ''}")

        if warn_out_of_range:
            if params_n < lo - eps or params_n > hi + eps:
                print(f"[WARN] {path_hint or 'CSV'} row {i}: {params_field}={params_n:.6f} outside expected [{lo},{hi}]")
            if psnr_n < lo - eps or psnr_n > hi + eps:
                print(f"[WARN] {path_hint or 'CSV'} row {i}: {psnr_field}={psnr_n:.6f} outside expected [{lo},{hi}]")

        net = row.get(net_field) if net_field in row else None
        points.append(Point2D(f1=params_n, f2=-psnr_n, net=net))

    return points


def rows_to_max_points(
    rows: List[Dict[str, str]],
    *,
    params_field: str = "Params_N",
    psnr_field: str = "PSNR_N",
    net_field: str = "Net",
    path_hint: str = "",
    strict_finite: bool = True,
) -> List[Point2D]:
    """
    Convert CSV rows to Point2D keeping PSNR positive (useful for plotting only).

    Transform:
        f1 = Params_N
        f2 = PSNR_N
    """
    points: List[Point2D] = []
    for i, row in enumerate(rows):
        if params_field not in row or psnr_field not in row:
            missing = [k for k in (params_field, psnr_field) if k not in row]
            raise KeyError(f"Row {i}: Missing columns {missing}{' ('+path_hint+')' if path_hint else ''}")

        params_n = _to_float(row.get(params_field), field=params_field, row_idx=i, path_hint=path_hint)
        psnr_n = _to_float(row.get(psnr_field), field=psnr_field, row_idx=i, path_hint=path_hint)

        if strict_finite:
            if not (params_n == params_n and psnr_n == psnr_n):
                raise ValueError(f"Row {i}: NaN detected{ ' ('+path_hint+')' if path_hint else ''}")

        net = row.get(net_field) if net_field in row else None
        points.append(Point2D(f1=params_n, f2=psnr_n, net=net))
    return points


def discover_csv_paths(root_dir: str, *, filename: str = "pareto_global.csv") -> Dict[str, str]:
    """
    Discover method -> csv_path mappings under root_dir.

    Expected structure:
        root_dir/
            methodA/pareto_global.csv
            methodB/pareto_global.csv
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    discovered: Dict[str, str] = {}
    for method in sorted(os.listdir(root_dir)):  # stable ordering
        method_dir = os.path.join(root_dir, method)
        if not os.path.isdir(method_dir):
            continue

        csv_path = os.path.join(method_dir, filename)
        if os.path.isfile(csv_path):
            discovered[method] = csv_path

    return discovered


def load_method_front(method_dir: str, *, minimize: bool = True) -> List[Point2D]:
    csv_path = os.path.join(method_dir, "pareto_global.csv")
    rows = read_front_csv(csv_path)
    if minimize:
        return rows_to_min_points(rows, path_hint=csv_path)
    return rows_to_max_points(rows, path_hint=csv_path)


def load_all_methods(root_dir: str, *, minimize: bool = True) -> Dict[str, List[Point2D]]:
    paths = discover_csv_paths(root_dir)
    fronts: Dict[str, List[Point2D]] = {}
    for method, csv_path in paths.items():
        rows = read_front_csv(csv_path)
        if minimize:
            fronts[method] = rows_to_min_points(rows, path_hint=csv_path)
        else:
            fronts[method] = rows_to_max_points(rows, path_hint=csv_path)
    return fronts


def summarize_points(points: Iterable[Point2D]) -> Tuple[float, float, float, float]:
    """
    Returns (min_f1, max_f1, min_f2, max_f2).
    """
    pts = list(points)
    if not pts:
        raise ValueError("summarize_points: empty point set")

    min_f1 = min(p.f1 for p in pts)
    max_f1 = max(p.f1 for p in pts)
    min_f2 = min(p.f2 for p in pts)
    max_f2 = max(p.f2 for p in pts)
    return (min_f1, max_f1, min_f2, max_f2)
