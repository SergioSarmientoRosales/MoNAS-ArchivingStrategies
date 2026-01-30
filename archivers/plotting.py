# plotting.py
from __future__ import annotations

import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from records import NormRecord


# ----------------------------
# Optional LOWESS dependency
# ----------------------------
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
except Exception:
    _lowess = None


# ----------------------------
# Helpers: points extraction
# ----------------------------
def xy_from_normrecord(nr: NormRecord, use_normalized: bool) -> Tuple[float, float]:
    """
    Return (x, y) for plotting.
    X is -PSNR (or -PSNR_N), Y is Params (or Params_N).
    """
    if use_normalized:
        x = -float(nr.psnr_n)
        y = float(nr.params_n)
    else:
        x = -float(nr.base.psnr)
        y = float(nr.base.params)
    return x, y


def front_to_xy(front: List[NormRecord], use_normalized: bool) -> Tuple[np.ndarray, np.ndarray]:
    pts = [xy_from_normrecord(nr, use_normalized) for nr in front]
    if not pts:
        return np.array([]), np.array([])
    pts.sort(key=lambda p: p[0])  # sort by x
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    return xs, ys


def cloud_to_xy(all_points: List[NormRecord], use_normalized: bool) -> Tuple[np.ndarray, np.ndarray]:
    pts = [xy_from_normrecord(nr, use_normalized) for nr in all_points]
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    return xs, ys


# ----------------------------
# Trend line smoothing
# ----------------------------
def trend_lowess(xs: np.ndarray, ys: np.ndarray, frac: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
    """
    LOWESS trend line. Requires statsmodels.
    xs must be sorted.
    """
    if _lowess is None or xs.size < 3:
        return xs, ys
    sm = _lowess(ys, xs, frac=frac, return_sorted=True)
    return sm[:, 0].astype(float), sm[:, 1].astype(float)


def trend_moving_average(xs: np.ndarray, ys: np.ndarray, window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dependency-free fallback smoothing. xs must be sorted.
    """
    n = xs.size
    if n < 3 or window <= 1:
        return xs, ys

    window = min(window, n if n % 2 == 1 else n - 1)
    if window < 3:
        return xs, ys

    k = window // 2
    tx = xs.copy()
    ty = ys.copy()

    for i in range(n):
        lo = max(0, i - k)
        hi = min(n, i + k + 1)
        ty[i] = float(np.mean(ys[lo:hi]))
    return tx, ty


def trend_line(xs: np.ndarray, ys: np.ndarray,
               method: str = "lowess",
               lowess_frac: float = 0.4,
               ma_window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unified trend line API.
    method:
      - "lowess" (preferred, if statsmodels is available)
      - "ma" (moving average fallback)
      - "none" (no smoothing)
    """
    if xs.size == 0:
        return xs, ys
    if method == "none":
        return xs, ys
    if method == "lowess":
        if _lowess is not None:
            return trend_lowess(xs, ys, frac=lowess_frac)
        return trend_moving_average(xs, ys, window=ma_window)
    if method == "ma":
        return trend_moving_average(xs, ys, window=ma_window)
    raise ValueError(f"Unknown trend method: {method}")


# ----------------------------
# Plotting: GLOBAL figures
# ----------------------------
def plot_cloud_and_all_fronts(
    all_points: List[NormRecord],
    fronts_by_method: Dict[str, List[NormRecord]],
    out_path: str,
    title: str,
    use_normalized: bool = False,
    trend: str = "lowess",
    lowess_frac: float = 0.4,
    ma_window: int = 5,
    show_front_points: bool = True,
) -> None:
    """
    One plot: cloud + trend line per archiver.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure()

    # Cloud
    cx, cy = cloud_to_xy(all_points, use_normalized)
    plt.scatter(cx, cy, s=8, alpha=0.15, label="Cloud")

    # Front trend lines
    for method, front in fronts_by_method.items():
        fx, fy = front_to_xy(front, use_normalized)
        if fx.size == 0:
            continue

        MIN_POINTS_FOR_SMOOTH = 12

        if fx.size >= MIN_POINTS_FOR_SMOOTH and trend != "none":
            tx, ty = trend_line(fx, fy, method=trend, lowess_frac=lowess_frac, ma_window=ma_window)
        else:
            tx, ty = fx, fy  # polyline for sparse fronts

        plt.plot(tx, ty, linewidth=2.5, label=method)

        if show_front_points:
            plt.scatter(fx, fy, s=14, alpha=0.18)

    plt.title(title)
    plt.xlabel("-PSNR" if not use_normalized else "-PSNR_N")
    plt.ylabel("Params" if not use_normalized else "Params_N")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_all_fronts_only(
    fronts_by_method: Dict[str, List[NormRecord]],
    out_path: str,
    title: str,
    use_normalized: bool = False,
    trend: str = "lowess",
    lowess_frac: float = 0.4,
    ma_window: int = 5,
    show_front_points: bool = True,
) -> None:
    """
    One plot: only fronts, each as a trend line.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure()

    for method, front in fronts_by_method.items():
        fx, fy = front_to_xy(front, use_normalized)
        if fx.size == 0:
            continue

        MIN_POINTS_FOR_SMOOTH = 12

        if fx.size >= MIN_POINTS_FOR_SMOOTH and trend != "none":
            tx, ty = trend_line(fx, fy, method=trend, lowess_frac=lowess_frac, ma_window=ma_window)
        else:
            tx, ty = fx, fy  # polyline for sparse fronts

        plt.plot(tx, ty, linewidth=2.5, label=method)
        plt.scatter(fx, fy, s=12, alpha=0.15)

        if show_front_points:
            plt.scatter(fx, fy, s=14, alpha=0.18)

    plt.title(title)
    plt.xlabel("-PSNR" if not use_normalized else "-PSNR_N")
    plt.ylabel("Params" if not use_normalized else "Params_N")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ----------------------------
# Plotting: per-method figures
# ----------------------------
def plot_cloud_and_front(
    all_points: List[NormRecord],
    front: List[NormRecord],
    out_path: str,
    title: str,
    use_normalized: bool = False,
    trend: str = "lowess",
    lowess_frac: float = 0.4,
    ma_window: int = 5,
    show_front_points: bool = True,
) -> None:
    """
    One plot: cloud + one front trend line (single method).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure()

    cx, cy = cloud_to_xy(all_points, use_normalized)
    plt.scatter(cx, cy, s=8, alpha=0.15, label="Cloud")

    fx, fy = front_to_xy(front, use_normalized)
    if fx.size > 0:
        tx, ty = trend_line(fx, fy, method=trend, lowess_frac=lowess_frac, ma_window=ma_window)
        plt.plot(tx, ty, linewidth=2.5, label="Front")
        if show_front_points:
            plt.scatter(fx, fy, s=18, alpha=0.25)

    plt.title(title)
    plt.xlabel("-PSNR" if not use_normalized else "-PSNR_N")
    plt.ylabel("Params" if not use_normalized else "Params_N")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_front_only(
    front: List[NormRecord],
    out_path: str,
    title: str,
    use_normalized: bool = False,
    trend: str = "lowess",
    lowess_frac: float = 0.4,
    ma_window: int = 5,
    show_front_points: bool = True,
) -> None:
    """
    One plot: one front only (trend line).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure()

    fx, fy = front_to_xy(front, use_normalized)
    if fx.size > 0:
        tx, ty = trend_line(fx, fy, method=trend, lowess_frac=lowess_frac, ma_window=ma_window)
        plt.plot(tx, ty, linewidth=2.5, label="Front")
        if show_front_points:
            plt.scatter(fx, fy, s=18, alpha=0.25)

    plt.title(title)
    plt.xlabel("-PSNR" if not use_normalized else "-PSNR_N")
    plt.ylabel("Params" if not use_normalized else "Params_N")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
