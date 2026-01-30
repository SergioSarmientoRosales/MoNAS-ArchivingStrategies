# plot_cloud_and_baseline_from_brain_outputs.py
# ------------------------------------------------------------
# Standalone script (no project modules required).
#
# Builds the global cloud exactly as in brain.py:
#   1) Load all raw records from ROOT\c_Reduced\<model>\*.csv|*.tsv
#   2) Global deduplication by chromosome (Net)
#   3) Global min-max normalization over the deduped cloud
#   4) Compute the exact Pareto front (PQ) from the normalized cloud
#
# Plots (IEEE-friendly grayscale), all in [0,1]x[0,1]:
#   - cloud_only_normalized.(png/pdf)              (full [0,1] view)
#   - baseline_only_normalized_zoom.(png/pdf)      (fixed window: dx=0.3, dy=0.2)
#   - cloud_plus_baseline_normalized_zoom.(png/pdf) (fixed window: dx=0.3, dy=0.2)
#
# IMPORTANT:
#   We plot x = 1 - PSNR_N (so x in [0,1]) and y = Params_N (in [0,1]).
#   The zoom window is centered on the PQ front bounding box center.
# ------------------------------------------------------------

from __future__ import annotations

import os
import csv
import math
from dataclasses import dataclass
from typing import List, Iterable, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt


# -----------------------------
# CONFIG
# -----------------------------
ROOT = r"G:\Mi unidad\Paper2\Seeds"
IN_ARCHIVE = os.path.join(ROOT, "mse_Reduced")
MODEL_DIRS = ["ensemble", "gbrt", "gpr", "knn", "rf", "svr", "xgb", "et"]

OUT_DIR = os.path.join(ROOT, "plots_from_cloud", "GLOBAL")
os.makedirs(OUT_DIR, exist_ok=True)

SAVE_FORMATS = ("png", "pdf")  # change to ("pdf",) for vector-only

# Fixed zoom window for non-dominated front
ZOOM_DX = 0.25
ZOOM_DY = 0.025


# -----------------------------
# IEEE-ish style
# -----------------------------
def set_ieee_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
    })


def save_multi(fig, out_base: str) -> None:
    for ext in SAVE_FORMATS:
        fig.savefig(f"{out_base}.{ext}", bbox_inches="tight")
    plt.close(fig)


def ieee_axes_clean(ax) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)


# -----------------------------
# Data model
# -----------------------------
@dataclass(frozen=True)
class Rec:
    net: str
    psnr: float
    params: float
    model: str
    seed: str
    source_file: str


# -----------------------------
# I/O
# -----------------------------
def iter_result_files(root: str, exts: Tuple[str, ...] = (".csv", ".tsv")) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.lower().endswith(exts):
                yield os.path.join(dirpath, fn)


def sniff_delimiter(path: str) -> str:
    if path.lower().endswith(".tsv"):
        return "\t"
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            sample = f.read(4096)
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";"])
        return dialect.delimiter
    except Exception:
        return ","


def read_records_from_file(path: str, model: str) -> List[Rec]:
    delim = sniff_delimiter(path)
    out: List[Rec] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delim)
        if not reader.fieldnames:
            return out

        seed_guess = ""
        base = os.path.basename(path)
        for token in base.replace("-", "_").split("_"):
            if token.lower().startswith("seed"):
                seed_guess = token
                break

        for r in reader:
            net = (r.get("Net") or r.get("net") or "").strip()
            if not net:
                continue
            try:
                psnr = float(r.get("PSNR") or r.get("psnr"))
                params = float(r.get("Params") or r.get("params"))
            except Exception:
                continue

            if not (math.isfinite(psnr) and math.isfinite(params)):
                continue

            out.append(Rec(
                net=net,
                psnr=psnr,
                params=params,
                model=model,
                seed=seed_guess,
                source_file=path,
            ))
    return out


def load_all_records() -> List[Rec]:
    all_recs: List[Rec] = []
    for model in MODEL_DIRS:
        model_path = os.path.join(IN_ARCHIVE, model)
        if not os.path.isdir(model_path):
            print(f"[WARN] Missing model directory: {model_path}")
            continue

        for fp in iter_result_files(model_path, exts=(".csv", ".tsv")):
            all_recs.extend(read_records_from_file(fp, model=model))

    print(f"[INFO] Loaded records: {len(all_recs)}")
    return all_recs


# -----------------------------
# Dedup + normalization
# -----------------------------
def deduplicate_by_net(records: List[Rec]) -> List[Rec]:
    seen: set[str] = set()
    out: List[Rec] = []
    for r in records:
        if r.net in seen:
            continue
        seen.add(r.net)
        out.append(r)
    print(f"[INFO] After global dedup by Net: {len(out)}")
    return out


def normalize(records: List[Rec]) -> List[Tuple[float, float, Rec]]:
    psnrs = [r.psnr for r in records]
    params = [r.params for r in records]
    psnr_min, psnr_max = min(psnrs), max(psnrs)
    par_min, par_max = min(params), max(params)

    d_psnr = psnr_max - psnr_min
    d_par = par_max - par_min
    if d_psnr == 0.0 or d_par == 0.0:
        raise ValueError("Degenerate min-max range (all PSNR or Params equal).")

    out: List[Tuple[float, float, Rec]] = []
    for r in records:
        psnr_n = (r.psnr - psnr_min) / d_psnr
        par_n = (r.params - par_min) / d_par
        out.append((psnr_n, par_n, r))

    ps = [p[0] for p in out]
    pa = [p[1] for p in out]
    print(f"[DEBUG] PSNR_N range:    {min(ps):.6f} .. {max(ps):.6f}")
    print(f"[DEBUG] Params_N range:  {min(pa):.6f} .. {max(pa):.6f}")
    print("[INFO] Global normalization computed from deduped cloud.")
    return out


# -----------------------------
# Exact Pareto front (2D scan) for:
#   u = Params_N (min)
#   v = 1 - PSNR_N (min)
# -----------------------------
def pareto_front(points: List[Tuple[float, float, Rec]]) -> List[Tuple[float, float, Rec]]:
    uv = [(p[1], 1.0 - p[0], p) for p in points]  # (u, v, payload)
    uv_sorted = sorted(uv, key=lambda t: (t[0], t[1]))

    front_payload: List[Tuple[float, float, Rec]] = []
    best_v = float("inf")
    for u, v, payload in uv_sorted:
        if v < best_v:
            front_payload.append(payload)
            best_v = v

    print(f"[INFO] Exact Pareto front size (from cloud): {len(front_payload)}")
    return front_payload


# -----------------------------
# Fixed zoom window limits (dx, dy) centered on front center
# -----------------------------
def fixed_window_limits(x: List[float], y: List[float], dx: float, dy: float) -> Tuple[float, float, float, float]:
    if not x or not y:
        return 0.0, 1.0, 0.0, 1.0

    cx = 0.5 * (min(x) + max(x))
    cy = 0.5 * (min(y) + max(y))

    xmin = cx - dx / 2.0
    xmax = cx + dx / 2.0
    ymin = cy - dy / 2.0
    ymax = cy + dy / 2.0

    # clip to [0,1]
    xmin = max(0.0, xmin)
    ymin = max(0.0, ymin)
    xmax = min(1.0, xmax)
    ymax = min(1.0, ymax)

    # If clipped window collapses (front near boundary), re-expand if possible
    if (xmax - xmin) < dx and xmin == 0.0:
        xmax = min(1.0, xmin + dx)
    if (xmax - xmin) < dx and xmax == 1.0:
        xmin = max(0.0, xmax - dx)
    if (ymax - ymin) < dy and ymin == 0.0:
        ymax = min(1.0, ymin + dy)
    if (ymax - ymin) < dy and ymax == 1.0:
        ymin = max(0.0, ymax - dy)

    return xmin, xmax, ymin, ymax


# -----------------------------
# Plotting in [0,1]x[0,1]:
#   x = 1 - PSNR_N
#   y = Params_N
# -----------------------------
def plot_cloud_only(points: List[Tuple[float, float, Rec]], out_base: str) -> None:
    fig = plt.figure(figsize=(5.2, 4.2))
    ax = fig.add_subplot(111)

    x = [1.0 - p[0] for p in points]   # 1 - PSNR_N
    y = [p[1] for p in points]         # Params_N

    ax.scatter(x, y, s=1, alpha=0.06, color="0.35", rasterized=True)
    ax.set_xlabel("Normalized PSNR ")
    ax.set_ylabel("Normalized Parameters")
    #  ax.set_title("Global unique solution cloud (normalized)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ieee_axes_clean(ax)

    fig.tight_layout()
    save_multi(fig, out_base)


def plot_front_only_zoom(front: List[Tuple[float, float, Rec]], out_base: str, dx: float, dy: float) -> None:
    fig = plt.figure(figsize=(5.2, 4.2))
    ax = fig.add_subplot(111)

    x = [1.0 - p[0] for p in front]  # 1 - PSNR_N
    y = [p[1] for p in front]        # Params_N

    pts = sorted(zip(x, y), key=lambda t: (t[0], t[1]))
    x = [t[0] for t in pts]
    y = [t[1] for t in pts]

    ax.scatter(x, y, s=55, facecolors="none", edgecolors="black", linewidths=1.3)
    ax.plot(x, y, linestyle=":", linewidth=2.2, color="black")

    xmin, xmax, ymin, ymax = fixed_window_limits(x, y, dx=dx, dy=dy)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel("Normalized Inverted PSNR")
    ax.set_ylabel("Normalized Parameters ")
    #  ax.set_title(f"PQ reference front (zoom: Δx={dx}, Δy={dy})")
    ieee_axes_clean(ax)

    fig.tight_layout()
    save_multi(fig, out_base)


def plot_cloud_plus_front_zoom(
    points: List[Tuple[float, float, Rec]],
    front: List[Tuple[float, float, Rec]],
    out_base: str,
    dx: float,
    dy: float,
) -> None:
    fig = plt.figure(figsize=(5.2, 4.2))
    ax = fig.add_subplot(111)

    cx = [1.0 - p[0] for p in points]
    cy = [p[1] for p in points]
    ax.scatter(cx, cy, s=1, alpha=0.05, color="0.45", rasterized=True)

    fx = [1.0 - p[0] for p in front]
    fy = [p[1] for p in front]
    pts = sorted(zip(fx, fy), key=lambda t: (t[0], t[1]))
    fx = [t[0] for t in pts]
    fy = [t[1] for t in pts]

    ax.scatter(fx, fy, s=55, facecolors="none", edgecolors="black", linewidths=1.3, zorder=3)
    ax.plot(fx, fy, linestyle=":", linewidth=2.2, color="black", zorder=2)

    xmin, xmax, ymin, ymax = fixed_window_limits(fx, fy, dx=dx, dy=dy)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel("Normalized Inverted PSNR (1 - PSNR$_N$)")
    ax.set_ylabel("Normalized Parameters (Params$_N$)")
    ax.set_title(f"Cloud + PQ reference (zoom: Δx={dx}, Δy={dy})")
    ieee_axes_clean(ax)

    fig.tight_layout()
    save_multi(fig, out_base)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    set_ieee_style()

    records = load_all_records()
    if not records:
        raise RuntimeError("No records loaded. Check IN_ARCHIVE and file formats/columns (Net, PSNR, Params).")

    records = deduplicate_by_net(records)

    cloud_norm = normalize(records)     # list[(psnr_n, params_n, rec)]
    pq_front = pareto_front(cloud_norm) # exact PQ from normalized cloud

    # Full cloud view
    plot_cloud_only(cloud_norm, os.path.join(OUT_DIR, "cloud_only_normalized"))
    print("[DONE] cloud_only_normalized.(png/pdf)")

    # Zoomed non-dominated front (fixed window)
    plot_front_only_zoom(
        pq_front,
        os.path.join(OUT_DIR, "baseline_only_normalized_zoom"),
        dx=ZOOM_DX,
        dy=ZOOM_DY,
    )
    print("[DONE] baseline_only_normalized_zoom.(png/pdf)")

    # Zoomed overlay
    plot_cloud_plus_front_zoom(
        cloud_norm,
        pq_front,
        os.path.join(OUT_DIR, "cloud_plus_baseline_normalized_zoom"),
        dx=ZOOM_DX,
        dy=ZOOM_DY,
    )
    print("[DONE] cloud_plus_baseline_normalized_zoom.(png/pdf)")

    print("[DONE] Outputs written under:", OUT_DIR)


if __name__ == "__main__":
    main()
