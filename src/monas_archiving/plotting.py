from __future__ import annotations

from pathlib import Path

import pandas as pd


def _load_pyplot(output_path: Path):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        note_path = output_path.parent / "plotting_skipped.txt"
        note_path.write_text(
            "Plotting skipped because matplotlib is not importable in this environment.\n"
            "Install dependencies with `pip install -r requirements.txt` in a clean environment "
            "to generate PNG figures.\n",
            encoding="utf-8",
        )
        return None
    return plt


def plot_objective_scatter(
    solution_cloud: pd.DataFrame,
    reference_front: pd.DataFrame,
    objective_columns: list[str],
    output_path: str | Path,
) -> None:
    """Plot solution cloud and reference front for two normalized objectives."""
    if len(objective_columns) != 2:
        return

    output = Path(output_path)
    plt = _load_pyplot(output)
    if plt is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    x_col, y_col = objective_columns

    plt.figure(figsize=(7, 5))
    plt.scatter(solution_cloud[x_col], solution_cloud[y_col], s=20, alpha=0.35, label="Solution cloud")
    plt.scatter(reference_front[x_col], reference_front[y_col], s=34, label="Reference front")
    plt.xlabel(f"{x_col} (0 is best)")
    plt.ylabel(f"{y_col} (0 is best)")
    plt.title("Normalized solution cloud and reference front")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def plot_metric_bars(metrics: pd.DataFrame, output_path: str | Path) -> None:
    """Create one bar plot per indicator value table."""
    if metrics.empty:
        return

    output = Path(output_path)
    plt = _load_pyplot(output)
    if plt is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)

    indicators = list(metrics["indicator"].drop_duplicates())
    fig, axes = plt.subplots(len(indicators), 1, figsize=(9, max(3, 3 * len(indicators))))
    if len(indicators) == 1:
        axes = [axes]

    for axis, indicator in zip(axes, indicators):
        subset = metrics[metrics["indicator"] == indicator].copy()
        subset["label"] = subset["archiver"] + " k=" + subset["k"].astype(str)
        axis.bar(subset["label"], subset["value"])
        axis.set_title(f"{indicator} ({subset['direction'].iloc[0]})")
        axis.tick_params(axis="x", labelrotation=45)
        axis.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
