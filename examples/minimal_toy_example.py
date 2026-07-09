from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from monas_archiving.config import load_config
from monas_archiving.pipeline import run_pipeline


def make_toy_data(path: Path) -> None:
    """Create a deterministic synthetic objective space."""
    rows = []
    for index in range(18):
        params = 5000 + index * 1400
        psnr = 31.0 + 0.18 * index - 0.006 * max(0, index - 11) ** 2
        if index in {5, 11, 16}:
            params -= 2200
        rows.append(
            {
                "architecture_id": f"toy_{index:03d}",
                "chromosome": str([index % 8, (index * 3) % 8, (index * 5) % 8]),
                "psnr": round(psnr, 4),
                "params": float(params),
                "model": "synthetic",
                "seed": 1,
                "generation": index,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_config(input_csv: Path, output_dir: Path) -> Path:
    config_path = output_dir / "toy_config.yaml"
    config = {
        "run_name": "minimal_toy_example",
        "input_path": str(input_csv),
        "output_dir": str(output_dir.parent),
        "architecture_id_column": "architecture_id",
        "chromosome_column": "chromosome",
        "deduplication_key": "architecture_id",
        "seed": 11,
        "objectives": [
            {"column": "psnr", "direction": "maximize"},
            {"column": "params", "direction": "minimize"},
        ],
        "archivers": [
            {"name": "pq"},
            {"name": "crowding"},
            {"name": "grid", "bins": 4},
            {"name": "epsilon", "eps": 0.2},
            {"name": "tight1"},
            {"name": "kmeans", "iters": 20},
            {"name": "entropy", "bins": 4},
            {"name": "hv"},
            {"name": "r2"},
        ],
        "truncation_sizes": [4],
        "indicators": ["igd_plus", "hypervolume", "r2", "epsilon", "hausdorff"],
        "hv_reference_point": [1.1, 1.1],
        "plot": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal toy MoNAS archiving example.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "examples",
        help="Directory where example outputs will be written.",
    )
    args = parser.parse_args()

    run_root = args.output_dir / "minimal_toy_example"
    input_csv = run_root / "toy_input.csv"
    make_toy_data(input_csv)
    config_path = write_config(input_csv, run_root)
    config = load_config(config_path)
    run_pipeline(config)

    metrics = pd.read_csv(config.run_dir / "metrics" / "archive_metrics.csv")
    print(metrics[["archiver", "k", "indicator", "value", "direction"]].to_string(index=False))
    print(f"Outputs written to {config.run_dir}")


if __name__ == "__main__":
    main()
